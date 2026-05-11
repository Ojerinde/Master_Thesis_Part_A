"""
Temporal Convolutional Network (PyTorch)
=========================================
TCN with dilated causal convolutions for GNSS spoofing detection.

Design Rationale:
- Exponentially increasing dilation rates capture multi-scale dependencies
- Causal padding maintains temporal ordering
- Residual connections (with 1×1 projection when channels differ) prevent
  vanishing gradients in deep networks
- More parameter-efficient than RNNs at equivalent receptive field depth

Reference:
- Bai et al. (2018): An Empirical Evaluation of Generic Convolutional
  and Recurrent Networks for Sequence Modelling
"""

from models.deep_learning.base_model import BaseDeepLearningModel
from config.model_configs import get_config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _TemporalBlock(nn.Module):
    """
    Single TCN block: two dilated causal Conv1D + residual connection.

    Causal padding is implemented manually: pad (kernel-1)*dilation on
    the left only, then trim the right after convolution.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout_rate: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation   # causal left-pad

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else None)

    def forward(self, x):
        # x: (batch, channels, length)
        residual = self.residual(x) if self.residual else x

        out = F.pad(x, (self.pad, 0))
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.drop1(out)

        out = F.pad(out, (self.pad, 0))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.drop2(out)

        return F.relu(out + residual)


class _TCNNet(nn.Module):

    def __init__(self, input_dim: int, num_filters: int, kernel_size: int,
                 num_blocks: int, dense_layers: list, dropout_rate: float):
        super().__init__()

        blocks = []
        in_ch = 1
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(_TemporalBlock(in_ch, num_filters,
                                         kernel_size, dilation, dropout_rate))
            in_ch = num_filters
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)      # global average

        dense_seq = []
        prev = num_filters
        for units in dense_layers:
            dense_seq += [nn.Linear(prev, units), nn.ReLU(),
                          nn.Dropout(dropout_rate)]
            prev = units
        dense_seq.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_seq)

    def forward(self, x):
        # x: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.dense(x).squeeze(-1)


class TCNModel(BaseDeepLearningModel):

    def __init__(self, input_dim: int, config=None, custom_config=None):
        super().__init__(input_dim, model_name='tcn')
        self.config = config or custom_config or get_config('tcn')
        self.config['input_dim'] = input_dim
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        cfg = self.config
        self.model = _TCNNet(
            input_dim=self.input_dim,
            num_filters=cfg.get('num_filters', 64),
            kernel_size=cfg.get('kernel_size', 3),
            num_blocks=cfg.get('num_blocks', 4),
            dense_layers=cfg.get('dense_layers', [128, 64]),
            dropout_rate=cfg.get('dropout_rate', 0.2),
        ).to(self.device)
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=None, batch_size=None):
        if self.model is None:
            self.build_model()

        epochs = epochs or self.config.get('epochs', 50)
        batch_size = batch_size or self.config.get('batch_size', 32)
        patience = self.config.get('patience', 10)
        lr = self.config.get('learning_rate', 1e-3)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-7)

        train_dl = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            val_dl = DataLoader(
                TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                              torch.tensor(y_val, dtype=torch.float32)),
                batch_size=batch_size * 4)

        best_val_loss, patience_ctr, best_weights = float('inf'), 0, None

        for _ in range(epochs):
            self.model.train()
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(Xb), yb).backward()
                optimizer.step()

            if not has_val:
                continue

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_dl:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(self.model(Xb), yb).item() * len(Xb)
            val_loss /= len(X_val)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr = 0
                best_weights = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        if best_weights is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_weights.items()})
        self.is_trained = True

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        logits = self.model(
            torch.tensor(X, dtype=torch.float32).to(self.device)
        ).cpu().numpy()
        proba = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - proba, proba])

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(
            torch.load(filepath, map_location=self.device))
        self.is_trained = True

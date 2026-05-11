"""
Bidirectional LSTM Model (PyTorch)
====================================
BiLSTM for GNSS spoofing detection.

Design Rationale:
- Bidirectional processing doubles the effective hidden state,
  learning forward and backward context across GNSS features
- Each BiLSTM layer is a separate nn.LSTM(bidirectional=True, num_layers=1)
  module; BatchNorm1d and explicit Dropout after each layer replace
  PyTorch's inter-layer dropout (which only activates when num_layers > 1)

Note on PyTorch dropout behaviour
----------------------------------
Same as LSTM: always pass dropout=0 to each individual nn.LSTM call
because num_layers=1. Regularisation comes from explicit nn.Dropout layers.

Reference:
- Schuster & Paliwal (1997): Bidirectional recurrent neural networks
"""

from models.deep_learning.base_model import BaseDeepLearningModel
from config.model_configs import get_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _BiLSTMNet(nn.Module):

    def __init__(self, input_dim: int, bilstm_layers: list,
                 dense_layers: list, dropout_rate: float):
        super().__init__()

        self.lstms = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_size = input_dim

        for cfg in bilstm_layers:
            # dropout=0: each nn.LSTM has num_layers=1 so inter-layer
            # dropout is irrelevant and causes a UserWarning if non-zero.
            self.lstms.append(nn.LSTM(
                input_size=in_size,
                hidden_size=cfg['units'],
                batch_first=True,
                bidirectional=True,
                dropout=0,
            ))
            # Output size is 2 * units (forward + backward concatenated)
            self.bns.append(nn.BatchNorm1d(cfg['units'] * 2))
            self.dropouts.append(nn.Dropout(dropout_rate))
            in_size = cfg['units'] * 2

        dense_blocks = []
        prev = in_size
        for units in dense_layers:
            dense_blocks += [nn.Linear(prev, units), nn.ReLU(),
                             nn.Dropout(dropout_rate)]
            prev = units
        dense_blocks.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_blocks)

    def forward(self, x):
        # x: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        for lstm, bn, drop in zip(self.lstms, self.bns, self.dropouts):
            x, _ = lstm(x)               # (batch, seq_len, 2*hidden)
            x = bn(x[:, -1, :])       # last time-step → (batch, 2*hidden)
            x = drop(x).unsqueeze(1)
        x = x.squeeze(1)
        return self.dense(x).squeeze(-1)


class BiLSTMModel(BaseDeepLearningModel):
    """
    Bidirectional LSTM for GNSS spoofing detection.

    Constructor accepts:
        BiLSTMModel(input_dim, config=<dict>)        — from experiment script
        BiLSTMModel(input_dim, custom_config=<dict>) — legacy compatibility
    """

    def __init__(self, input_dim: int, config=None, custom_config=None):
        super().__init__(input_dim, model_name='bilstm')
        self.config = config or custom_config or get_config('bilstm')
        self.config['input_dim'] = input_dim
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        cfg = self.config
        self.model = _BiLSTMNet(
            input_dim=self.input_dim,
            bilstm_layers=cfg['bilstm_layers'],
            dense_layers=cfg['dense_layers'],
            dropout_rate=cfg['dropout_rate'],
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

"""
1D CNN Model (PyTorch)
======================
1D Convolutional Neural Network for GNSS signal feature analysis.

Design Rationale:
- Convolutional filters learn local correlations across GNSS features
- BatchNorm + Dropout for regularisation and stable training
- AdaptiveAvgPool eliminates fixed-size constraint after convolutions
- Early stopping on val_loss with best-weight restoration

Reference:
- LeCun et al. (1998): Convolutional networks for images, speech, and time series
"""

from models.deep_learning.base_model import BaseDeepLearningModel
from config.model_configs import get_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── PyTorch nn.Module ────────────────────────────────────────────────────────

class _CNN1DNet(nn.Module):
    """Internal PyTorch module — not used directly outside this file."""

    def __init__(self, input_dim: int, conv_layers: list,
                 dense_layers: list, dropout_rate: float):
        super().__init__()

        conv_blocks = []
        in_channels = 1          # treat each sample as a 1-channel 1-D signal
        for cfg in conv_layers:
            out_channels = cfg['filters']
            conv_blocks += [
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=cfg['kernel_size'], padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout_rate),
            ]
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)   # output: (batch, channels, 1)

        dense_blocks = []
        prev = in_channels
        for units in dense_layers:
            dense_blocks += [
                nn.Linear(prev, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            prev = units

        dense_blocks.append(nn.Linear(prev, 1))   # binary output
        self.dense = nn.Sequential(*dense_blocks)

    def forward(self, x):
        # x: (batch, features)  →  (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)   # (batch, channels)
        return self.dense(x).squeeze(-1)


# ── Public wrapper (matches BaseDeepLearningModel interface) ─────────────────

class CNN1DModel(BaseDeepLearningModel):
    """
    1D CNN for GNSS spoofing detection.

    Constructor accepts either:
        CNN1DModel(input_dim, config=<dict>)       — from experiment script
        CNN1DModel(input_dim, custom_config=<dict>) — legacy compatibility
    """

    def __init__(self, input_dim: int, config=None, custom_config=None):
        super().__init__(input_dim, model_name='cnn_1d')
        self.config = config or custom_config or get_config('cnn_1d')
        self.config['input_dim'] = input_dim
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        cfg = self.config
        self.model = _CNN1DNet(
            input_dim=self.input_dim,
            conv_layers=cfg['conv_layers'],
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

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            val_ds = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
            val_dl = DataLoader(val_ds, batch_size=batch_size * 4)

        best_val_loss = float('inf')
        patience_ctr = 0
        best_weights = None

        for epoch in range(epochs):
            # ── Training pass ──────────────────────────────────────────────
            self.model.train()
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()

            if not has_val:
                continue

            # ── Validation pass ────────────────────────────────────────────
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
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(Xt).cpu().numpy()
        proba = 1 / (1 + np.exp(-logits))          # sigmoid
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

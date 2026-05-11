"""
CNN-LSTM Hybrid Model (PyTorch)
================================
Combined spatial-temporal architecture for GNSS spoofing detection.

Design Rationale:
- CNN layers extract local feature correlations (spatial patterns)
- LSTM layer models sequential dependencies across extracted features
- Hybrid captures both local patterns and global signal context

Reference:
- Donahue et al. (2015): Long-term recurrent convolutional networks
"""

from models.deep_learning.base_model import BaseDeepLearningModel
from config.model_configs import get_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _CNNLSTMNet(nn.Module):

    def __init__(self, input_dim: int, conv_layers: list, lstm_units: int,
                 dense_layers: list, dropout_rate: float):
        super().__init__()

        # CNN feature extractor
        conv_blocks = []
        in_ch = 1
        for cfg in conv_layers:
            out_ch = cfg['filters']
            conv_blocks += [
                nn.Conv1d(in_ch, out_ch,
                          kernel_size=cfg['kernel_size'], padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
                nn.Dropout(dropout_rate),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_blocks)

        # LSTM temporal modelling
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=lstm_units,
                            batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(lstm_units)

        # Classifier head
        dense_blocks = []
        prev = lstm_units
        for units in dense_layers:
            dense_blocks += [nn.Linear(prev, units), nn.ReLU(),
                             nn.Dropout(dropout_rate)]
            prev = units
        dense_blocks.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_blocks)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)            # (batch, 1, features) — channel dim
        x = self.cnn(x)               # (batch, out_ch, features)
        x = x.permute(0, 2, 1)        # (batch, features, out_ch) — LSTM seq
        x, _ = self.lstm(x)           # (batch, features, lstm_units)
        x = self.bn_lstm(x[:, -1, :])  # last time-step → (batch, lstm_units)
        return self.dense(x).squeeze(-1)


class CNNLSTMModel(BaseDeepLearningModel):

    def __init__(self, input_dim: int, config=None, custom_config=None):
        super().__init__(input_dim, model_name='cnn_lstm')
        self.config = config or custom_config or get_config('cnn_lstm')
        self.config['input_dim'] = input_dim
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        cfg = self.config
        self.model = _CNNLSTMNet(
            input_dim=self.input_dim,
            conv_layers=cfg['conv_layers'],
            lstm_units=cfg['lstm_units'],
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

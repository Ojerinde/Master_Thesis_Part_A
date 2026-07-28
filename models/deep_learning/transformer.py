"""
Transformer Model (PyTorch)
============================
Multi-Head Self-Attention Transformer for GNSS spoofing detection.

Design Rationale:
- Self-attention captures global dependencies across all GNSS features
  simultaneously, unlike RNNs which process sequentially
- Each feature is its own token (FT-Transformer-style per-feature tokenizer:
  Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data,"
  NeurIPS 2021); attention learns which feature combinations are
  discriminative for spoofing detection. There is no temporal sequence to
  tokenize across (each sample is one independent epoch, same as every other
  model in this file), so per-feature tokenization is what gives attention a
  genuine sequence (seq_len = 9) to operate on at all.
- BatchNorm1d on the raw input, before tokenization: nn.Linear-style init
  assumes roughly unit-variance input; MinMaxScaler output (variance << 1)
  violates that and was the direct, verified cause of a collapse to a
  constant predictor (AUC=0.5, identical output regardless of input) on the
  full corpus. BatchNorm1d restores that assumption regardless of whatever
  scaler sits upstream.
- LayerNorm + residual connections enable deep, stable training

Verified head-to-head on the full 209,000-row corpus against a variant with
all 9 features collapsed into a single token (seq_len=1, so self-attention
was a mathematical no-op -- a residual MLP in a Transformer's clothes):
performance was statistically indistinguishable (fragility 95% CIs
overlapped almost entirely), so the choice came down to which one is
actually doing what "Transformer" means. This is that one.

Reference:
- Vaswani et al. (2017): Attention Is All You Need
- Gorishniy, Rubachev, Khrulkov, Babenko (2021): Revisiting Deep Learning
  Models for Tabular Data (FT-Transformer), NeurIPS 2021, arXiv:2106.11959
"""

from models.deep_learning.base_model import BaseDeepLearningModel
from config.model_configs import get_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _FeatureTokenizer(nn.Module):
    """Per-feature affine embedding (FT-Transformer's numerical tokenizer):
    each of the input_dim scalar features gets its own d-dim token,
    T_j = b_j + x_j * W_j, instead of collapsing all features into one
    token. Gives self-attention a genuine multi-token sequence (seq_len =
    input_dim) to learn cross-feature structure over."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, embed_dim))
        self.bias = nn.Parameter(torch.empty(input_dim, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, input_dim, embed_dim)
        return x.unsqueeze(-1) * self.weight + self.bias


class _TransformerBlock(nn.Module):
    """Single Transformer block: MultiHeadAttention + FFN with residuals."""

    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, dropout_rate: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout_rate,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, seq, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class _TransformerNet(nn.Module):

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int,
                 num_blocks: int, mlp_units: list, dropout_rate: float):
        super().__init__()

        embed_dim = ff_dim           # project input to ff_dim
        # nn.Linear's default init assumes roughly unit-variance input.
        # MinMaxScaler output lives in [0,1] (variance << 1), which starves the
        # embedding projection of the input scale its init expects and was the
        # direct cause of a collapse to a constant predictor (AUC=0.5) on the
        # full corpus. BatchNorm1d restores that assumption regardless of
        # whatever scaler sits upstream.
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.tokenizer = _FeatureTokenizer(input_dim, embed_dim)

        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

        mlp_layers = []
        prev = embed_dim
        for units in mlp_units:
            mlp_layers += [nn.Linear(prev, units), nn.ReLU(),
                           nn.Dropout(dropout_rate)]
            prev = units
        mlp_layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # x: (batch, features)
        x = self.input_norm(x)
        x = self.tokenizer(x)              # (batch, features, embed_dim): seq_len = features
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)                  # average pool over the per-feature tokens
        return self.mlp(x).squeeze(-1)


class TransformerModel(BaseDeepLearningModel):

    def __init__(self, input_dim: int, config=None, custom_config=None):
        super().__init__(input_dim, model_name='transformer')
        self.config = config or custom_config or get_config('transformer')
        self.config['input_dim'] = input_dim
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        cfg = self.config
        self.model = _TransformerNet(
            input_dim=self.input_dim,
            num_heads=cfg['num_heads'],
            ff_dim=cfg['ff_dim'],
            num_blocks=cfg['num_transformer_blocks'],
            mlp_units=cfg['mlp_units'],
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

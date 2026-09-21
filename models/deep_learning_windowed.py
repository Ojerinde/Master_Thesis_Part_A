"""
Windowed counterparts of the six deep detectors.
================================================

These mirror models/deep_learning/ exactly in width, depth, dropout and every
other hyperparameter drawn from config/model_configs.py. The single difference
is the axis the architecture traverses.

In the epoch-level track the input is (batch, 9) and each network unsqueezes it,
so the recurrent models run for one step and the convolutional and attention
models slide along the nine-measurement axis -- an ordering that is a convention
of the feature list rather than a physical dimension.

Here the input is (batch, W, 9): the sequence axis is time, the nine
measurements are channels, and the architectures are exercised on the temporal
structure they were designed for. Holding the hyperparameters fixed means a
difference between the two tracks is attributable to the input, which is the
comparison this second track exists to make.
"""
import torch
import torch.nn as nn


class WindowedLSTM(nn.Module):
    def __init__(self, n_feat, lstm_layers, dense_layers, dropout_rate,
                 bidirectional=False):
        super().__init__()
        self.lstms, self.bns, self.drops = (nn.ModuleList() for _ in range(3))
        in_size, mult = n_feat, 2 if bidirectional else 1
        for cfg in lstm_layers:
            self.lstms.append(nn.LSTM(input_size=in_size,
                                      hidden_size=cfg["units"],
                                      batch_first=True, dropout=0,
                                      bidirectional=bidirectional))
            self.bns.append(nn.BatchNorm1d(cfg["units"] * mult))
            self.drops.append(nn.Dropout(dropout_rate))
            in_size = cfg["units"] * mult
        self.head = _mlp(in_size, dense_layers, dropout_rate)

    def forward(self, x):                      # (batch, W, feat)
        for i, (lstm, bn, drop) in enumerate(zip(self.lstms, self.bns, self.drops)):
            x, _ = lstm(x)                     # real recurrence over W steps
            last = i == len(self.lstms) - 1
            if last:
                x = drop(bn(x[:, -1, :]))      # final step
            else:
                b, w, f = x.shape
                x = drop(bn(x.reshape(-1, f)).reshape(b, w, f))
        return self.head(x).squeeze(-1)


class WindowedCNN(nn.Module):
    def __init__(self, n_feat, conv_layers, dense_layers, dropout_rate):
        super().__init__()
        layers, ch = [], n_feat                # features are channels
        for c in conv_layers:
            layers += [nn.Conv1d(ch, c["filters"], c["kernel_size"], padding="same"),
                       nn.BatchNorm1d(c["filters"]), nn.ReLU(), nn.Dropout(dropout_rate)]
            ch = c["filters"]
        self.conv = nn.Sequential(*layers)
        self.head = _mlp(ch, dense_layers, dropout_rate)

    def forward(self, x):                      # (batch, W, feat)
        x = self.conv(x.permute(0, 2, 1))      # -> (batch, ch, W): conv over TIME
        return self.head(x.mean(dim=2)).squeeze(-1)


class WindowedCNNLSTM(nn.Module):
    def __init__(self, n_feat, conv_layers, lstm_units, dense_layers, dropout_rate):
        super().__init__()
        layers, ch = [], n_feat
        for c in conv_layers:
            layers += [nn.Conv1d(ch, c["filters"], c["kernel_size"], padding="same"),
                       nn.BatchNorm1d(c["filters"]), nn.ReLU()]
            ch = c["filters"]
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(ch, lstm_units, batch_first=True)
        self.bn = nn.BatchNorm1d(lstm_units)
        self.head = _mlp(lstm_units, dense_layers, dropout_rate)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))      # (batch, ch, W)
        x, _ = self.lstm(x.permute(0, 2, 1))   # recurrence over TIME
        return self.head(self.bn(x[:, -1, :])).squeeze(-1)


class _TemporalBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, dilation, dropout_rate):
        super().__init__()
        pad = (kernel - 1) * dilation          # causal padding, trimmed below
        self.pad = pad
        self.c1 = nn.Conv1d(ch_in, ch_out, kernel, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(ch_out, ch_out, kernel, padding=pad, dilation=dilation)
        self.bn1, self.bn2 = nn.BatchNorm1d(ch_out), nn.BatchNorm1d(ch_out)
        self.drop = nn.Dropout(dropout_rate)
        self.res = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else None

    def forward(self, x):
        r = x if self.res is None else self.res(x)
        y = self.drop(torch.relu(self.bn1(self.c1(x)[:, :, :-self.pad or None])))
        y = self.drop(torch.relu(self.bn2(self.c2(y)[:, :, :-self.pad or None])))
        return torch.relu(y + r)


class WindowedTCN(nn.Module):
    def __init__(self, n_feat, num_filters, kernel_size, num_blocks,
                 dense_layers, dropout_rate):
        super().__init__()
        blocks, ch = [], n_feat
        for b in range(num_blocks):
            blocks.append(_TemporalBlock(ch, num_filters, kernel_size,
                                         2 ** b, dropout_rate))
            ch = num_filters
        self.tcn = nn.Sequential(*blocks)
        self.head = _mlp(ch, dense_layers, dropout_rate)

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))       # dilations reach back over TIME
        return self.head(x[:, :, -1]).squeeze(-1)


class WindowedTransformer(nn.Module):
    def __init__(self, n_feat, num_heads, ff_dim, num_blocks, mlp_units,
                 dropout_rate, window, d_model=128):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)         # one token per timestep
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        enc = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim,
                                         dropout_rate, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_blocks)
        self.head = _mlp(d_model, mlp_units, dropout_rate)

    def forward(self, x):
        x = self.enc(self.proj(x) + self.pos)  # attention across TIME steps
        return self.head(x.mean(dim=1)).squeeze(-1)


def _mlp(in_dim, dense_layers, dropout_rate):
    blocks, prev = [], in_dim
    for u in dense_layers:
        blocks += [nn.Linear(prev, u), nn.ReLU(), nn.Dropout(dropout_rate)]
        prev = u
    blocks.append(nn.Linear(prev, 1))
    return nn.Sequential(*blocks)


def build(name, n_feat, window, cfg):
    """Construct a windowed model from the same config the epoch track uses."""
    if name == "lstm":
        return WindowedLSTM(n_feat, cfg["lstm_layers"], cfg["dense_layers"],
                            cfg["dropout_rate"], bidirectional=False)
    if name == "bilstm":
        return WindowedLSTM(n_feat, cfg["bilstm_layers"], cfg["dense_layers"],
                            cfg["dropout_rate"], bidirectional=True)
    if name == "cnn_1d":
        return WindowedCNN(n_feat, cfg["conv_layers"], cfg["dense_layers"],
                           cfg["dropout_rate"])
    if name == "cnn_lstm":
        return WindowedCNNLSTM(n_feat, cfg["conv_layers"], cfg["lstm_units"],
                               cfg["dense_layers"], cfg["dropout_rate"])
    if name == "tcn":
        return WindowedTCN(n_feat, cfg["num_filters"], cfg["kernel_size"],
                           cfg["num_blocks"], cfg["dense_layers"],
                           cfg["dropout_rate"])
    if name == "transformer":
        return WindowedTransformer(n_feat, cfg["num_heads"], cfg["ff_dim"],
                                   cfg["num_transformer_blocks"],
                                   cfg["mlp_units"], cfg["dropout_rate"], window)
    raise ValueError("unknown model: " + name)

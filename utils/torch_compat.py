"""
torch_compat.py — cuDNN/GPU compatibility helpers.

The problem
-----------
cuDNN's fused LSTM/GRU kernel requires the module to be in **train() mode
during the forward pass** so it saves the workspace needed for backward.
If the module is in eval() mode during forward, cuDNN uses a faster
inference-only kernel (no workspace), and calling backward() later raises:

    RuntimeError: cudnn RNN backward can only be called in training mode

This fires on GPU for every gradient-based attack (FGSM / PGD) and for
adversarial training's inner PGD when applied to any RNN backbone
(lstm / bilstm / cnn_lstm).  It is invisible on CPU because CPU never
uses the cuDNN kernel.

Why previous attempts failed
-----------------------------
* ``torch.backends.cudnn.flags(enabled=False)`` — ignored on Kaggle's
  torch 2.x build; the cuDNN kernel is still selected.
* ``rnn_backward_enabled`` (sets only nn.RNNBase to train()) — fails
  because on some torch builds cuDNN's kernel selection depends on more
  internal state than just ``module.training``; the full ``model.train()``
  call is required to initialise this state correctly.

The fix: grad_compat_mode
-------------------------
1. Call ``module.train()`` on the ENTIRE model first (initialises cuDNN).
2. Then explicitly set BatchNorm and Dropout submodules back to eval()
   so gradients are deterministic (BN uses frozen running stats, no
   dropout noise).
Result: LSTM in train() (cuDNN happy), BN/Dropout in eval() (deterministic).

This is a no-op for non-RNN backbones (Transformer / TCN / CNN-1D) since
their forward+backward works in eval() mode.
"""

from __future__ import annotations
import contextlib
import torch.nn as nn


@contextlib.contextmanager
def grad_compat_mode(module: nn.Module):
    """Set ``module`` to a cuDNN-compatible state for gradient computation.

    Enters with the module in any mode; exits with the original mode restored.
    Inside the block:
      - entire module is in train() (cuDNN RNN backward works)
      - BatchNorm and Dropout submodules are individually set back to eval()
        (deterministic gradients: frozen BN stats, no dropout noise)
    """
    was_training = module.training
    module.train()                          # full train() — initialises cuDNN
    frozen = []
    for m in module.modules():
        if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
            if m.training:                  # only touch what just changed
                m.eval()
                frozen.append(m)
    try:
        yield
    finally:
        module.train(was_training)          # restores eval() if it was eval
        if was_training:                    # if was train, re-enable BN/Dropout
            for m in frozen:
                m.train()

"""
FGSM Attack — PyTorch Implementation
======================================
Fast Gradient Sign Method for adversarial example generation.

Scope
-----
DL models only. Tree-based classifiers have piecewise-constant
non-differentiable boundaries; gradient attacks are meaningless on them.
Classical model robustness is assessed via transfer attacks and
feature-space attacks (DLSA, SNA, TPA).

Reference
---------
Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).
Explaining and harnessing adversarial examples. ICLR. arXiv:1412.6572.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from utils.gnss_constraints import GNSSConstraintEnforcer


class FGSMAttack:
    """
    FGSM for PyTorch deep learning models.

    Parameters
    ----------
    model_wrapper : fitted DL model wrapper (has .model nn.Module attribute)
    epsilon       : L-inf perturbation budget in normalised feature space.
                    epsilon=0.10 ≈ ±196 Hz Doppler or ±0.52 dB CN0
                    (Kaplan & Hegarty, 2017).
    gnss_enforcer : pre-fitted GNSSConstraintEnforcer from main().
                    If None, no physical bound clipping is applied.
    feature_names : kept for API compatibility; not used when enforcer
                    is supplied externally.
    """

    def __init__(self, model_wrapper,
                 epsilon: float = 0.1,
                 gnss_enforcer: Optional[GNSSConstraintEnforcer] = None,
                 feature_names: Optional[list] = None):
        self.model_wrapper = model_wrapper
        self.nn_module = model_wrapper.model
        self.epsilon = epsilon
        self.device = next(self.nn_module.parameters()).device
        self.criterion = nn.BCEWithLogitsLoss()
        self.gnss_enforcer = gnss_enforcer   # pre-fitted; may be None

    @torch.enable_grad()
    def _compute_signed_gradient(self, X_t: torch.Tensor,
                                 y_t: torch.Tensor) -> np.ndarray:
        """Return sign(∂L/∂x) via PyTorch autograd."""
        self.nn_module.eval()
        X_var = X_t.clone().detach().requires_grad_(True)
        logits = self.nn_module(X_var)
        loss = self.criterion(logits, y_t)
        loss.backward()
        return X_var.grad.sign().detach().cpu().numpy()

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate FGSM adversarial examples.

        Parameters
        ----------
        X : (n, d) normalised feature matrix
        y : (n,)   binary labels {0, 1}

        Returns
        -------
        X_adv : (n, d) float32
        """
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        signed_grad = self._compute_signed_gradient(X_t, y_t)
        X_adv = X + self.epsilon * signed_grad

        # Projection onto L-inf ball around original X
        X_adv = np.clip(X_adv, X - self.epsilon, X + self.epsilon)

        # Physical constraint clipping — applied first so GNSS bounds are respected.
        if self.gnss_enforcer is not None:
            X_adv = self.gnss_enforcer.clip_to_gnss_bounds(X_adv)

        # Re-enforce epsilon-ball after physical constraint clipping.
        # The enforcer can only restrict the perturbation budget, never expand it.
        X_adv = np.clip(X_adv, X - self.epsilon, X + self.epsilon)

        return X_adv.astype(np.float32)

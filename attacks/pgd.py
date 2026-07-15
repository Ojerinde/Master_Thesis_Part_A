"""
PGD Attack — PyTorch Implementation
=====================================
Projected Gradient Descent (Madry et al., 2018).

Gold standard for adversarial robustness evaluation.
Iterative FGSM with L-inf projection at every step.

Scope: DL models only (same rationale as fgsm.py).

Bug fix vs previous version
-----------------------------
Previously constructed GNSSConstraintEnforcer internally without fit(),
so normalised_bounds_ was always empty. Now accepts a pre-fitted enforcer
from main() where it is fitted on training data.

Reference
---------
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
Towards deep learning models resistant to adversarial attacks.
ICLR 2018. arXiv:1706.06083.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from utils.gnss_constraints import GNSSConstraintEnforcer
from utils.torch_compat import grad_compat_mode


class PGDAttack:
    """
    PGD-Linf for PyTorch deep learning models.

    Parameters
    ----------
    model_wrapper : fitted DL model wrapper
    epsilon       : total L-inf perturbation budget (normalised space)
    alpha         : step size per iteration (default: epsilon / 4)
    num_iter      : PGD steps. 40 is the community standard for thorough
                    evaluation (Madry et al., 2018).
    random_start  : random initialisation within epsilon ball.
                    Recommended True — avoids saddle-point sensitivity.
    gnss_enforcer : pre-fitted GNSSConstraintEnforcer from main().
    feature_names : kept for API compatibility.
    """

    def __init__(self, model_wrapper,
                 epsilon: float = 0.1,
                 alpha: Optional[float] = None,
                 num_iter: int = 40,
                 random_start: bool = True,
                 gnss_enforcer: Optional[GNSSConstraintEnforcer] = None,
                 feature_names: Optional[list] = None):
        self.model_wrapper = model_wrapper
        self.nn_module = model_wrapper.model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / 4.0
        self.num_iter = num_iter
        self.random_start = random_start
        self.device = next(self.nn_module.parameters()).device
        self.criterion = nn.BCEWithLogitsLoss()
        self.gnss_enforcer = gnss_enforcer   # pre-fitted; may be None

    @torch.enable_grad()
    def _gradient_step(self, X_adv_t: torch.Tensor,
                       y_t: torch.Tensor) -> torch.Tensor:
        """Single signed gradient step."""
        X_adv_t = X_adv_t.clone().detach().requires_grad_(True)
        # grad_compat_mode: see utils/torch_compat for explanation.
        with grad_compat_mode(self.nn_module):
            logits = self.nn_module(X_adv_t)
            loss = self.criterion(logits, y_t)
            loss.backward()
        return X_adv_t.grad.sign().detach()

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate PGD adversarial examples.

        Algorithm (Madry et al., 2018)
        --------------------------------
        1. δ ← Uniform(-ε, ε)  [random start]
        2. For t = 1…T:
             a. g ← sign(∇_x L(x+δ, y))
             b. δ ← δ + α·g
             c. δ ← clip(δ, -ε, ε)
        3. Return x + δ

        Parameters
        ----------
        X : (n, d) normalised feature matrix
        y : (n,)   binary labels {0, 1}
        """
        X_orig = X.astype(np.float32)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        if self.random_start:
            rng = np.random.default_rng(42)
            delta = rng.uniform(-self.epsilon, self.epsilon,
                                X_orig.shape).astype(np.float32)
        else:
            delta = np.zeros_like(X_orig)

        for _ in range(self.num_iter):
            X_cur = torch.tensor(X_orig + delta,
                                 dtype=torch.float32).to(self.device)
            sign_g = self._gradient_step(X_cur, y_t).cpu().numpy()
            delta = delta + self.alpha * sign_g
            delta = np.clip(delta, -self.epsilon, self.epsilon)

        X_adv = (X_orig + delta).astype(np.float32)

        # Apply physical constraint clipping first.
        if self.gnss_enforcer is not None:
            X_adv = self.gnss_enforcer.clip_to_gnss_bounds(X_adv)

        # Re-enforce epsilon ball AFTER the enforcer.
        # Same reason as FGSMAttack: the enforcer must only restrict the
        # perturbation budget, never expand it. Without this second clip,
        # outlier test samples outside the training data range get "corrected"
        # by the enforcer, producing spurious L-inf >> epsilon (observed:
        # L-inf=11.2037 for all epsilon in {0.05, 0.10, 0.20}).
        X_adv = np.clip(X_adv, X_orig - self.epsilon, X_orig + self.epsilon)

        return X_adv

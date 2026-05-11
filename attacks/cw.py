"""
Carlini & Wagner L2 Attack -- PyTorch Implementation.

Optimization-based attack that finds the minimal L2 perturbation to cause
misclassification. Uses tanh-space reparametrisation to enforce box
constraints and binary search over the Lagrangian constant c.

Reference:
    Carlini, N. & Wagner, D. (2017). Towards Evaluating the Robustness
    of Neural Networks. IEEE S&P. arXiv:1608.04644.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from utils.gnss_constraints import GNSSConstraintEnforcer


class CWAttack:
    """
    C&W L2 attack for PyTorch deep-learning GNSS spoofing detectors.

    Parameters
    ----------
    model_wrapper   : fitted DL model wrapper (has .model nn.Module attribute)
    confidence      : margin kappa -- higher = more confident misclassification
    learning_rate   : Adam learning rate for the inner optimisation loop
    max_iterations  : optimisation steps per binary-search iteration
    binary_search_steps : binary search iterations for constant c
    initial_const   : starting value for Lagrangian multiplier c
    gnss_enforcer   : pre-fitted GNSSConstraintEnforcer; applied after attack
    """

    def __init__(
        self,
        model_wrapper,
        confidence: float = 0.0,
        learning_rate: float = 5e-3,
        max_iterations: int = 200,
        binary_search_steps: int = 5,
        initial_const: float = 1e-2,
        gnss_enforcer: Optional[GNSSConstraintEnforcer] = None,
        batch_size: int = 128,
    ):
        self.model_wrapper = model_wrapper
        self.nn_module = model_wrapper.model
        self.device = next(self.nn_module.parameters()).device
        self.confidence = confidence
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.gnss_enforcer = gnss_enforcer
        self.batch_size = batch_size

    # --- tanh-space helpers ---------------------------------------------------

    @staticmethod
    def _to_tanh(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Map [lo, hi] -> (-inf, inf) via arctanh."""
        x_norm = (x - lo) / (hi - lo)                              # [0, 1]
        x_norm = x_norm.clamp(1e-6, 1.0 - 1e-6)                   # avoid ±inf
        return torch.atanh(2.0 * x_norm - 1.0)

    @staticmethod
    def _from_tanh(w: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Map (-inf, inf) -> [lo, hi] via tanh."""
        return (torch.tanh(w) + 1.0) / 2.0 * (hi - lo) + lo

    # --- loss function --------------------------------------------------------

    def _cw_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                 const: torch.Tensor) -> torch.Tensor:
        """
        f(x') = max(Z(x')_t - max_{j!=t} Z(x')_j, -kappa)

        For binary classification the logit is scalar; positive = class 1.
        Untargeted attack: we want to push away from the true class.
        """
        logits = logits.squeeze(-1)
        # For binary: logit > 0 => class 1.  We flip the sign for class-1 targets
        # so that positive f means "still correctly classified".
        sign = 2.0 * targets - 1.0   # +1 for class 1, -1 for class 0
        f = sign * logits + self.confidence
        return const * torch.clamp(f, min=0.0)

    # --- main attack loop -----------------------------------------------------

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate C&W-L2 adversarial examples.

        Parameters
        ----------
        X : (n, d) normalised feature matrix (float32)
        y : (n,)   true binary labels {0, 1}

        Returns
        -------
        X_adv : (n, d) adversarial examples (float32)
        """
        self.nn_module.eval()
        n = len(X)
        X_adv = X.copy().astype(np.float32)

        # Compute per-feature bounds from the data (works in normalised space)
        lo = torch.tensor(X.min(axis=0), dtype=torch.float32,
                          device=self.device)
        hi = torch.tensor(X.max(axis=0), dtype=torch.float32,
                          device=self.device)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            x_batch = torch.tensor(
                X[start:end], dtype=torch.float32, device=self.device)
            y_batch = torch.tensor(
                y[start:end], dtype=torch.float32, device=self.device)
            batch_len = end - start

            best_adv = x_batch.clone()
            best_l2 = torch.full(
                (batch_len,), float('inf'), device=self.device)

            c_lo = torch.zeros(batch_len, device=self.device)
            c_hi = torch.full((batch_len,), 1e4, device=self.device)
            const = torch.full(
                (batch_len,), self.initial_const, device=self.device)

            for _ in range(self.binary_search_steps):
                # Initialise w in tanh space from original sample
                w = self._to_tanh(x_batch, lo, hi).clone(
                ).detach().requires_grad_(True)
                optimiser = torch.optim.Adam([w], lr=self.lr)

                for _ in range(self.max_iter):
                    x_new = self._from_tanh(w, lo, hi)
                    delta = x_new - x_batch
                    l2_dist = delta.pow(2).sum(dim=1)

                    logits = self.nn_module(x_new)
                    adv_loss = self._cw_loss(logits, y_batch, const)

                    loss = l2_dist.sum() + adv_loss.sum()

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                # Check which samples actually fooled the model
                with torch.no_grad():
                    x_final = self._from_tanh(w, lo, hi)
                    preds = (torch.sigmoid(self.nn_module(
                        x_final).squeeze(-1)) >= 0.5).long()
                    success = preds != y_batch.long()
                    l2_final = (x_final - x_batch).pow(2).sum(dim=1)

                    # Update best adversarial if successful AND lower L2
                    improved = success & (l2_final < best_l2)
                    best_adv[improved] = x_final[improved]
                    best_l2[improved] = l2_final[improved]

                    # Binary search update
                    c_hi[success] = torch.min(c_hi[success], const[success])
                    c_lo[~success] = torch.max(c_lo[~success], const[~success])
                    const = (c_lo + c_hi) / 2.0

            X_adv[start:end] = best_adv.detach().cpu().numpy()

        # GNSS physical constraint clipping
        if self.gnss_enforcer is not None:
            X_adv = self.gnss_enforcer.clip_to_gnss_bounds(X_adv)

        return X_adv.astype(np.float32)

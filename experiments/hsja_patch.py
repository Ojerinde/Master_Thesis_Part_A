"""
Torch>=2 compatibility patch for cleverhans 4.0.0 HopSkipJump.
=============================================================
cleverhans' `binary_search_batch` calls `torch.min(tensor, python_float)`, which
newer torch (2.x) rejects (it expects a tensor or a dim, not a scalar). This
module re-defines that single function -- identical to upstream except
`torch.min(t, theta)` -> `torch.clamp(t, max=theta)` (mathematically the same
element-wise minimum against a scalar) -- and monkeypatches it onto the
cleverhans module. `hop_skip_jump_attack` resolves `binary_search_batch` from the
module globals at call time, so importing this module before calling the attack
is sufficient. Re-exports `hop_skip_jump_attack` for convenience.
"""
import numpy as np
import torch
import cleverhans.torch.attacks.hop_skip_jump_attack as _m
from cleverhans.torch.attacks.hop_skip_jump_attack import (  # noqa: F401
    hop_skip_jump_attack, compute_distance, project,
)


def _binary_search_batch(original_image, perturbed_images, decision_function,
                         shape, constraint, theta):
    """Upstream binary_search_batch with the torch>=2 scalar-min fix."""
    dists_post_update = torch.stack([
        compute_distance(original_image, pi, constraint)
        for pi in perturbed_images
    ])
    if constraint == np.inf:
        highs = dists_post_update
        thresholds = torch.clamp(dists_post_update * theta, max=float(theta))
    else:
        highs = torch.ones(len(perturbed_images)).to(original_image.device)
        thresholds = theta
    lows = torch.zeros(len(perturbed_images)).to(original_image.device)

    while torch.max((highs - lows) / thresholds) > 1:
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids,
                             shape, constraint)
        decisions = decision_function(mid_images)
        lows = torch.where(decisions == 0, mids, lows)
        highs = torch.where(decisions == 1, mids, highs)

    out_images = project(original_image, perturbed_images, highs,
                         shape, constraint)
    dists = torch.stack([
        compute_distance(original_image, out_image, constraint)
        for out_image in out_images
    ])
    _, idx = torch.min(dists, 0)
    dist = dists_post_update[idx]
    out_image = out_images[idx].unsqueeze(0)
    return out_image, dist


_m.binary_search_batch = _binary_search_batch

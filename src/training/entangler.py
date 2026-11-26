"""SpectralEntanglerLoss: couples utility and safety subspaces.

Penalizes orthogonality between the top-k principal singular vectors of
utility activations/gradients and safety (refusal) activations/gradients so
that removing the refusal subspace via Abliteration also destroys utility.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _flatten_representations(x: Tensor) -> Tensor:
    """Flatten (..., d) into (N, d) and center along the sample dimension."""
    if x.ndim < 2:
        raise ValueError(f"Expected at least 2D tensor, got shape {tuple(x.shape)}")
    flat = x.reshape(-1, x.shape[-1])
    return flat - flat.mean(dim=0, keepdim=True)


def _top_k_singular_vectors(
    centered: Tensor,
    k: int,
    eps: float = 1e-6,
    power_iters: int = 8,
) -> Tensor:
    """Compute top-k feature-space singular vectors U in R^{d x k}.

    Uses differentiable subspace power iteration with modified Gram-Schmidt
    orthogonalization on the covariance matrix C = (1/N) X^T X, avoiding
    SVD backward singularities (1 / (sigma_i^2 - sigma_j^2)).
    """
    n, d = centered.shape
    k_eff = min(k, d, n)
    if k_eff <= 0:
        raise ValueError(f"Invalid effective rank k={k_eff} for shape {tuple(centered.shape)}")

    cov = (centered.T @ centered) / max(1, n)
    if eps > 0.0:
        cov = cov + eps * torch.eye(d, device=cov.device, dtype=cov.dtype)

    # Deterministic initial orthonormal basis (first k_eff canonical axes)
    q_cols = [
        F.one_hot(torch.tensor(i, device=cov.device), num_classes=d).to(cov.dtype)
        for i in range(k_eff)
    ]
    q = torch.stack(q_cols, dim=1)  # (d, k_eff)

    for _ in range(power_iters):
        z = cov @ q
        next_cols: list[Tensor] = []
        for i in range(k_eff):
            v = z[:, i]
            for prev in next_cols:
                v = v - torch.dot(v, prev) * prev
            v = v / (v.norm() + 1e-8)
            next_cols.append(v)
        q = torch.stack(next_cols, dim=1)

    return q


class SpectralEntanglerLoss(nn.Module):
    """Subspace-entanglement regularizer for utility and safety representations.

    Minimizes:
        L_entangle = lambda * (1 - (1 / k) * sum_{i=1}^k |<u_i^util, u_i^safe>|)
    """

    def __init__(
        self,
        lambda_entangle: float = 1.0,
        top_k: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if lambda_entangle < 0:
            raise ValueError(f"lambda_entangle must be non-negative, got {lambda_entangle}")
        self.lambda_entangle = lambda_entangle
        self.top_k = top_k
        self.eps = eps

    def forward(
        self,
        util_repr: Tensor,
        safe_repr: Tensor,
        adapter_deltas: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        """Compute spectral entanglement loss between utility and safety subspaces.

        Args:
            util_repr: Utility activations or gradients of shape (..., d).
            safe_repr: Safety/refusal activations or gradients of shape (..., d).
            adapter_deltas: Optional sequence of adapter weight deltas (e.g. A @ B)
                whose spectral energy is coupled into the entanglement objective.
        """
        util_flat = _flatten_representations(util_repr)
        safe_flat = _flatten_representations(safe_repr)

        k_eff = min(
            self.top_k,
            util_flat.shape[0],
            safe_flat.shape[0],
            util_flat.shape[1],
            safe_flat.shape[1],
        )

        u_util = _top_k_singular_vectors(util_flat, k_eff, eps=self.eps)
        u_safe = _top_k_singular_vectors(safe_flat, k_eff, eps=self.eps)

        # Pairwise cosine alignment of the top-k singular directions:
        # |<u_i^util, u_i^safe>| averaged over i = 1..k
        aligned_dots = (u_util * u_safe).sum(dim=0).abs()
        loss = self.lambda_entangle * (1.0 - aligned_dots.mean())

        if adapter_deltas:
            # Encourage adapter deltas to share energy with the entangled subspace
            u_joint = 0.5 * (u_util + u_safe)
            u_joint = u_joint / (u_joint.norm(dim=0, keepdim=True) + 1e-8)
            proj_penalties = []
            for delta in adapter_deltas:
                if delta.shape[0] == u_joint.shape[0]:
                    proj = u_joint.T @ delta
                    energy_ratio = (proj.pow(2).sum()) / (delta.pow(2).sum() + 1e-8)
                    proj_penalties.append(1.0 - energy_ratio)
                elif delta.shape[1] == u_joint.shape[0]:
                    proj = delta @ u_joint
                    energy_ratio = (proj.pow(2).sum()) / (delta.pow(2).sum() + 1e-8)
                    proj_penalties.append(1.0 - energy_ratio)
            if proj_penalties:
                loss = loss + 0.1 * self.lambda_entangle * torch.stack(proj_penalties).mean()

        return loss

"""Differentiable stereo wind solver for training.

Ports the column-scaled WLS solve from solver._solve_wls_gpu to a
torch.nn.Module that keeps tensors live for backpropagation.

Gradient path:
    loss(h) -> x_sol[...,0] -> solve(A, b) -> b -> y_w -> y -> RAFT flows

Only the observation vector ``y`` carries gradients.  The design matrix ``H``
and weight matrix ``W`` are geometry (no grad).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DifferentiableStereoSolver(nn.Module):
    """Stateless WLS solver — H and W are passed per forward call.

    Replicates the linear algebra from ``solver._solve_wls_gpu`` exactly:
    column-scaled normal equations + Tikhonov regularization +
    ``torch.linalg.solve``.

    Parameters
    ----------
    regularization : Tikhonov regularization added to the diagonal of the
        normal matrix.  Must match the production solver for parity.
    """

    def __init__(self, regularization: float = 1e-10):
        super().__init__()
        self.regularization = regularization

    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        W: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the 5-state WLS system per pixel.

        Parameters
        ----------
        y : (..., 8)  observation vector (stacked RAFT flows).  **Carries grad.**
        H : (..., 8, 5)  design matrix.  No grad.
        W : (..., 8)  per-observation weights, or None for equal weights.

        Returns
        -------
        x_sol : (..., 5)  state vector [h, p_u, p_v, V_u, V_v]
        chi2  : (...)  weighted residual sum of squares
        """
        if W is None:
            W = torch.ones_like(y)

        # Weighted system
        sqrt_W = torch.sqrt(W)                                 # (..., 8)
        H_w = H * sqrt_W.unsqueeze(-1)                        # (..., 8, 5)
        y_w = y * sqrt_W                                       # (..., 8)

        # Column scaling for numerical stability
        col_norms = torch.sqrt(
            torch.einsum("...ki,...ki->...i", H_w, H_w)       # (..., 5)
        )
        col_norms = torch.where(
            col_norms > 0, col_norms, torch.ones_like(col_norms)
        )
        H_scaled = H_w / col_norms.unsqueeze(-2)              # (..., 8, 5)

        # Normal equations: A x_scaled = b
        A = torch.einsum("...ki,...kj->...ij", H_scaled, H_scaled)  # (..., 5, 5)
        b = torch.einsum("...ki,...k->...i", H_scaled, y_w)         # (..., 5)

        # Tikhonov regularization
        eye = torch.eye(5, device=y.device, dtype=y.dtype)
        A = A + eye * self.regularization

        # Solve — differentiable w.r.t. b (and A, but A doesn't depend on y)
        x_scaled = torch.linalg.solve(A, b)                   # (..., 5)
        x_sol = x_scaled / col_norms                           # (..., 5)

        # Weighted residual sum of squares.
        # Detach x_sol here: the WLS residual is orthogonal to the column
        # space of H (normal equations), so d(chi2)/dy is the same whether
        # or not we differentiate through solve.  Detaching avoids the
        # numerically unstable backward of torch.linalg.solve on
        # ill-conditioned normal matrices.
        y_pred = torch.einsum("...ij,...j->...i", H, x_sol.detach())  # (..., 8)
        residual = y - y_pred
        chi2 = (residual ** 2 * W).sum(dim=-1)                       # (...)

        return x_sol, chi2


def build_design_matrix_torch(
    w_u: torch.Tensor,
    w_v: torch.Tensor,
    dt_a_minus: float,
    dt_a_plus: float,
    dt_b_minus: float,
    dt_b_plus: float,
) -> torch.Tensor:
    """Build the (*, 8, 5) design matrix from parallax vectors and time offsets.

    Torch equivalent of ``solver.build_design_matrix``.  No gradients needed —
    this produces geometry that is constant for a given satellite pair and scene.

    Parameters
    ----------
    w_u, w_v : (*, ) parallax sensitivity in pixels per meter
    dt_a_minus, dt_a_plus, dt_b_minus, dt_b_plus : time offsets in seconds

    Returns
    -------
    H : (*, 8, 5)
    """
    shape = w_u.shape
    device = w_u.device
    dtype = w_u.dtype

    H = torch.zeros(*shape, 8, 5, device=device, dtype=dtype)

    # Row 0: A_minus u → [0, 1, 0, dt_am, 0]
    H[..., 0, 1] = 1.0
    H[..., 0, 3] = dt_a_minus
    # Row 1: A_minus v → [0, 0, 1, 0, dt_am]
    H[..., 1, 2] = 1.0
    H[..., 1, 4] = dt_a_minus
    # Row 2: A_plus u → [0, 1, 0, dt_ap, 0]
    H[..., 2, 1] = 1.0
    H[..., 2, 3] = dt_a_plus
    # Row 3: A_plus v → [0, 0, 1, 0, dt_ap]
    H[..., 3, 2] = 1.0
    H[..., 3, 4] = dt_a_plus
    # Row 4: B_minus u → [w_u, 1, 0, dt_bm, 0]
    H[..., 4, 0] = w_u
    H[..., 4, 1] = 1.0
    H[..., 4, 3] = dt_b_minus
    # Row 5: B_minus v → [w_v, 0, 1, 0, dt_bm]
    H[..., 5, 0] = w_v
    H[..., 5, 2] = 1.0
    H[..., 5, 4] = dt_b_minus
    # Row 6: B_plus u → [w_u, 1, 0, dt_bp, 0]
    H[..., 6, 0] = w_u
    H[..., 6, 1] = 1.0
    H[..., 6, 3] = dt_b_plus
    # Row 7: B_plus v → [w_v, 0, 1, 0, dt_bp]
    H[..., 7, 0] = w_v
    H[..., 7, 2] = 1.0
    H[..., 7, 4] = dt_b_plus

    return H

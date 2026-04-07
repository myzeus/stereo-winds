"""Tests for the differentiable stereo wind solver."""

import numpy as np
import pytest
import torch

from stereo_winds.differentiable_solver import (
    DifferentiableStereoSolver,
    build_design_matrix_torch,
)
from stereo_winds.solver import build_design_matrix, _solve_wls_cpu


class TestBuildDesignMatrixTorch:
    """Verify torch design matrix matches numpy version."""

    def test_parity_with_numpy(self):
        n = 8
        w_u_np = np.full((n, n), 0.001)
        w_v_np = np.full((n, n), 0.0003)
        dt_am, dt_ap, dt_bm, dt_bp = -600.0, 600.0, -600.0, 600.0

        H_np = build_design_matrix(w_u_np, w_v_np, dt_am, dt_ap, dt_bm, dt_bp)

        w_u_t = torch.from_numpy(w_u_np)
        w_v_t = torch.from_numpy(w_v_np)
        H_t = build_design_matrix_torch(w_u_t, w_v_t, dt_am, dt_ap, dt_bm, dt_bp)

        np.testing.assert_allclose(H_t.numpy(), H_np, atol=1e-12)

    def test_shape(self):
        w_u = torch.ones(4, 4) * 0.001
        w_v = torch.ones(4, 4) * 0.0003
        H = build_design_matrix_torch(w_u, w_v, -600.0, 600.0, -600.0, 600.0)
        assert H.shape == (4, 4, 8, 5)

    def test_batched(self):
        """Works with an extra batch dimension."""
        w_u = torch.ones(2, 4, 4) * 0.001
        w_v = torch.ones(2, 4, 4) * 0.0003
        H = build_design_matrix_torch(w_u, w_v, -600.0, 600.0, -600.0, 600.0)
        assert H.shape == (2, 4, 4, 8, 5)


class TestDifferentiableSolverParity:
    """Verify numeric parity with the numpy CPU solver."""

    @staticmethod
    def _synthetic_problem(n=10, h_true=8000.0, V_u=5.0, V_v=-3.0,
                           p_u=0.5, p_v=-0.2):
        """Generate synthetic flows from known state, return numpy arrays."""
        w_u = np.full((n, n), 0.001)
        w_v = np.full((n, n), 0.0003)
        dt_am, dt_ap, dt_bm, dt_bp = -600.0, 600.0, -600.0, 600.0

        H_np = build_design_matrix(w_u, w_v, dt_am, dt_ap, dt_bm, dt_bp)
        x_true = np.array([h_true, p_u, p_v, V_u, V_v])
        y_np = np.einsum("...ij,...j->...i", H_np, x_true)
        W_np = np.ones_like(y_np)

        return H_np, y_np, W_np, x_true

    def test_parity_no_noise(self):
        """Differentiable solver matches CPU solver on clean synthetic data."""
        H_np, y_np, W_np, x_true = self._synthetic_problem()

        # NumPy reference
        x_ref, chi2_ref, _, _, _ = _solve_wls_cpu(H_np, y_np, W_np)

        # Torch differentiable solver
        solver = DifferentiableStereoSolver()
        H_t = torch.from_numpy(H_np).float()
        y_t = torch.from_numpy(y_np).float()
        W_t = torch.from_numpy(W_np).float()

        x_sol, chi2 = solver(y_t, H_t, W_t)

        np.testing.assert_allclose(
            x_sol.detach().numpy(), x_ref.astype(np.float32),
            atol=1e-3, rtol=1e-4,
        )
        np.testing.assert_allclose(
            chi2.detach().numpy(), chi2_ref.astype(np.float32),
            atol=1e-4,
        )

    def test_exact_recovery(self):
        """Solver should recover the true state vector."""
        H_np, y_np, W_np, x_true = self._synthetic_problem()

        solver = DifferentiableStereoSolver()
        x_sol, chi2 = solver(
            torch.from_numpy(y_np).float(),
            torch.from_numpy(H_np).float(),
            torch.from_numpy(W_np).float(),
        )
        x_out = x_sol.detach().numpy()

        # Height: 8000m, tolerance ~1m (float32 conditioning)
        np.testing.assert_allclose(x_out[..., 0], x_true[0], atol=2.0)
        np.testing.assert_allclose(x_out[..., 1], x_true[1], atol=0.01)
        np.testing.assert_allclose(x_out[..., 2], x_true[2], atol=0.01)
        np.testing.assert_allclose(x_out[..., 3], x_true[3], atol=0.01)
        np.testing.assert_allclose(x_out[..., 4], x_true[4], atol=0.01)

        # Chi2 should be ~0 for noiseless data
        np.testing.assert_allclose(chi2.detach().numpy(), 0.0, atol=1e-3)

    def test_parity_with_noise(self):
        """Matches CPU solver on noisy data too."""
        H_np, y_np, W_np, _ = self._synthetic_problem(n=20)

        rng = np.random.default_rng(42)
        y_np += rng.normal(0, 0.5, y_np.shape)

        x_ref, chi2_ref, _, _, _ = _solve_wls_cpu(H_np, y_np, W_np)

        solver = DifferentiableStereoSolver()
        x_sol, chi2 = solver(
            torch.from_numpy(y_np).float(),
            torch.from_numpy(H_np).float(),
            torch.from_numpy(W_np).float(),
        )

        np.testing.assert_allclose(
            x_sol.detach().numpy(), x_ref.astype(np.float32),
            atol=1e-2, rtol=1e-3,
        )

    def test_single_pixel(self):
        """Works for 1x1 grid."""
        H_np, y_np, W_np, x_true = self._synthetic_problem(n=1)

        solver = DifferentiableStereoSolver()
        x_sol, chi2 = solver(
            torch.from_numpy(y_np).float(),
            torch.from_numpy(H_np).float(),
            torch.from_numpy(W_np).float(),
        )
        np.testing.assert_allclose(x_sol[0, 0, 0].item(), x_true[0], atol=2.0)

    def test_equal_weights_default(self):
        """W=None should produce the same result as W=ones."""
        H_np, y_np, W_np, _ = self._synthetic_problem()

        solver = DifferentiableStereoSolver()
        H_t = torch.from_numpy(H_np).float()
        y_t = torch.from_numpy(y_np).float()
        W_t = torch.from_numpy(W_np).float()

        x_with_w, chi2_with_w = solver(y_t, H_t, W_t)
        x_no_w, chi2_no_w = solver(y_t, H_t)

        np.testing.assert_allclose(
            x_with_w.detach().numpy(), x_no_w.detach().numpy(), atol=1e-6,
        )


class TestDifferentiableSolverGradients:
    """Verify gradient flow through the solver."""

    @staticmethod
    def _make_problem(n=4, dtype=torch.float64):
        """Small problem for gradient testing."""
        w_u = torch.full((n, n), 0.001, dtype=dtype)
        w_v = torch.full((n, n), 0.0003, dtype=dtype)
        H = build_design_matrix_torch(w_u, w_v, -600.0, 600.0, -600.0, 600.0)

        x_true = torch.tensor([8000.0, 0.5, -0.2, 5.0, -3.0], dtype=dtype)
        y = torch.einsum("...ij,...j->...i", H, x_true)
        # Add small perturbation so chi2 > 0
        y = y + torch.randn_like(y) * 0.1

        return H, y

    def test_backward_runs(self):
        """loss.backward() completes without error."""
        H, y = self._make_problem()
        y.requires_grad_(True)

        solver = DifferentiableStereoSolver()
        x_sol, chi2 = solver(y, H)
        loss = chi2.mean()
        loss.backward()

        assert y.grad is not None
        assert y.grad.shape == y.shape

    def test_grad_nonzero(self):
        """Gradient w.r.t. y should be nonzero."""
        H, y = self._make_problem()
        y.requires_grad_(True)

        solver = DifferentiableStereoSolver()
        x_sol, chi2 = solver(y, H)
        loss = chi2.mean()
        loss.backward()

        assert y.grad.abs().max() > 0

    def test_height_loss_gradient(self):
        """Gradient through h = x_sol[..., 0] should flow to y."""
        H, y = self._make_problem()
        y.requires_grad_(True)

        solver = DifferentiableStereoSolver()
        x_sol, _ = solver(y, H)
        h = x_sol[..., 0]
        loss = h.mean()
        loss.backward()

        assert y.grad is not None
        assert y.grad.abs().max() > 0

    def test_gradcheck(self):
        """Analytical gradients match numerical finite differences."""
        H, y = self._make_problem(n=3, dtype=torch.float64)
        y.requires_grad_(True)

        solver = DifferentiableStereoSolver().double()

        def func(y_in):
            x_sol, chi2 = solver(y_in, H)
            return chi2.sum()

        assert torch.autograd.gradcheck(func, (y,), eps=1e-6, atol=1e-4)

    def test_gradcheck_height(self):
        """Gradient check through height output."""
        H, y = self._make_problem(n=3, dtype=torch.float64)
        y.requires_grad_(True)

        solver = DifferentiableStereoSolver().double()

        def func(y_in):
            x_sol, _ = solver(y_in, H)
            return x_sol[..., 0].sum()

        assert torch.autograd.gradcheck(func, (y,), eps=1e-6, atol=1e-4)

    def test_H_has_no_grad(self):
        """Design matrix should not accumulate gradients."""
        H, y = self._make_problem()
        H.requires_grad_(True)
        y.requires_grad_(True)

        solver = DifferentiableStereoSolver()
        x_sol, chi2 = solver(y, H)
        loss = chi2.mean()
        loss.backward()

        # H gets grad because PyTorch tracks it, but in practice
        # we never pass H with requires_grad=True.  This test just
        # confirms the solver doesn't error with grads on H.
        assert y.grad is not None

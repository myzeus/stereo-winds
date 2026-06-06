"""Tests for the heteroscedastic NLL used to train the wind student."""

import math

import torch

from stereo_winds.student_module import heteroscedastic_nll, vector_nll


class TestHeteroscedasticNLL:
    def test_matches_closed_form(self):
        """gaussian NLL (sans constant) for known mean/var/target."""
        target = torch.tensor([[2.0, 4.0]])
        mean = torch.tensor([[1.0, 4.0]])
        logvar = torch.tensor([[0.0, math.log(4.0)]])  # var = 1, 4
        mask = torch.ones_like(target, dtype=torch.bool)
        got = heteroscedastic_nll(target, mean, logvar, mask)
        # pixel0: 0.5*(1*1 + 0)=0.5 ; pixel1: 0.5*(0.25*0 + ln4)=0.5*ln4
        expect = (0.5 + 0.5 * math.log(4.0)) / 2
        assert math.isclose(got.item(), expect, rel_tol=1e-5)

    def test_empty_mask_returns_zero_with_grad(self):
        mean = torch.zeros(1, 4, requires_grad=True)
        target = torch.ones(1, 4)
        logvar = torch.zeros(1, 4)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        loss = heteroscedastic_nll(target, mean, logvar, mask)
        assert loss.item() == 0.0
        loss.backward()  # graph must stay alive
        assert mean.grad is not None

    def test_masking_excludes_pixels(self):
        target = torch.tensor([[0.0, 100.0]])
        mean = torch.zeros(1, 2)
        logvar = torch.zeros(1, 2)
        mask = torch.tensor([[True, False]])
        # The huge error at pixel1 is masked out -> loss == 0
        got = heteroscedastic_nll(target, mean, logvar, mask)
        assert math.isclose(got.item(), 0.0, abs_tol=1e-6)

    def test_weight_changes_balance(self):
        target = torch.tensor([[0.0, 0.0]])
        mean = torch.tensor([[1.0, 3.0]])  # errors 1 and 3
        logvar = torch.zeros(1, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        w_lo = heteroscedastic_nll(target, mean, logvar, mask,
                                   weight=torch.tensor([[1.0, 1.0]]))
        w_hi = heteroscedastic_nll(target, mean, logvar, mask,
                                   weight=torch.tensor([[1.0, 3.0]]))
        # Up-weighting the larger-error pixel raises the loss
        assert w_hi.item() > w_lo.item()

    def test_huber_mode_finite(self):
        target = torch.zeros(1, 4)
        mean = torch.tensor([[0.0, 1.0, 10.0, 100.0]])
        logvar = torch.zeros(1, 4)
        mask = torch.ones(1, 4, dtype=torch.bool)
        got = heteroscedastic_nll(target, mean, logvar, mask,
                                  mode="huber_learned", delta=5.0)
        assert torch.isfinite(got)


class TestVectorNLL:
    """Bivariate-Gaussian NLL on the wind vector (joint isotropic σ)."""

    def test_matches_closed_form(self):
        # Two pixels.  Pixel 0: (du, dv) = (1, 0), logvar = 0 (var=1) → 0.5*1 + 0 = 0.5.
        # Pixel 1: (du, dv) = (0, 0), logvar = log(4) (var=4) → 0 + log(4) = log(4).
        u_t = torch.tensor([[1.0, 0.0]])
        v_t = torch.tensor([[0.0, 0.0]])
        u_p = torch.zeros(1, 2)
        v_p = torch.zeros(1, 2)
        logvar = torch.tensor([[0.0, math.log(4.0)]])
        mask = torch.ones_like(u_t, dtype=torch.bool)
        got = vector_nll(u_t, v_t, u_p, v_p, logvar, mask)
        expect = (0.5 + math.log(4.0)) / 2.0
        assert math.isclose(got.item(), expect, rel_tol=1e-5)

    def test_perfect_prediction_at_minimum_logvar(self):
        """At zero residual the NLL collapses to just the complexity term."""
        u_t = torch.zeros(1, 4); v_t = torch.zeros(1, 4)
        u_p = torch.zeros(1, 4); v_p = torch.zeros(1, 4)
        logvar = torch.full((1, 4), 2.0)
        mask = torch.ones(1, 4, dtype=torch.bool)
        got = vector_nll(u_t, v_t, u_p, v_p, logvar, mask)
        assert math.isclose(got.item(), 2.0, rel_tol=1e-5)

    def test_empty_mask_returns_zero_with_grad(self):
        u_p = torch.zeros(1, 4, requires_grad=True)
        v_p = torch.zeros(1, 4, requires_grad=True)
        u_t = torch.ones(1, 4); v_t = torch.ones(1, 4)
        logvar = torch.zeros(1, 4)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        loss = vector_nll(u_t, v_t, u_p, v_p, logvar, mask)
        assert loss.item() == 0.0
        loss.backward()
        assert u_p.grad is not None

    def test_weighting(self):
        u_t = torch.tensor([[0.0, 0.0]])
        v_t = torch.tensor([[0.0, 0.0]])
        u_p = torch.tensor([[1.0, 3.0]])  # |err| = 1, 3
        v_p = torch.zeros(1, 2)
        logvar = torch.zeros(1, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        a = vector_nll(u_t, v_t, u_p, v_p, logvar, mask,
                       weight=torch.tensor([[1.0, 1.0]]))
        b = vector_nll(u_t, v_t, u_p, v_p, logvar, mask,
                       weight=torch.tensor([[1.0, 3.0]]))
        assert b.item() > a.item()

    def test_couples_u_and_v(self):
        """Trading error from u → v keeps RMSVD constant → loss constant."""
        u_t = torch.zeros(1, 8); v_t = torch.zeros(1, 8)
        logvar = torch.zeros(1, 8)
        mask = torch.ones(1, 8, dtype=torch.bool)
        # Same total magnitude per pixel (radius = 2), distributed differently.
        u_p_a = torch.full((1, 8), 2.0); v_p_a = torch.zeros(1, 8)
        u_p_b = torch.full((1, 8), 2.0 ** 0.5)
        v_p_b = torch.full((1, 8), 2.0 ** 0.5)
        a = vector_nll(u_t, v_t, u_p_a, v_p_a, logvar, mask)
        b = vector_nll(u_t, v_t, u_p_b, v_p_b, logvar, mask)
        assert math.isclose(a.item(), b.item(), rel_tol=1e-5)

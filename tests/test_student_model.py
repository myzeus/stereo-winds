"""Tests for the pixelwise wind student model."""

import torch

from stereo_winds.student_model import PixelwiseWindStudent, UNetWindStudent


class TestPixelwiseWindStudent:
    def test_output_shapes_and_keys(self):
        model = PixelwiseWindStudent(in_channels=28, hidden=32, n_layers=3)
        out = model(torch.randn(2, 28, 64, 64))
        assert set(out) == {"u_mean", "v_mean", "h_mean",
                            "u_logvar", "v_logvar", "h_logvar"}
        for v in out.values():
            assert v.shape == (2, 64, 64)
            assert torch.isfinite(v).all()

    def test_predictions_unbounded(self):
        """Means are predicted in standardized space (no tanh/sigmoid bounding).

        The wrapping LightningModule denormalizes for physical units.
        """
        model = PixelwiseWindStudent(in_channels=8, hidden=16, n_layers=2)
        out = model(torch.randn(1, 8, 16, 16))
        for k in ("u_mean", "v_mean", "h_mean"):
            assert torch.isfinite(out[k]).all()

    def test_logvar_clamped(self):
        model = PixelwiseWindStudent(in_channels=8, hidden=16, n_layers=2,
                                     logvar_min=-4.0, logvar_max=4.0)
        out = model(torch.randn(1, 8, 16, 16) * 100)
        assert out["u_logvar"].min() >= -4.0
        assert out["u_logvar"].max() <= 4.0

    def test_pixelwise_no_context_is_local(self):
        """Without context convs, each output pixel depends only on its input."""
        model = PixelwiseWindStudent(in_channels=4, hidden=16, n_layers=2,
                                     context=False).eval()
        x = torch.randn(1, 4, 8, 8)
        out_a = model(x)["u_mean"]
        x2 = x.clone()
        x2[0, :, 0, 0] += 5.0  # perturb a single pixel
        out_b = model(x2)["u_mean"]
        diff = (out_a - out_b).abs()
        # Only pixel (0,0) should change
        assert diff[0, 0, 0] > 0
        diff[0, 0, 0] = 0
        assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)

    def test_gradients_flow(self):
        model = PixelwiseWindStudent(in_channels=12, hidden=16, n_layers=2)
        out = model(torch.randn(1, 12, 16, 16))
        loss = sum(v.mean() for v in out.values())
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


class TestUNetWindStudent:
    """Same output contract as PixelwiseWindStudent + actually uses spatial context."""

    def test_output_shapes_and_keys_per_component(self):
        model = UNetWindStudent(in_channels=16, base_channels=16, n_levels=2)
        out = model(torch.randn(2, 16, 32, 32))
        assert set(out) == {"u_mean", "v_mean", "h_mean",
                            "u_logvar", "v_logvar", "h_logvar"}
        for v in out.values():
            assert v.shape == (2, 32, 32)
            assert torch.isfinite(v).all()

    def test_output_keys_joint(self):
        model = UNetWindStudent(in_channels=16, base_channels=16, n_levels=2,
                                wind_logvar_mode="joint")
        out = model(torch.randn(1, 16, 32, 32))
        assert set(out) == {"u_mean", "v_mean", "h_mean", "uv_logvar", "h_logvar"}
        assert out["uv_logvar"].shape == out["u_mean"].shape

    def test_uses_spatial_context(self):
        """Perturbing a single input pixel changes outputs at nearby pixels too
        — confirms the U-Net actually mixes neighbourhood info (unlike the
        pixelwise variant where only the perturbed pixel changes).
        """
        model = UNetWindStudent(in_channels=4, base_channels=16, n_levels=2).eval()
        x = torch.randn(1, 4, 32, 32)
        out_a = model(x)["u_mean"]
        x2 = x.clone()
        x2[0, :, 16, 16] += 5.0
        out_b = model(x2)["u_mean"]
        diff = (out_a - out_b).abs()
        # The perturbed pixel must change
        assert diff[0, 16, 16] > 0
        # And at least some neighbours must also change (vs pixelwise which would not)
        diff[0, 16, 16] = 0
        n_changed = (diff > 1e-6).sum().item()
        assert n_changed > 10, f"only {n_changed} neighbouring pixels changed"

    def test_logvar_clamped(self):
        model = UNetWindStudent(in_channels=8, base_channels=8, n_levels=2,
                                logvar_min=-4.0, logvar_max=4.0)
        out = model(torch.randn(1, 8, 32, 32) * 100)
        assert out["u_logvar"].min() >= -4.0
        assert out["u_logvar"].max() <= 4.0

    def test_gradients_flow(self):
        model = UNetWindStudent(in_channels=12, base_channels=16, n_levels=2)
        out = model(torch.randn(1, 12, 32, 32))
        loss = sum(v.mean() for v in out.values())
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_param_count_modest(self):
        """Sanity-check the model size is in a reasonable range for base=32, levels=3."""
        model = UNetWindStudent(in_channels=17, base_channels=32, n_levels=3)
        n_params = sum(p.numel() for p in model.parameters())
        assert 500_000 < n_params < 5_000_000, f"unet param count {n_params:,} out of expected range"

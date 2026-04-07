"""Lightning training module for RAFT fine-tuning with stereo wind supervision.

Phase 1: Self-supervised RSS minimization — the chi2 residual from the WLS
    solver measures how well the 8 RAFT flow observations are explained by
    the 5-state geometric model.  No external labels needed.

Phase 2 (future): Add EarthCARE height supervision via sparse loss on
    the solved height field.

Usage
-----
    from stereo_winds.lightning_module import StereoWindsModule

    module = StereoWindsModule(
        raft_ckpt="zeus/zeus/networks/weights/raft-128.202509.epoch1434.ckpt",
        learning_rate=1e-5,
    )
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu")
    trainer.fit(module, train_dataloader)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps as plt_cmaps
import numpy as np
import torch
import torch.nn as nn
import wandb
from pytorch_lightning import LightningModule

# Ensure zeus is importable for RAFT
_zeus_path = str(Path(__file__).resolve().parent.parent / "zeus")
if _zeus_path not in sys.path:
    sys.path.insert(0, _zeus_path)

from zeus.networks.raft.raft import RAFT

from .differentiable_solver import DifferentiableStereoSolver

logger = logging.getLogger(__name__)

# RAFT configuration matching the pretrained checkpoint
RAFT_ARGS = {
    "small": False,
    "lr": 1e-4,
    "mixed_precision": False,
    "dropout": 0.0,
    "corr_levels": 4,
    "corr_radius": 4,
}


class StereoWindsModule(LightningModule):
    """Fine-tune RAFT for geometrically self-consistent stereo flows.

    Parameters
    ----------
    raft_ckpt : path to RAFT checkpoint (PL or FlowRunner format)
    learning_rate : optimizer LR (default 1e-5 for fine-tuning)
    freeze_fnet : freeze the feature encoder (train only update block + cnet)
    bounds_weight : weight for the physical bounds penalty on h
    iters : RAFT refinement iterations (default 12)
    """

    def __init__(
        self,
        raft_ckpt: str,
        learning_rate: float = 1e-5,
        freeze_fnet: bool = False,
        bounds_weight: float = 1e-4,
        height_reg_weight: float = 0.0,
        wind_reg_weight: float = 0.0,
        ec_weight: float = 0.0,
        iters: int = 12,
        log_media_step: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # RAFT optical flow model
        self.raft = RAFT(RAFT_ARGS.copy())
        self._load_raft_weights(raft_ckpt)

        if freeze_fnet:
            for p in self.raft.fnet.parameters():
                p.requires_grad = False
            logger.info("Frozen fnet (%d params)",
                        sum(p.numel() for p in self.raft.fnet.parameters()))

        # Frozen pretrained RAFT for regularization
        self.height_reg_weight = height_reg_weight
        self.wind_reg_weight = wind_reg_weight
        self._needs_ref = height_reg_weight > 0 or wind_reg_weight > 0
        if self._needs_ref:
            self.raft_ref = RAFT(RAFT_ARGS.copy())
            self._load_raft_weights(raft_ckpt, target=self.raft_ref)
            for p in self.raft_ref.parameters():
                p.requires_grad = False
            self.raft_ref.eval()
            logger.info("Regularization: h_weight=%.4f, wind_weight=%.4f",
                        height_reg_weight, wind_reg_weight)

        # Differentiable stereo solver
        self.solver = DifferentiableStereoSolver()

        # EarthCARE height supervision
        self.ec_weight = ec_weight
        if ec_weight > 0:
            from .sparse_loss import SparseHuberLoss, SparseCloudContainmentLoss
            self.ec_loss_fn = SparseHuberLoss(delta=1.0)  # 1 km Huber threshold
            self.ec_layer_loss_fn = SparseCloudContainmentLoss(delta=1.0)
            logger.info("EarthCARE supervision enabled (weight=%.4f)", ec_weight)

        self.learning_rate = learning_rate
        self.bounds_weight = bounds_weight
        self.iters = iters
        self.log_media_step = log_media_step

    def _load_raft_weights(self, ckpt_path: str, target=None) -> None:
        """Load RAFT weights from PL or FlowRunner checkpoint."""
        model = target if target is not None else self.raft
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in data:
            sd = data["state_dict"]
        elif "model" in data:
            sd = data["model"]
        else:
            sd = data

        # Strip "module." prefix from DataParallel wrapping if present
        sd = {k.removeprefix("module."): v for k, v in sd.items()}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Missing keys in RAFT checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in RAFT checkpoint: %s", unexpected)
        logger.info("Loaded RAFT weights from %s (%d params)",
                    Path(ckpt_path).name, len(sd))

    def on_train_start(self) -> None:
        """Freeze BatchNorm running stats to prevent drift with small batches."""
        self.raft.freeze_bn()

    def on_before_optimizer_step(self, optimizer) -> None:
        """Replace NaN/Inf gradients with zeros before the optimizer step.

        RAFT's correlation volume backward (grid_sample) can produce NaN
        gradients for pixels with large flows or ill-conditioned patches.
        """
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def _flows_to_y(self, d1, d2, d3, d4):
        """Negate v-components and stack into (B, H, W, 8) observation vector."""
        d1 = torch.stack([ d1[:, 0], -d1[:, 1]], dim=1)
        d2 = torch.stack([ d2[:, 0], -d2[:, 1]], dim=1)
        d3 = torch.stack([ d3[:, 0], -d3[:, 1]], dim=1)
        d4 = torch.stack([ d4[:, 0], -d4[:, 1]], dim=1)
        return torch.stack([
            d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1],
            d3[:, 0], d3[:, 1], d4[:, 0], d4[:, 1],
        ], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run 4x RAFT + solver.  Returns dict with h, chi2, x_sol.

        When training (test_mode=False), also returns per-iteration solutions
        in ``iter_results`` for multi-iteration loss.
        """
        a0 = batch["A0"]
        a_minus = batch["A_minus"]
        a_plus = batch["A_plus"]
        b_minus = batch["B_minus"]
        b_plus = batch["B_plus"]
        H_mat = batch["H_matrix"]

        is_training = self.training

        # 4 RAFT forward passes
        d1_all = self.raft(a0, a_minus, iters=self.iters, test_mode=not is_training)
        d2_all = self.raft(a0, a_plus,  iters=self.iters, test_mode=not is_training)
        d3_all = self.raft(a0, b_minus, iters=self.iters, test_mode=not is_training)
        d4_all = self.raft(a0, b_plus,  iters=self.iters, test_mode=not is_training)

        if not is_training:
            # Inference: single final flow
            y = self._flows_to_y(d1_all[0], d2_all[0], d3_all[0], d4_all[0])
            x_sol, chi2 = self.solver(y, H_mat)
            return {"h": x_sol[..., 0], "chi2": chi2, "x_sol": x_sol, "y": y}

        # Training: solve at each RAFT iteration for multi-iteration loss
        n_iters = len(d1_all)
        iter_results = []
        for i in range(n_iters):
            y_i = self._flows_to_y(d1_all[i], d2_all[i], d3_all[i], d4_all[i])
            x_sol_i, chi2_i = self.solver(y_i, H_mat)
            iter_results.append({"h": x_sol_i[..., 0], "chi2": chi2_i, "x_sol": x_sol_i, "y": y_i})

        # Final iteration is the primary output
        final = iter_results[-1]
        final["iter_results"] = iter_results
        return final

    def _forward_ref(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run frozen pretrained RAFT + solver. Returns x_sol (no grad)."""
        a0 = batch["A0"]
        d1 = self.raft_ref(a0, batch["A_minus"], iters=self.iters, test_mode=True)[0]
        d2 = self.raft_ref(a0, batch["A_plus"],  iters=self.iters, test_mode=True)[0]
        d3 = self.raft_ref(a0, batch["B_minus"], iters=self.iters, test_mode=True)[0]
        d4 = self.raft_ref(a0, batch["B_plus"],  iters=self.iters, test_mode=True)[0]

        d1 = torch.stack([ d1[:, 0], -d1[:, 1]], dim=1)
        d2 = torch.stack([ d2[:, 0], -d2[:, 1]], dim=1)
        d3 = torch.stack([ d3[:, 0], -d3[:, 1]], dim=1)
        d4 = torch.stack([ d4[:, 0], -d4[:, 1]], dim=1)

        y_ref = torch.stack([
            d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1],
            d3[:, 0], d3[:, 1], d4[:, 0], d4[:, 1],
        ], dim=-1)
        x_sol, _ = self.solver(y_ref, batch["H_matrix"])
        return x_sol

    def _compute_iter_loss(self, h, chi2, batch):
        """Compute RSS + EC loss for a single RAFT iteration's output."""
        valid = torch.isfinite(chi2) & torch.isfinite(h)

        # RSS loss
        if valid.any():
            rss_loss = chi2[valid].mean()
        else:
            rss_loss = chi2.sum() * 0

        # EarthCARE height supervision (sparse)
        ec_loss = torch.tensor(0.0, device=h.device)
        if self.ec_weight > 0 and "ec_mask" in batch:
            ec_mask = batch["ec_mask"]
            if ec_mask.any():
                if "ec_layer_tops" in batch:
                    # Cloud containment: zero loss inside cloud, Huber outside
                    ec_loss = self.ec_layer_loss_fn(
                        h / 1000,
                        batch["ec_layer_tops"] / 1000,
                        batch["ec_layer_bases"] / 1000,
                        batch["ec_layer_mask"],
                        ec_mask,
                    ) * self.ec_weight
                else:
                    # Single CTH fallback
                    ec_height = batch["ec_height"]
                    ec_loss = self.ec_loss_fn(
                        h / 1000, ec_height / 1000, ec_mask,
                    ) * self.ec_weight

        return rss_loss + ec_loss, rss_loss, ec_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.forward(batch)
        h = out["h"]
        chi2 = out["chi2"]
        iter_results = out.get("iter_results", [out])

        # Multi-iteration loss with exponential weighting (γ=0.8)
        gamma = 0.8
        n_iters = len(iter_results)
        total_iter_loss = torch.tensor(0.0, device=h.device)
        for i, ir in enumerate(iter_results):
            weight = gamma ** (n_iters - 1 - i)
            iter_loss, _, _ = self._compute_iter_loss(ir["h"], ir["chi2"], batch)
            total_iter_loss = total_iter_loss + weight * iter_loss

        # Final iteration stats for logging
        valid = torch.isfinite(chi2) & torch.isfinite(h)
        _, rss_loss, ec_loss = self._compute_iter_loss(h, chi2, batch)

        # Physical bounds penalty (on final iteration only)
        if valid.any():
            h_valid = h[valid].detach()
            bounds_penalty = (
                torch.clamp(-h_valid, min=0).pow(2).mean()
                + torch.clamp(h_valid - 20000, min=0).pow(2).mean()
            ) * self.bounds_weight
        else:
            bounds_penalty = torch.tensor(0.0, device=h.device)

        # Regularization against pretrained reference
        h_reg_loss = torch.tensor(0.0, device=h.device)
        wind_reg_loss = torch.tensor(0.0, device=h.device)
        if self._needs_ref and valid.any():
            with torch.no_grad():
                ref_xsol = self._forward_ref(batch)
                h_ref = ref_xsol[..., 0]
                vu_ref = ref_xsol[..., 3]
                vv_ref = ref_xsol[..., 4]
            ref_valid = valid & torch.isfinite(h_ref) & torch.isfinite(vu_ref)

            if ref_valid.any() and self.height_reg_weight > 0:
                h_reg_loss = self._gaussian_kl(
                    h[ref_valid] / 1000, h_ref[ref_valid] / 1000,
                ) * self.height_reg_weight

            if ref_valid.any() and self.wind_reg_weight > 0:
                x_sol = out["x_sol"]
                vu_ft = x_sol[..., 3][ref_valid]
                vv_ft = x_sol[..., 4][ref_valid]
                vu_r = vu_ref[ref_valid]
                vv_r = vv_ref[ref_valid]
                wind_reg_loss = (
                    self._gaussian_kl(vu_ft, vu_r)
                    + self._gaussian_kl(vv_ft, vv_r)
                ) * self.wind_reg_weight

        ec_count = int(batch["ec_mask"].sum()) if "ec_mask" in batch else 0
        loss = total_iter_loss + bounds_penalty + h_reg_loss + wind_reg_loss

        # Logging (final iteration values)
        self.log("train/rss_loss", rss_loss, prog_bar=True)
        self.log("train/bounds_penalty", bounds_penalty)
        self.log("train/h_reg_loss", h_reg_loss)
        self.log("train/wind_reg_loss", wind_reg_loss)
        self.log("train/ec_loss", ec_loss)
        self.log("train/ec_count", float(ec_count))
        self.log("train/total_loss", loss, prog_bar=True)
        self.log("train/valid_frac", valid.float().mean())
        if valid.any():
            self.log("train/h_mean_km", h[valid].mean() / 1000)
            self.log("train/h_std_km", h[valid].std() / 1000)
            # Monitor flow magnitudes to detect collapse
            y = out["y"]
            flow_mag = y[..., :2].pow(2).sum(-1).sqrt()
            self.log("train/flow_mag_mean", flow_mag[torch.isfinite(flow_mag)].mean())

        # Log images periodically
        if self.global_step % self.log_media_step == 0 and valid.any():
            self._log_images(batch, out, valid)

        return loss

    @staticmethod
    def _gaussian_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """KL(q || p) assuming both are 1D Gaussian, estimated from samples.

        q : samples from the fine-tuned distribution (carries grad)
        p : samples from the pretrained distribution (detached)
        """
        mu_q, mu_p = q.mean(), p.mean()
        var_q = q.var() + 1e-8
        var_p = p.var() + 1e-8
        return 0.5 * (var_q / var_p + (mu_p - mu_q).pow(2) / var_p
                       - 1 + torch.log(var_p / var_q))

    @staticmethod
    def _make_grid(tiles: list[np.ndarray], border: int = 2) -> np.ndarray:
        """Concatenate images into a row grid with a white border between them."""
        padded = []
        for i, t in enumerate(tiles):
            if i > 0:
                sep = np.ones((t.shape[0], border, *t.shape[2:]), dtype=t.dtype)
                padded.append(sep)
            padded.append(t)
        return np.concatenate(padded, axis=1)

    def _log_images(
        self,
        batch: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        valid: torch.Tensor,
        n_img: int = 8,
    ) -> None:
        """Log diagnostic image grids to W&B."""
        B = min(batch["A0"].shape[0], n_img)

        def _colormap(arr, cmap, vmin, vmax):
            cm = plt_cmaps[cmap]
            normed = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
            normed = np.nan_to_num(normed, nan=0.0)
            return (cm(normed)[..., :3] * 255).astype(np.uint8)

        # y layout: [D1u, D1v, D2u, D2v, D3u, D3v, D4u, D4v]
        # D1/D2 = temporal (A0→A∓), D3/D4 = cross-sat (A0→B∓)
        # x_sol layout: [h, p_u, p_v, V_u, V_v]
        FLOW_LIM = 10  # px, symmetric for diverging cmap

        grids = {k: [] for k in (
            "input", "height_km", "chi2",
            "D1_u", "D1_v", "D2_u", "D2_v",
            "D3_u", "D3_v", "D4_u", "D4_v",
            "V_u", "V_v",
        )}
        for i in range(B):
            a0 = batch["A0"][i, 0].detach().cpu().numpy()
            h = out["h"][i].detach().cpu().numpy()
            chi2_i = out["chi2"][i].detach().cpu().numpy()
            y = out["y"][i].detach().cpu().numpy()
            x_sol = out["x_sol"][i].detach().cpu().numpy()
            vmask = valid[i].detach().cpu().numpy()

            gray = (np.clip(a0, 0, 1) * 255).astype(np.uint8)
            grids["input"].append(np.stack([gray, gray, gray], axis=-1))

            h_km = np.where(vmask, h / 1000, np.nan)
            grids["height_km"].append(_colormap(h_km, "turbo", 0, 18))

            chi2_vis = np.where(vmask, np.clip(chi2_i, 0, 5), np.nan)
            grids["chi2"].append(_colormap(chi2_vis, "hot", 0, 5))

            # 4 flow pairs — u and v components (diverging colormap)
            for j, name in enumerate(("D1_u", "D1_v", "D2_u", "D2_v",
                                      "D3_u", "D3_v", "D4_u", "D4_v")):
                grids[name].append(_colormap(y[..., j], "RdBu_r", -FLOW_LIM, FLOW_LIM))

            # Solved wind velocity (pixels/sec → show as pixels for now)
            grids["V_u"].append(_colormap(x_sol[..., 3], "RdBu_r", -FLOW_LIM, FLOW_LIM))
            grids["V_v"].append(_colormap(x_sol[..., 4], "RdBu_r", -FLOW_LIM, FLOW_LIM))

        logs = {"trainer/global_step": self.global_step}
        for key, tiles in grids.items():
            logs[f"train/images/{key}"] = wandb.Image(self._make_grid(tiles))

        # Histograms of height and solved wind velocity over valid pixels
        h_all = out["h"].detach().cpu()
        x_all = out["x_sol"].detach().cpu()
        vmask_all = valid.detach().cpu()
        if vmask_all.any():
            h_valid = h_all[vmask_all].numpy() / 1000  # km
            v_u = x_all[..., 3][vmask_all].numpy()
            v_v = x_all[..., 4][vmask_all].numpy()
            logs["train/histograms/height_km"] = wandb.Histogram(h_valid, num_bins=64)
            logs["train/histograms/V_u"] = wandb.Histogram(v_u, num_bins=64)
            logs["train/histograms/V_v"] = wandb.Histogram(v_v, num_bins=64)

        self.logger.experiment.log(logs, step=self.global_step)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.forward(batch)
        h = out["h"]
        chi2 = out["chi2"]

        valid = torch.isfinite(chi2) & torch.isfinite(h)
        if valid.any():
            rss_loss = chi2[valid].mean()
            self.log("val/rss_loss", rss_loss, prog_bar=True)
            self.log("val/h_mean_km", h[valid].mean() / 1000)

        return chi2[valid].mean() if valid.any() else torch.tensor(0.0)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=1e-6,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

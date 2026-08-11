"""Multi-scene RAFT optical flow inference for stereo wind retrieval.

Computes 4 disparity fields from 5 input scenes:
  D1: A0 → A_minus  (temporal backward, same satellite)
  D2: A0 → A_plus   (temporal forward, same satellite)
  D3: A0 → B_minus  (cross-satellite backward)
  D4: A0 → B_plus   (cross-satellite forward)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_compat_checkpoint(ckpt_path: str) -> str:
    """Convert PyTorch Lightning checkpoint to FlowRunner format if needed.

    FlowRunner expects {"model": state_dict, "global_step": int}.
    PL checkpoints have {"state_dict": ..., "global_step": ...}.
    Returns the path to the compatible checkpoint.
    """
    import torch

    ckpt_path = str(ckpt_path)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Already in compat format
    if "model" in data:
        return ckpt_path

    # PL format — convert and save alongside
    if "state_dict" in data:
        compat_path = ckpt_path.rsplit(".", 1)[0] + "_compat.pt"
        p = Path(compat_path)
        if not p.exists():
            logger.info("Converting PL checkpoint to FlowRunner format: %s", p.name)
            # Strip 'raft.' prefix from fine-tuned PL checkpoints
            sd = data["state_dict"]
            if any(k.startswith("raft.") for k in sd):
                sd = {k.removeprefix("raft."): v for k, v in sd.items()
                      if k.startswith("raft.")}
            torch.save(
                {"model": sd, "global_step": data.get("global_step", 0)},
                compat_path,
            )
        return compat_path

    # Unknown format — pass through and let FlowRunner handle it
    return ckpt_path


class StereoDisparity:
    """Wraps FlowRunner for multi-pair stereo disparity computation.

    Parameters
    ----------
    model_ckpt_path : path to RAFT checkpoint (PL or FlowRunner format)
    tile_size, overlap, batch_size, device : FlowRunner settings
    """

    def __init__(
        self,
        model_ckpt_path: str,
        tile_size: int = 512,
        overlap: int = 256,
        batch_size: int = 8,
        device: str = "cpu",
        lowmem: bool = False,
    ):
        self.lowmem = lowmem
        from stereo_winds.flow import FlowRunner

        compat_path = _ensure_compat_checkpoint(model_ckpt_path)

        self.runner = FlowRunner(
            model_ckpt_path=compat_path,
            model_name="raft",
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
        )

    def _run_pair(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Run RAFT on a single image pair.

        Parameters
        ----------
        img1, img2 : (H, W) 2D arrays

        Returns
        -------
        flow : (2, H, W) pixel displacements (du, dv)
            v-component is negated so that positive = northward,
            matching the solver convention (RAFT uses positive-down).
        """
        # FlowRunner expects (1, 1, H, W) — batch, channels, height, width
        i1 = img1[np.newaxis, np.newaxis, :, :]
        i2 = img2[np.newaxis, np.newaxis, :, :]
        flow = self.runner.forward(i1, i2, lowmem=self.lowmem)

        # Squeeze batch dimension if present
        if flow.ndim == 4:
            flow = flow[0]

        # Negate v-component: RAFT positive-down → solver positive-north
        flow[1] = -flow[1]

        return flow

    def compute_all(
        self,
        images: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute all 4 disparity fields.

        Parameters
        ----------
        images : dict with keys A_minus, A0, A_plus, B_minus, B_plus
            Each value is a (H, W) 2D array. B scenes should already
            be remapped onto A's grid.

        Returns
        -------
        disparities : dict with keys D1, D2, D3, D4
            Each value is (2, H, W) pixel displacement array.
            v-component is positive-north (solver convention).
        """
        a0 = images["A0"]

        disparities = {}

        # D1: A0 → A_minus (temporal backward)
        disparities["D1"] = self._run_pair(a0, images["A_minus"])

        # D2: A0 → A_plus (temporal forward)
        disparities["D2"] = self._run_pair(a0, images["A_plus"])

        # D3/D4 may be cross-INSTRUMENT pairs (e.g. GOES ABI vs remapped
        # MTG FCI). FlowRunner.preprocess applies per-image histogram
        # equalization, which removes inter-instrument gain/offset and
        # (monotone) calibration differences before RAFT — no additional
        # radiometric matching is needed here.

        # D3: A0 → B_minus (cross-satellite backward)
        disparities["D3"] = self._run_pair(a0, images["B_minus"])

        # D4: A0 → B_plus (cross-satellite forward)
        disparities["D4"] = self._run_pair(a0, images["B_plus"])

        return disparities

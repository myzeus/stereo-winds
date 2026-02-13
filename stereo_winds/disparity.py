"""Multi-scene RAFT optical flow inference for stereo wind retrieval.

Computes 4 disparity fields from 5 input scenes:
  D1: A0 → A_minus  (temporal backward, same satellite)
  D2: A0 → A_plus   (temporal forward, same satellite)
  D3: A0 → B_minus  (cross-satellite backward)
  D4: A0 → B_plus   (cross-satellite forward)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class StereoDisparity:
    """Wraps FlowRunner for multi-pair stereo disparity computation.

    Parameters
    ----------
    model_ckpt_path : path to RAFT checkpoint
    tile_size, overlap, batch_size, device : FlowRunner settings
    """

    def __init__(
        self,
        model_ckpt_path: str,
        tile_size: int = 512,
        overlap: int = 256,
        batch_size: int = 8,
        device: str = "cpu",
    ):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zeus"))
        from zeus.inference.inference_flows import FlowRunner

        self.runner = FlowRunner(
            model_ckpt_path=model_ckpt_path,
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
        """
        # FlowRunner expects (1, 1, H, W) — batch, channels, height, width
        i1 = img1[np.newaxis, np.newaxis, :, :]
        i2 = img2[np.newaxis, np.newaxis, :, :]
        return self.runner.forward(i1, i2)

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
        """
        a0 = images["A0"]

        disparities = {}

        # D1: A0 → A_minus (temporal backward)
        disparities["D1"] = self._run_pair(a0, images["A_minus"])

        # D2: A0 → A_plus (temporal forward)
        disparities["D2"] = self._run_pair(a0, images["A_plus"])

        # D3: A0 → B_minus (cross-satellite backward)
        disparities["D3"] = self._run_pair(a0, images["B_minus"])

        # D4: A0 → B_plus (cross-satellite forward)
        disparities["D4"] = self._run_pair(a0, images["B_plus"])

        return disparities

"""Tiled RAFT optical-flow inference for stereo wind retrieval.

A self-contained flow runner: loads a RAFT checkpoint, histogram-equalizes the
two input images, runs the network over overlapping tiles, and reassembles a
full-resolution ``(2, H, W)`` displacement field (du, dv).

Vendored and trimmed from the internal ``zeus`` ``FlowRunner`` so that
stereo-winds runs standalone. The RAFT network under ``stereo_winds.flow.raft``
is a single-channel adaptation of RAFT (Teed & Deng, ECCV 2020); the WindFlow
teacher checkpoint loads into it unchanged.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.stats
import torch
from torch import nn

from stereo_winds.flow.raft.raft import RAFT

MODEL_NAME = "raft"
RAFT_ARGS = {
    "small": False,
    "lr": 1e-4,
    "mixed_precision": False,
    "dropout": 0.0,
    "corr_levels": 4,
    "corr_radius": 4,
}
TILE_SIZE = 512
OVERLAP_SMALL = 128


def image_histogram_equalization(image: np.ndarray, number_bins: int = 500) -> np.ndarray:
    """CDF-based histogram equalization to [0, 1]; non-finite -> 0.

    Reproduces the exact preprocessing WindFlow was run with, so retrieved
    flows match the original zeus pipeline bit-for-bit.
    """
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()
    cdf = cdf / cdf[-1]
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized[~np.isfinite(image_equalized)] = 0.0
    return image_equalized.reshape(image.shape)


def histogram_equalize(image: np.ndarray, n_bins: int = 500) -> np.ndarray:
    """CDF-based histogram equalization to [0, 1], ignoring non-finite pixels.

    Differs from :func:`image_histogram_equalization`, which zeroes non-finite
    pixels *before* histogramming so those zeros shape the CDF. Here the
    histogram is built from finite pixels only and non-finite positions are set
    to 0 afterwards, so off-disk NaNs do not distort the mapping. Use this when
    preparing full-disk scenes that are largely off-earth; use
    ``image_histogram_equalization`` to reproduce WindFlow's exact
    preprocessing.
    """
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    vals = image[finite]
    hist, bins = np.histogram(vals, n_bins, density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    out = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)
    out[~finite] = 0.0
    return out.astype(np.float32)


class FlowRunner:
    """Run a RAFT flow model over large images via overlapping tiles.

    Parameters
    ----------
    model_ckpt_path : str
        Checkpoint in ``{"model": state_dict, "global_step": int}`` format
        (``disparity._ensure_compat_checkpoint`` converts Lightning checkpoints
        to this).
    tile_size, overlap, batch_size, device : tiling / hardware settings.
    """

    def __init__(
        self,
        model_ckpt_path: str,
        model_name: str = MODEL_NAME,
        tile_size: int = TILE_SIZE,
        overlap: int = OVERLAP_SMALL,
        batch_size: int = 16,
        device: str = "cpu",
    ):
        if model_name.lower() != "raft":
            raise ValueError("standalone FlowRunner only supports model_name='raft'")
        self.model_name = "raft"
        self.model = RAFT(RAFT_ARGS)

        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device
        self.model_ckpt_path = model_ckpt_path

        if self.device == "cuda:0":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.model = self.model.to(device)
        self.model = torch.nn.DataParallel(self.model)
        self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.model_ckpt_path, map_location=self.device)
        self.global_step = checkpoint.get("global_step", 0)
        try:
            self.model.module.load_state_dict(checkpoint["model"])
        except Exception:
            self.model.load_state_dict(checkpoint["model"])

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        x[~np.isfinite(x)] = 0.0
        return image_histogram_equalization(x)

    def forward(self, img1: np.ndarray, img2: np.ndarray, lowmem: bool = True) -> np.ndarray:
        mask = img1 == img1
        mask[~mask] = np.nan

        x0 = self.preprocess(img1.copy())
        x1 = self.preprocess(img2.copy())

        flows = self.inference_lowmem(x0, x1) if lowmem else self.inference(x0, x1)
        return flows * mask

    def get_pdf_for_gaussian_filter(self, x: int, trim: int = 0) -> np.ndarray:
        nsig = 3
        xrand = np.linspace(-nsig, nsig, x + 1 - trim * 2)
        kern1d = np.diff(scipy.stats.norm.cdf(xrand))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d / kern2d.sum()

    def get_model_outputs(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        try:
            flows = self.model(x0, x1, test_mode=True)[0]
        except TypeError:
            flows = self.model(torch.cat([x0, x1], 1))[0]
        return flows.detach().cpu().numpy()

    def inference(self, x0: np.ndarray, x1: np.ndarray, trim: int = 0) -> np.ndarray:
        x0_patches, upperleft = self.split_array(x0)
        x1_patches, _ = self.split_array(x1)

        x0_patches = torch.from_numpy(x0_patches).float()
        x1_patches = torch.from_numpy(x1_patches).float()

        pred = []
        for batch in range(0, x1_patches.shape[0], self.batch_size):
            x0_batch = x0_patches[batch : batch + self.batch_size].to(self.device)
            x1_batch = x1_patches[batch : batch + self.batch_size].to(self.device)
            pred.append(self.get_model_outputs(x0_batch, x1_batch))
        pred = np.concatenate(pred, 0)
        return self.reassemble_split_array(arr=pred, upperleft=upperleft, trim=trim)

    def split_array(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = arr.shape[-2:]
        patches = []
        indices = []
        step_size = self.tile_size - self.overlap
        for i in range(0, h, step_size):
            for j in range(0, w, step_size):
                i = min(i, h - self.tile_size)
                j = min(j, w - self.tile_size)
                indices.append([i, j])
                patches.append(arr[:, :, i : i + self.tile_size, j : j + self.tile_size])
        indices = np.array(indices)
        patches = np.concatenate(patches, axis=0)
        return patches, indices

    def reassemble_split_array(self, arr, upperleft, trim=0, use_2d_gaussian=True):
        if len(arr) == 0:
            raise ValueError("No patches to reassemble")

        tile_h, tile_w = arr.shape[-2:]
        full_h = int(upperleft[:, 0].max()) + tile_h
        full_w = int(upperleft[:, 1].max()) + tile_w
        shape = (2, full_h, full_w)
        counter = np.zeros(shape)
        out_sum = np.zeros(shape)

        factor = (
            self.get_pdf_for_gaussian_filter(x=tile_h, trim=trim) if use_2d_gaussian else 1
        )

        for i, x in enumerate(arr):
            ix, iy = upperleft[i]
            counter[:, ix + trim : ix + tile_h - trim, iy + trim : iy + tile_w - trim] += factor
            out = x[:, trim:-trim, trim:-trim] if trim > 0 else x
            out_sum[:, ix + trim : ix + tile_h - trim, iy + trim : iy + tile_w - trim] += out * factor

        return out_sum / counter

    def inference_lowmem(self, x0: np.ndarray, x1: np.ndarray, trim: int = 0) -> np.ndarray:
        h, w = x0.shape[-2:]

        shape = (2, h, w)
        f_sum = np.zeros(shape)
        f_counter = np.zeros((1, h, w))

        pdf = self.get_pdf_for_gaussian_filter(x=self.tile_size, trim=trim)

        window_size = self.tile_size - self.overlap
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                i = min(i, h - self.tile_size)
                j = min(j, w - self.tile_size)
                x0_sub = x0[:, :, i : i + self.tile_size, j : j + self.tile_size]
                x1_sub = x1[:, :, i : i + self.tile_size, j : j + self.tile_size]
                x0_sub = torch.from_numpy(x0_sub).float().to(self.device)
                x1_sub = torch.from_numpy(x1_sub).float().to(self.device)

                flows = self.get_model_outputs(x0_sub, x1_sub)[0]

                if trim > 0:
                    f_counter[:, i + trim : i + self.tile_size - trim, j + trim : j + self.tile_size - trim] += pdf
                    f_sum[:, i + trim : i + self.tile_size - trim, j + trim : j + self.tile_size - trim] += (
                        flows[:, trim:-trim, trim:-trim] * pdf
                    )
                else:
                    f_counter[0, i : i + self.tile_size, j : j + self.tile_size] += pdf
                    f_sum[:, i : i + self.tile_size, j : j + self.tile_size] += flows * pdf

        return f_sum / f_counter

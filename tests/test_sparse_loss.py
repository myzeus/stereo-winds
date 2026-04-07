"""Tests for sparse loss functions."""

import torch
import numpy as np

from stereo_winds.sparse_loss import SparseL1Loss, SparseHuberLoss


class TestSparseL1Loss:

    def test_basic(self):
        loss_fn = SparseL1Loss()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([1.5, 2.5, 3.5, 4.5])
        mask = torch.tensor([True, True, False, False])
        loss = loss_fn(pred, target, mask)
        assert torch.allclose(loss, torch.tensor(0.5))

    def test_empty_mask(self):
        loss_fn = SparseL1Loss()
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([3.0, 4.0])
        mask = torch.tensor([False, False])
        loss = loss_fn(pred, target, mask)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        loss_fn = SparseL1Loss()
        pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([1.5, 2.5, 3.5])
        mask = torch.tensor([True, False, True])
        loss = loss_fn(pred, target, mask)
        loss.backward()
        assert pred.grad is not None
        # Masked-out index should have zero gradient
        assert pred.grad[1].item() == 0.0
        assert pred.grad[0].abs().item() > 0

    def test_2d(self):
        loss_fn = SparseL1Loss()
        pred = torch.randn(4, 4)
        target = torch.randn(4, 4)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[1, 2] = True
        mask[3, 0] = True
        loss = loss_fn(pred, target, mask)
        expected = (torch.abs(pred[1, 2] - target[1, 2]) + torch.abs(pred[3, 0] - target[3, 0])) / 2
        assert torch.allclose(loss, expected)


class TestSparseHuberLoss:

    def test_small_errors_quadratic(self):
        """For |error| < delta, Huber ≈ 0.5 * error^2 / delta."""
        loss_fn = SparseHuberLoss(delta=1.0)
        pred = torch.tensor([0.0])
        target = torch.tensor([0.3])
        mask = torch.tensor([True])
        loss = loss_fn(pred, target, mask)
        expected = 0.5 * 0.3**2 / 1.0
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)

    def test_large_errors_linear(self):
        """For |error| >> delta, Huber ≈ |error| - delta/2."""
        loss_fn = SparseHuberLoss(delta=1.0)
        pred = torch.tensor([0.0])
        target = torch.tensor([10.0])
        mask = torch.tensor([True])
        loss = loss_fn(pred, target, mask)
        expected = 10.0 - 0.5
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_empty_mask(self):
        loss_fn = SparseHuberLoss()
        pred = torch.tensor([1.0])
        target = torch.tensor([100.0])
        mask = torch.tensor([False])
        assert loss_fn(pred, target, mask).item() == 0.0

    def test_gradient_flows(self):
        loss_fn = SparseHuberLoss(delta=2.0)
        pred = torch.randn(5, requires_grad=True)
        target = torch.randn(5)
        mask = torch.tensor([True, False, True, False, True])
        loss = loss_fn(pred, target, mask)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad[1].item() == 0.0

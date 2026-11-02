"""Bounded GPU command calibration, separate from physical task references."""

import torch


class LowTargetCommandCompensation:
    def __init__(self, count, device):
        self.bias = torch.zeros(count, device=device)
        self.capture_height = torch.zeros_like(self.bias)
        self.intended = torch.zeros(count, 3, device=device)
        self.applied = torch.zeros_like(self.bias)
        self.peak = torch.zeros_like(self.bias)

    def apply(self, rows, intended, actual, *, adapt, retain, reset, dt):
        bias = torch.where(reset, 0.0, self.bias[rows])
        low = intended[:, 2] <= 0.18
        near = (actual[:, :2] - intended[:, :2]).norm(dim=-1) <= 0.03
        calibrate = adapt & low & near
        error = actual[:, 2] - intended[:, 2]
        error = torch.where(error.abs() <= 0.002, 0.0, error)
        delta = (-2.0 * dt * error).clamp(-0.04 * dt, 0.04 * dt)
        bias = torch.where(calibrate, (bias + delta).clamp(-0.02, 0.0), bias)
        bias = torch.where(retain, bias, 0.0)
        self.bias[rows] = bias
        self.capture_height[rows] = torch.where(
            calibrate, intended[:, 2], self.capture_height[rows]
        )
        # Preserve the learned correction during closing, then release it as the
        # physical lift reference rises. The final 30 cm reference is unchanged.
        weight = (0.30 - intended[:, 2]) / (0.30 - self.capture_height[rows]).clamp_min(0.001)
        offset = bias * weight.clamp(0.0, 1.0)
        commanded = intended.clone()
        commanded[:, 2] += offset
        self.intended[rows] = intended
        self.applied[rows] = offset
        self.peak[rows] = torch.maximum(self.peak[rows], offset.abs())
        return commanded

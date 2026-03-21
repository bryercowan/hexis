"""SSIM computation and reward shaping for visual change detection."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def compute_ssim(img_a: bytes, img_b: bytes) -> float:
    """Fast SSIM on grayscale thumbnails (160x90).

    Returns 1.0 on error (assumes no change).
    """
    try:
        a = np.array(
            Image.open(io.BytesIO(img_a)).convert("L").resize((160, 90)),
            dtype=np.float64,
        )
        b = np.array(
            Image.open(io.BytesIO(img_b)).convert("L").resize((160, 90)),
            dtype=np.float64,
        )
        mu_a, mu_b = a.mean(), b.mean()
        sig_a2 = ((a - mu_a) ** 2).mean()
        sig_b2 = ((b - mu_b) ** 2).mean()
        sig_ab = ((a - mu_a) * (b - mu_b)).mean()
        C1, C2 = 6.5025, 58.5225
        num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (sig_a2 + sig_b2 + C2)
        return float(num / den) if den != 0 else 1.0
    except Exception:
        return 1.0


def ssim_to_reward(ssim: float) -> float:
    """Convert SSIM to shaped reward for RL training.

    Thresholds:
        < 0.85 → +5.0  (popup fully dismissed — large visual change)
        < 0.92 → +2.0  (moderate change)
        < 0.97 → +0.5  (small change)
        ≥ 0.97 → -0.5  (no effect — missed)
    """
    if ssim < 0.85:
        return 5.0
    if ssim < 0.92:
        return 2.0
    if ssim < 0.97:
        return 0.5
    return -0.5

"""Stuck loop detection via screenshot hash and click clustering."""

from __future__ import annotations

import hashlib
import math

CLICK_RADIUS_PX = 15
CLICK_RADIUS_COUNT = 3


class StuckDetector:
    """Detects stuck loops via screenshot hash and click clustering."""

    def __init__(self) -> None:
        self._screen_hashes: list[str] = []
        self._click_coords: list[tuple[int, int]] = []

    def record_screenshot(self, screenshot_bytes: bytes) -> bool:
        """Record screenshot hash. Returns True if same as last."""
        h = hashlib.md5(screenshot_bytes).hexdigest()
        is_same = len(self._screen_hashes) > 0 and self._screen_hashes[-1] == h
        self._screen_hashes.append(h)
        return is_same

    def record_click(self, x: int, y: int) -> bool:
        """Record click. Returns True if stuck in small radius."""
        self._click_coords.append((x, y))
        if len(self._click_coords) < CLICK_RADIUS_COUNT:
            return False
        recent = self._click_coords[-5:]
        cx = sum(p[0] for p in recent) / len(recent)
        cy = sum(p[1] for p in recent) / len(recent)
        clustered = sum(
            1 for px, py in recent
            if math.sqrt((px - cx) ** 2 + (py - cy) ** 2) < CLICK_RADIUS_PX
        )
        return clustered >= CLICK_RADIUS_COUNT

    def reset(self) -> None:
        self._screen_hashes.clear()
        self._click_coords.clear()

"""Records every action with before/after screenshots for skill distillation.

Saves a JSONL file where each line is one action. Screenshots saved as
separate JPEG files (referenced by filename, not inlined as base64).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


class TrajectoryLogger:
    """Records every action with before/after screenshots for skill distillation."""

    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir / "trajectory"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._shots_dir = self._dir / "screenshots"
        self._shots_dir.mkdir(exist_ok=True)
        self._log_path = self._dir / "actions.jsonl"
        self._log_file = open(self._log_path, "w")
        self._shot_idx = 0

    def _save_screenshot(self, screenshot_bytes: bytes) -> str:
        """Save screenshot to disk, return filename."""
        fname = f"shot_{self._shot_idx:05d}.jpg"
        (self._shots_dir / fname).write_bytes(screenshot_bytes)
        self._shot_idx += 1
        return fname

    def record_action(
        self,
        *,
        step: int,
        url: str,
        tool: str,
        action: dict,
        reasoning: str,
        screenshot_before: bytes,
        screenshot_after: bytes,
        ssim: float,
    ) -> None:
        """Record a single action with context."""
        before_fname = self._save_screenshot(screenshot_before)
        after_fname = self._save_screenshot(screenshot_after)

        entry = {
            "step": step,
            "url": url,
            "tool": tool,
            "action": action,
            "reasoning": reasoning[:500],
            "screenshot_before": before_fname,
            "screenshot_after": after_fname,
            "ssim": round(ssim, 4),
            "timestamp": time.time(),
        }
        self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()

    def close(self) -> None:
        self._log_file.close()
        log.info("Trajectory saved: %s (%d screenshots)", self._log_path, self._shot_idx)

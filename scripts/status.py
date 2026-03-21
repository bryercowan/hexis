#!/usr/bin/env python3
"""Print self-improvement pipeline status.

Usage:
    python scripts/status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cua_sl.self_improve.harness import SelfImprovementHarness


def main() -> None:
    harness = SelfImprovementHarness()
    harness.print_status()


if __name__ == "__main__":
    main()

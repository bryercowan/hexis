"""Heuristic trajectory labeling for expert classification.

Uses tool name, SSIM, action region, and reasoning length to assign
an expert label to each trajectory entry.
"""

from __future__ import annotations


def label_trajectory_entry(entry: dict) -> str:
    """Heuristic labeling of a trajectory entry.

    Uses tool name, SSIM, and action region to assign an expert label.
    Returns an expert label string (e.g. "dismiss_popups", "none").
    """
    tool = entry.get("tool", "")

    # Direct tool uses are easy
    if tool == "dismiss_popups":
        return "dismiss_popups"
    if tool == "solve_radio_modal":
        return "solve_radio_modal"

    # For computer actions, use heuristics
    action = entry.get("action", {})
    ssim = entry.get("ssim", 1.0)
    action_type = action.get("action", "")

    # Low SSIM = something changed = potentially useful training data
    if ssim > 0.98:
        return "none"  # No visual change = not a motor skill action

    # Check reasoning length — short reasoning = motor
    reasoning = entry.get("reasoning", "")
    if len(reasoning) > 200:
        return "none"  # Long reasoning = cognitive, not motor

    # Check if it's a click with large visual change
    coord = action.get("coordinate", [])
    if len(coord) == 2 and action_type in ("left_click", "click"):
        if ssim < 0.90:
            return "dismiss_popups"

    return "none"

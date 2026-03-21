"""Dynamic tool and system prompt generation from MoE policy.

Each registered expert becomes a Claude tool. When Claude calls it,
the generic execute_expert() loop runs the expert head autonomously.
"""

from __future__ import annotations

import base64
import logging

from hexis.harness.page import BenchmarkPage
from hexis.harness.ssim import compute_ssim
from hexis.model.expert_head import ACTION_NAMES
from hexis.model.policy import MoEPolicy

log = logging.getLogger(__name__)

VIEWPORT_W, VIEWPORT_H = 1280, 720

SYSTEM_BASE = """\
You are a computer-use agent completing an interactive benchmark challenge.

TASK: Navigate through all 30 steps of the benchmark. Each step has:
1. A challenge to solve (read text, find hidden content, click elements, etc.)
2. A 6-character code to enter in the yellow code form
3. Popups/overlays to dismiss (click green Dismiss/Close buttons — gray X = FAKE)
4. A radio modal to solve (scroll to find "Correct Choice", click it)

IMPORTANT RULES:
- ALL "Next"/"Continue" buttons are DECOYS — they show a red error toast. Never click them.
- The code entry form has a yellow background. Type the 6-char code and press Enter.
- After entering the code, a blocking modal appears with radio options. Scroll down in the modal
  to find the option containing "Correct Choice", click it, then look for a submit/confirm button.
- Dismiss ALL popups/overlays FIRST before attempting the challenge or code entry.
- When you see "click here 3 times" — click that element exactly 3 times.

You have the standard computer tool for clicking, typing, scrolling, etc.\
"""


def build_tools(policy: MoEPolicy) -> list[dict]:
    """Build tool list from available expert heads in the policy."""
    tools: list[dict] = [
        {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": VIEWPORT_W,
            "display_height_px": VIEWPORT_H,
        },
    ]

    for expert_name in policy.available_experts:
        subgoal = policy._expert_subgoals.get(expert_name, expert_name)
        tools.append({
            "type": "custom",
            "name": expert_name,
            "description": (
                f"Expert skill: {subgoal}. "
                f"Runs autonomously for up to 15 vision-only actions. "
                f"Returns a count of actions taken and a fresh screenshot."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_text": {
                        "type": "string",
                        "description": "Optional target text to condition the expert on.",
                    },
                },
            },
        })

    return tools


def build_system_prompt(policy: MoEPolicy) -> str:
    """Build system prompt based on available experts."""
    parts = [SYSTEM_BASE]

    experts = policy.available_experts
    if experts:
        parts.append("\nAvailable expert skills:")
        for name in experts:
            subgoal = policy._expert_subgoals.get(name, name)
            parts.append(f"- `{name}` tool: {subgoal}")
    else:
        parts.append(
            "\n- Click green Dismiss/Close buttons to dismiss popups. Gray X buttons are FAKE."
            "\n- For the radio modal: scroll down within the modal to find 'Correct Choice', click it."
        )

    return "\n".join(parts)


async def execute_expert(
    policy: MoEPolicy,
    page: BenchmarkPage,
    expert_name: str,
    max_actions: int = 15,
    conditioning_text: str | None = None,
) -> dict:
    """Generic expert execution loop — same for every expert.

    Runs the expert head autonomously with SSIM-based stopping.
    Returns dict with actions_taken, screenshot_b64.
    """
    subgoal = policy._expert_subgoals.get(expert_name, expert_name)
    no_effect = 0
    actions_taken = 0

    for _ in range(max_actions):
        pre_shot = await page.screenshot()
        result = policy.forward_from_bytes(
            pre_shot,
            subgoal=subgoal,
            conditioning_text=conditioning_text,
            expert_name=expert_name,
        )

        # High entropy = no expert matched (shouldn't happen if called correctly)
        if result["coord_entropy"].item() > 100:
            break

        # Convert to pixel action
        x = result["coords"][0, 0].item() * VIEWPORT_W
        y = result["coords"][0, 1].item() * VIEWPORT_H
        action_idx = result["action_type"][0].item()
        action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else "click"

        # Execute action
        if action_name == "click":
            await page.click_px(x, y)
        elif action_name == "scroll_up":
            await page.scroll_px(x, y, -300)
        elif action_name == "scroll_down":
            await page.scroll_px(x, y, 300)
        elif action_name == "done":
            break
        elif action_name == "wait":
            import asyncio
            await asyncio.sleep(0.5)
        else:
            await page.click_px(x, y)

        actions_taken += 1

        # SSIM check: stop if no effect for 3 consecutive actions
        post_shot = await page.screenshot()
        ssim = compute_ssim(pre_shot, post_shot)
        if ssim >= 0.99:
            no_effect += 1
            if no_effect >= 3:
                break
        else:
            no_effect = 0

    final_shot = await page.screenshot()
    return {
        "actions_taken": actions_taken,
        "screenshot_b64": base64.b64encode(final_shot).decode("ascii"),
    }

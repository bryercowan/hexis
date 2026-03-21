"""Claude teacher loop: API calls, expert dispatch, trajectory recording.

Replaces CNN dispatch with generic MoEPolicy dispatch.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path

import anthropic
from playwright.async_api import async_playwright

from hexis.harness.page import BenchmarkPage
from hexis.harness.ssim import compute_ssim
from hexis.model.policy import MoEPolicy
from hexis.self_improve.harness import SelfImprovementHarness
from hexis.self_improve.pattern_detector import PatternDetector
from hexis.teacher.stuck import StuckDetector
from hexis.teacher.tools import (
    VIEWPORT_H,
    VIEWPORT_W,
    build_system_prompt,
    build_tools,
    execute_expert,
)
from hexis.teacher.trajectory_logger import TrajectoryLogger
from hexis.teacher.window import SlidingWindow
from hexis.util.image import b64_to_content_block, screenshot_to_content_block

from hexis.data.schemas import ExpertStatus, TrajectoryRecord

log = logging.getLogger(__name__)

BENCHMARK_URL = "https://serene-frangipane-7fd25b.netlify.app"


def _register_proposal(harness: SelfImprovementHarness, proposal) -> None:
    """Register a detected pattern as a new expert in the registry."""
    existing = harness.registry.get(proposal.name)
    if existing is None:
        harness.registry.register(
            proposal.name,
            subgoal=proposal.subgoal,
            action_types=proposal.action_types,
            status=ExpertStatus.COLLECTING,
        )
        log.info("SELF-IMPROVE: Registered new expert '%s' (subgoal: %s)",
                 proposal.name, proposal.subgoal)


def _store_proposal_pairs(harness: SelfImprovementHarness, proposal) -> None:
    """Store training pairs from a proposal into the trajectory store."""
    stored = 0
    for pair in proposal.training_pairs:
        screenshot_before = pair.get("screenshot_before", b"")
        if not screenshot_before:
            continue
        if isinstance(screenshot_before, bytes):
            screenshot_b64 = base64.b64encode(screenshot_before).decode("ascii")
        elif isinstance(screenshot_before, str):
            screenshot_b64 = screenshot_before
        else:
            continue

        record = TrajectoryRecord(
            screenshot_b64=screenshot_b64,
            action=pair.get("action", {}),
            expert_label=proposal.name,
            metadata={
                "step": pair.get("step", 0),
                "url": pair.get("url", ""),
                "ssim": pair.get("ssim", 0.0),
                "source": "pattern_detector",
            },
        )
        if harness.store.add(record, force=True):
            stored += 1
    if stored > 0:
        log.info("SELF-IMPROVE: Stored %d new pairs for '%s' (total: %d)",
                 stored, proposal.name, harness.store.count(proposal.name))


async def run_benchmark(
    version: int,
    max_steps: int,
    model: str,
    headed: bool,
    policy: MoEPolicy | None = None,
    verbose: bool = False,
    improvement_harness: SelfImprovementHarness | None = None,
    min_training_pairs: int = 200,
    min_occurrences: int = 3,
    min_sft_samples: int = 20,
    router_first: bool = False,
) -> dict:
    """Run the unified Claude + expert benchmark loop.

    When router_first=True and a policy with experts is provided, each action
    is first routed through the MoE router. If the router is confident, the
    expert handles it autonomously (fast, cheap, vision-only). If not, Claude
    decides (slow, expensive, but records trajectory for future training).
    """

    client = anthropic.Anthropic()

    # Build tools/prompt from policy (or defaults if no policy)
    if policy is not None:
        system_prompt = build_system_prompt(policy)
        tools = build_tools(policy)
    else:
        from hexis.teacher.tools import SYSTEM_BASE
        system_prompt = SYSTEM_BASE + (
            "\n- Click green Dismiss/Close buttons to dismiss popups. Gray X buttons are FAKE."
            "\n- For the radio modal: scroll down within the modal to find 'Correct Choice', click it."
        )
        tools = [{
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": VIEWPORT_W,
            "display_height_px": VIEWPORT_H,
        }]

    window = SlidingWindow()
    stuck = StuckDetector()

    total_input_tokens = 0
    total_output_tokens = 0
    actions_taken = 0
    router_actions = 0
    claude_actions = 0
    start_time = time.time()
    step_start_time = time.time()
    step_actions = 0
    step_times: list[dict] = []
    expert_tool_times: list[float] = []
    claude_call_times: list[float] = []

    # Trajectory logging
    results_dir = Path("results/benchmark_unified")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    traj_logger = TrajectoryLogger(results_dir / f"v{version}_{ts}")

    # Pattern detection
    detector = PatternDetector(
        min_occurrences=min_occurrences,
        min_training_pairs=min_training_pairs,
    )
    step_action_buffer: list[dict] = []
    current_step_num = 1

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed)
        context = await browser.new_context(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
        )
        pw_page = await context.new_page()
        page = BenchmarkPage(pw_page, VIEWPORT_W, VIEWPORT_H)

        # Navigate and click START
        url = f"{BENCHMARK_URL}?version={version}"
        log.info("Navigating to %s", url)
        await pw_page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)

        start_btn = await pw_page.query_selector('button:has-text("START")')
        if start_btn:
            await start_btn.click()
            log.info("Clicked START button")
            await asyncio.sleep(1)

        # Initial screenshot
        init_screenshot = await page.screenshot()
        window.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "The benchmark has started. Complete all 30 steps."},
                screenshot_to_content_block(init_screenshot),
            ],
        })

        finished = False
        last_url = page.url

        while actions_taken < max_steps and not finished:
            # --- Router-first pass: try expert before asking Claude ---
            if router_first and policy is not None and policy.available_experts:
                pre_shot = await page.screenshot()
                routed_expert = policy.route_screenshot(pre_shot)

                if routed_expert is not None:
                    log.info(
                        "Action %d: ROUTER → %s (autonomous)",
                        actions_taken + 1, routed_expert,
                    )
                    _expert_t0 = time.time()
                    result = await execute_expert(
                        policy, page, routed_expert,
                        conditioning_text=policy._expert_subgoals.get(routed_expert),
                    )
                    _expert_dt = time.time() - _expert_t0
                    expert_tool_times.append(_expert_dt)
                    router_actions += result["actions_taken"]
                    actions_taken += result["actions_taken"]
                    step_actions += result["actions_taken"]

                    post_shot_bytes = base64.b64decode(result["screenshot_b64"])
                    ssim = compute_ssim(pre_shot, post_shot_bytes)

                    traj_entry = dict(
                        step=current_step_num, url=page.url,
                        tool=routed_expert,
                        action={"actions_taken": result["actions_taken"]},
                        reasoning=f"Router dispatched to {routed_expert}",
                        screenshot_before=pre_shot,
                        screenshot_after=post_shot_bytes, ssim=ssim,
                    )
                    traj_logger.record_action(**traj_entry)
                    step_action_buffer.append(traj_entry)
                    if improvement_harness:
                        improvement_harness.store_trajectory_entry(traj_entry)

                    log.info(
                        "  → %d actions in %.2fs (ssim=%.3f)",
                        result["actions_taken"], _expert_dt, ssim,
                    )

                    # Update conversation window so Claude has context if called next
                    window.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                f"[Router used '{routed_expert}' skill autonomously, "
                                f"{result['actions_taken']} actions taken. "
                                f"Here is the current screen.]"
                            )},
                            b64_to_content_block(result["screenshot_b64"]),
                        ],
                    })

                    # Check URL transition after expert action
                    current_url = page.url
                    if current_url != last_url:
                        if "/finish" in current_url or "finish" in current_url:
                            step_dt = time.time() - step_start_time
                            step_times.append({"step": current_step_num, "time_s": step_dt, "actions": step_actions})
                            log.info("Benchmark finished! URL: %s", current_url)
                            finished = True
                        else:
                            step_dt = time.time() - step_start_time
                            step_times.append({"step": current_step_num, "time_s": step_dt, "actions": step_actions})
                            log.info("Step %d complete: %.1fs, %d actions", current_step_num, step_dt, step_actions)
                            step_start_time = time.time()
                            step_actions = 0
                            stuck.reset()
                            if step_action_buffer:
                                proposals = detector.analyze_step(current_step_num, step_action_buffer)
                                for proposal in proposals:
                                    log.info("SELF-IMPROVE: Skill proposed: '%s' (%d pairs)", proposal.name, len(proposal.training_pairs))
                                    if improvement_harness:
                                        _register_proposal(improvement_harness, proposal)
                            if improvement_harness:
                                for name, proposal in detector.collecting_skills.items():
                                    _store_proposal_pairs(improvement_harness, proposal)
                                ready_skills = detector.skills_ready_for_training()
                                for proposal in ready_skills:
                                    log.info("SELF-IMPROVE: '%s' ready for training!", proposal.name)
                            step_action_buffer = []
                            try:
                                current_step_num = int(current_url.split("/")[-1].split("?")[0].replace("step", ""))
                            except (ValueError, IndexError):
                                current_step_num += 1
                        last_url = current_url

                    continue  # Skip Claude call, try router again next iteration

            # --- Claude fallback: router declined or not available ---
            claude_actions += 1
            # Call Claude
            try:
                _claude_t0 = time.time()
                response = client.beta.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=window.context,
                    tools=tools,
                    betas=["computer-use-2025-01-24"],
                )
                _claude_dt = time.time() - _claude_t0
                claude_call_times.append(_claude_dt)
            except anthropic.RateLimitError as e:
                retry_after = getattr(e, "retry_after", 5) or 5
                log.warning("Rate limited, waiting %ds", retry_after)
                await asyncio.sleep(retry_after)
                continue
            except anthropic.InternalServerError:
                log.warning("API 500 error — retrying in 3s")
                await asyncio.sleep(3)
                continue

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            assistant_content = response.content
            if not assistant_content:
                continue

            window.append({"role": "assistant", "content": assistant_content})

            # Extract reasoning text
            reasoning_text = ""
            for block in assistant_content:
                if hasattr(block, "text") and block.text:
                    reasoning_text += block.text + " "
            reasoning_text = reasoning_text.strip()

            # Process tool uses
            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input if hasattr(block, "input") else {}
                actions_taken += 1
                step_actions += 1

                if tool_name == "computer":
                    # Execute computer action
                    action_str = tool_input.get("action", "unknown")
                    coord = tool_input.get("coordinate", [])
                    log.info(
                        "Action %d: computer.%s%s",
                        actions_taken, action_str,
                        f" ({coord[0]}, {coord[1]})" if len(coord) == 2 else "",
                    )

                    pre_shot = await page.screenshot()
                    await page.execute_computer_action(tool_input)
                    post_shot = await page.screenshot()

                    ssim = compute_ssim(pre_shot, post_shot)
                    traj_entry = dict(
                        step=current_step_num, url=page.url,
                        tool="computer", action=tool_input,
                        reasoning=reasoning_text,
                        screenshot_before=pre_shot,
                        screenshot_after=post_shot, ssim=ssim,
                    )
                    traj_logger.record_action(**traj_entry)
                    step_action_buffer.append(traj_entry)

                    if improvement_harness:
                        improvement_harness.store_trajectory_entry(traj_entry)

                    # Stuck detection
                    warnings = []
                    if stuck.record_screenshot(post_shot):
                        warnings.append(
                            "WARNING: Screen unchanged after action. "
                            "Try a different approach or location."
                        )
                    if action_str in ("left_click", "double_click") and len(coord) == 2:
                        if stuck.record_click(int(coord[0]), int(coord[1])):
                            warnings.append(
                                "WARNING: You've been clicking in the same area repeatedly. "
                                "Try a completely different part of the screen."
                            )

                    result_content = [screenshot_to_content_block(post_shot)]
                    if warnings:
                        result_content.insert(
                            0, {"type": "text", "text": "\n".join(warnings)},
                        )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_content,
                    })

                elif policy is not None and tool_name in policy.available_experts:
                    # Execute expert tool
                    log.info("Action %d: %s (expert skill)", actions_taken, tool_name)
                    pre_shot = await page.screenshot()
                    _expert_t0 = time.time()
                    result = await execute_expert(
                        policy, page, tool_name,
                        conditioning_text=tool_input.get("target_text"),
                    )
                    _expert_dt = time.time() - _expert_t0
                    expert_tool_times.append(_expert_dt)
                    log.info(
                        "  → %d actions in %.2fs",
                        result["actions_taken"], _expert_dt,
                    )
                    post_shot_bytes = base64.b64decode(result["screenshot_b64"])
                    expert_entry = dict(
                        step=current_step_num, url=page.url,
                        tool=tool_name,
                        action={"actions_taken": result["actions_taken"]},
                        reasoning=reasoning_text,
                        screenshot_before=pre_shot,
                        screenshot_after=post_shot_bytes,
                        ssim=compute_ssim(pre_shot, post_shot_bytes),
                    )
                    traj_logger.record_action(**expert_entry)
                    step_action_buffer.append(expert_entry)
                    if improvement_harness:
                        improvement_harness.store_trajectory_entry(expert_entry)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [
                            {"type": "text", "text": f"Expert '{tool_name}' ran {result['actions_taken']} actions."},
                            b64_to_content_block(result["screenshot_b64"]),
                        ],
                    })

                else:
                    log.warning("Unknown tool: %s", tool_name)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                        "is_error": True,
                    })

            if tool_results:
                window.append({"role": "user", "content": tool_results})

            # Check for step transition
            current_url = page.url
            if current_url != last_url:
                if "/finish" in current_url or "finish" in current_url:
                    step_dt = time.time() - step_start_time
                    step_times.append({"step": current_step_num, "time_s": step_dt, "actions": step_actions})
                    log.info("Benchmark finished! URL: %s", current_url)
                    finished = True
                else:
                    step_dt = time.time() - step_start_time
                    step_times.append({"step": current_step_num, "time_s": step_dt, "actions": step_actions})
                    log.info("Step %d complete: %.1fs, %d actions", current_step_num, step_dt, step_actions)
                    step_start_time = time.time()
                    step_actions = 0
                    stuck.reset()

                    # Pattern detection → register proposals → store training pairs
                    if step_action_buffer:
                        proposals = detector.analyze_step(current_step_num, step_action_buffer)
                        for proposal in proposals:
                            log.info(
                                "SELF-IMPROVE: Skill proposed: '%s' (%d pairs, %d occurrences)",
                                proposal.name, len(proposal.training_pairs), proposal.occurrences,
                            )
                            # Register in expert registry
                            if improvement_harness:
                                _register_proposal(improvement_harness, proposal)

                    # Store training pairs from collecting skills
                    if improvement_harness:
                        for name, proposal in detector.collecting_skills.items():
                            _store_proposal_pairs(improvement_harness, proposal)

                        # Check if any skills hit training threshold
                        ready_skills = detector.skills_ready_for_training()
                        for proposal in ready_skills:
                            log.info(
                                "SELF-IMPROVE: '%s' has %d pairs — ready for training!",
                                proposal.name, len(proposal.training_pairs),
                            )

                    step_action_buffer = []
                    try:
                        current_step_num = int(
                            current_url.split("/")[-1].split("?")[0].replace("step", "")
                        )
                    except (ValueError, IndexError):
                        current_step_num += 1

                last_url = current_url

            # Prompt to continue if Claude ended without tool use
            if response.stop_reason == "end_turn" and not tool_results:
                shot = await page.screenshot()
                window.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Continue with the next action."},
                        screenshot_to_content_block(shot),
                    ],
                })

        elapsed = time.time() - start_time
        cost = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000

        avg_step_time = sum(s["time_s"] for s in step_times) / max(len(step_times), 1)
        avg_claude_time = sum(claude_call_times) / max(len(claude_call_times), 1)

        log.info("=== TIMING SUMMARY ===")
        log.info("Steps completed: %d", len(step_times))
        log.info("Avg time/step: %.1fs", avg_step_time)
        log.info("Avg Claude API call: %.2fs", avg_claude_time)
        if expert_tool_times:
            log.info("Expert tool invocations: %d", len(expert_tool_times))
        if router_first:
            log.info("Router-handled actions: %d", router_actions)
            log.info("Claude-fallback actions: %d", claude_actions)
            if router_actions + claude_actions > 0:
                log.info("Router autonomy: %.0f%%",
                         100 * router_actions / (router_actions + claude_actions))

        # --- Self-improvement: end-of-run training trigger ---
        training_results = {}
        if improvement_harness:
            # Final store flush for any remaining collecting skills
            for name, proposal in detector.collecting_skills.items():
                _store_proposal_pairs(improvement_harness, proposal)

            # Report collection status
            log.info("=== SELF-IMPROVEMENT STATUS ===")
            improvement_harness.print_status()

            # Trigger training for skills with enough data
            deployed_experts = []
            for expert_info in improvement_harness.registry.list_all():
                name = expert_info["name"]
                n_samples = improvement_harness.store.count(name)
                if n_samples >= min_sft_samples:
                    log.info(
                        "SELF-IMPROVE: Training '%s' (%d samples >= %d threshold)",
                        name, n_samples, min_sft_samples,
                    )
                    try:
                        ckpt_path, data_req = improvement_harness.train_expert(name)
                        training_results[name] = {
                            "checkpoint": ckpt_path,
                            "data_request": data_req.to_dict() if data_req else None,
                            "samples": n_samples,
                        }
                        if ckpt_path:
                            log.info("SELF-IMPROVE: '%s' trained → %s", name, ckpt_path)
                            # Auto-deploy
                            improvement_harness.deploy_expert(name, ckpt_path, policy)
                            deployed_experts.append(name)
                            training_results[name]["deployed"] = True
                        elif data_req:
                            log.info("SELF-IMPROVE: '%s' needs more data: %s",
                                     name, data_req.reason)
                    except Exception as e:
                        log.error("SELF-IMPROVE: Training '%s' failed: %s", name, e)
                        training_results[name] = {"error": str(e), "samples": n_samples}
                else:
                    log.info(
                        "SELF-IMPROVE: '%s' has %d samples (need %d) — skipping training",
                        name, n_samples, min_sft_samples,
                    )
                    training_results[name] = {"skipped": True, "samples": n_samples}

            # Train router whenever we have ANY expert label data + none data
            # The router must learn: "this → expert X" AND "this → ask Claude"
            router_result = None
            expert_labels = [l for l in improvement_harness.store.labels()
                            if l != "none" and improvement_harness.store.count(l) >= min_sft_samples]
            has_none = improvement_harness.store.count("none") > 0
            if len(expert_labels) >= 1 and has_none:
                log.info("SELF-IMPROVE: Training router on %d expert labels + none: %s",
                         len(expert_labels), expert_labels)
                router_ckpt = improvement_harness.train_router()
                if router_ckpt:
                    router_result = {"checkpoint": router_ckpt}
                    log.info("SELF-IMPROVE: Router trained → %s", router_ckpt)
                    if policy is not None:
                        try:
                            policy.load_router(router_ckpt)
                            log.info("SELF-IMPROVE: Router hot-loaded into policy")
                        except Exception as e:
                            log.warning("SELF-IMPROVE: Failed to load router: %s", e)
            training_results["_router"] = router_result

        result = {
            "version": version,
            "actions": actions_taken,
            "max_steps": max_steps,
            "finished": finished,
            "elapsed_s": round(elapsed, 1),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost_usd": round(cost, 4),
            "model": model,
            "experts": policy.available_experts if policy else [],
            "final_url": page.url,
            "timing": {
                "avg_step_time_s": round(avg_step_time, 1),
                "avg_claude_call_s": round(avg_claude_time, 2),
                "expert_invocations": len(expert_tool_times),
                "steps": step_times,
            },
            "self_improve": {
                "proposals": [
                    {"name": p.name, "pairs": len(p.training_pairs), "occurrences": p.occurrences}
                    for p in detector.proposals
                ],
                "collecting": {
                    name: {
                        "pairs": len(p.training_pairs),
                        "progress": f"{p.collection_progress:.0%}",
                    }
                    for name, p in detector.collecting_skills.items()
                },
                "trajectory_store": improvement_harness.store.stats() if improvement_harness else {},
                "training": training_results,
            },
        }

        log.info("=== RESULTS ===")
        log.info(json.dumps(result, indent=2))

        run_dir = results_dir / f"v{version}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))

        traj_logger.close()
        await browser.close()

    return result

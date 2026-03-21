"""Expert RL: REINFORCE + SSIM reward on live browser.

Starting from an SFT checkpoint, trains the expert head using REINFORCE
with SSIM-based reward. The frozen Qwen-VL backbone extracts features;
only the tiny expert head (~2M params) is updated.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from hexis.harness.ssim import compute_ssim, ssim_to_reward
from hexis.model.backbone import VLMBackbone
from hexis.model.expert_head import ACTION_NAMES, ExpertActionHead

log = logging.getLogger(__name__)

BENCHMARK_URL = "https://serene-frangipane-7fd25b.netlify.app"
VIEWPORT_W, VIEWPORT_H = 1280, 720


def compute_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Discounted cumulative returns (reversed)."""
    out: list[float] = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        out.insert(0, R)
    return out


def save_rl_checkpoint(
    expert: ExpertActionHead,
    optimizer: torch.optim.Optimizer,
    episode: int,
    path: Path,
    subgoal: str,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expert_state_dict": expert.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "subgoal": subgoal,
            "backbone_dim": expert.backbone_dim,
            "expert_dim": expert.expert_dim,
            "num_action_types": expert.num_action_types,
        },
        path / "expert.pt",
    )
    log.info("Saved RL checkpoint → %s (ep=%d)", path, episode)


def train_rl(
    backbone: VLMBackbone,
    expert: ExpertActionHead,
    subgoal: str,
    output_dir: str | Path,
    episodes: int = 2000,
    max_actions: int = 15,
    lr: float = 1e-4,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    kl_coeff: float = 0.1,
    max_grad_norm: float = 5.0,
    temp_start: float = 1.0,
    temp_end: float = 0.3,
    warmup: int = 500,
    headed: bool = False,
    log_interval: int = 10,
    save_interval: int = 100,
) -> dict:
    """Run RL training loop on live benchmark.

    Returns dict with avg_reward, best_avg_reward, checkpoint_path.
    """
    from playwright.sync_api import sync_playwright

    device = next(expert.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frozen reference policy (for KL penalty)
    ref_expert = copy.deepcopy(expert)
    ref_expert.eval()
    for p in ref_expert.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=lr * 0.01,
    )

    reward_window: deque[float] = deque(maxlen=100)
    best_avg_reward = -float("inf")
    warmup_eps = min(warmup, episodes // 2)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        ctx = browser.new_context(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
        page = ctx.new_page()

        train_start = time.time()
        total_dismissals = 0

        for episode in range(episodes):
            ep_start = time.time()

            # Temperature annealing
            if episode < warmup_eps:
                temperature = temp_start - (temp_start - temp_end) * (episode / max(warmup_eps, 1))
            else:
                temperature = temp_end

            version = (episode % 3) + 1

            try:
                page.goto(
                    f"{BENCHMARK_URL}?version={version}",
                    wait_until="domcontentloaded",
                    timeout=20000,
                )
                page.wait_for_timeout(1200)
                start_btn = page.query_selector('button:has-text("START")')
                if start_btn:
                    start_btn.click()
                    page.wait_for_timeout(600)
            except Exception as e:
                log.warning("ep=%d navigation error: %s", episode, e)
                continue

            transitions: list[dict] = []
            ep_rewards: list[float] = []
            ep_ssims: list[float] = []

            for step in range(max_actions):
                try:
                    before = page.screenshot(type="jpeg", quality=80)
                except Exception:
                    break

                with torch.no_grad():
                    img_in = backbone.preprocess_image(before)
                    vis, gh, gw = backbone.vision_features(
                        img_in["pixel_values"], img_in["image_grid_thw"],
                    )
                    ids, mask = backbone.tokenize_subgoal(subgoal)
                    txt = backbone.text_features(ids, mask)

                rl_out = expert.forward_rl(
                    vis, txt, gh, gw,
                    temperature=temperature,
                    action_temperature=1.0,
                )

                x_px = int(rl_out["coords"][0, 0].item() * VIEWPORT_W)
                y_px = int(rl_out["coords"][0, 1].item() * VIEWPORT_H)
                x_px = max(0, min(x_px, VIEWPORT_W - 1))
                y_px = max(0, min(y_px, VIEWPORT_H - 1))

                action_idx = rl_out["action_type"][0].item()
                action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else "click"

                try:
                    if action_name == "click":
                        page.mouse.click(x_px, y_px)
                    elif action_name == "scroll_up":
                        page.mouse.move(x_px, y_px)
                        page.mouse.wheel(0, -300)
                    elif action_name == "scroll_down":
                        page.mouse.move(x_px, y_px)
                        page.mouse.wheel(0, 300)
                    elif action_name == "done":
                        break
                    elif action_name == "wait":
                        page.wait_for_timeout(500)
                    else:
                        page.mouse.click(x_px, y_px)
                    page.wait_for_timeout(250)
                    after = page.screenshot(type="jpeg", quality=80)
                except Exception:
                    break

                ssim = compute_ssim(before, after)
                reward = ssim_to_reward(ssim)

                ref_coord_lp = None
                if kl_coeff > 0:
                    with torch.no_grad():
                        ref_out = ref_expert.forward_rl(
                            vis, txt, gh, gw,
                            temperature=temperature,
                            action_temperature=1.0,
                        )
                        ref_coord_lp = ref_out["coord_log_prob"].detach()

                transitions.append({
                    "coord_lp": rl_out["coord_log_prob"],
                    "action_lp": rl_out["action_type_log_prob"],
                    "coord_ent": rl_out["coord_entropy"],
                    "action_ent": rl_out["action_type_entropy"],
                    "ref_coord_lp": ref_coord_lp,
                })
                ep_rewards.append(reward)
                ep_ssims.append(ssim)

                if reward >= 5.0:
                    total_dismissals += 1

            if not transitions:
                continue

            # REINFORCE update
            returns = compute_returns(ep_rewards, gamma=gamma)
            ret_arr = np.array(returns)
            ret_std = ret_arr.std()
            if ret_std > 1e-6:
                ret_arr = (ret_arr - ret_arr.mean()) / ret_std
            else:
                ret_arr = ret_arr - ret_arr.mean()
            returns_norm = ret_arr.tolist()

            optimizer.zero_grad()
            policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            entropy_sum = torch.tensor(0.0, device=device)
            kl_sum = torch.tensor(0.0, device=device)

            for t, adv in zip(transitions, returns_norm):
                lp = t["coord_lp"].squeeze() + t["action_lp"].squeeze()
                policy_loss = policy_loss + (-adv * lp)
                entropy_sum = entropy_sum + t["coord_ent"].squeeze() + t["action_ent"].squeeze()

            if kl_coeff > 0:
                for t in transitions:
                    ref_lp = t.get("ref_coord_lp")
                    if ref_lp is not None:
                        kl = t["coord_lp"].squeeze() - ref_lp.squeeze()
                        kl_sum = kl_sum + kl

            n = float(len(transitions))
            loss = (
                policy_loss / n
                - entropy_coeff * entropy_sum / n
                + kl_coeff * kl_sum / n
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            ep_reward = sum(ep_rewards)
            reward_window.append(ep_reward)
            avg100 = float(np.mean(list(reward_window)))

            if episode % log_interval == 0 or episode == episodes - 1:
                log.info(
                    "ep=%d v=%d  reward=%.1f  avg_ssim=%.3f  temp=%.2f  avg100=%.1f  [%.1fs]",
                    episode, version, ep_reward,
                    float(np.mean(ep_ssims)),
                    temperature, avg100, time.time() - ep_start,
                )

            if episode > 0 and episode % save_interval == 0:
                save_rl_checkpoint(expert, optimizer, episode, output_dir / "latest", subgoal)

            if avg100 > best_avg_reward and len(reward_window) >= 20:
                best_avg_reward = avg100
                save_rl_checkpoint(expert, optimizer, episode, output_dir / "best", subgoal)

        browser.close()

    save_rl_checkpoint(expert, optimizer, episodes, output_dir / "final", subgoal)

    elapsed = time.time() - train_start
    log.info(
        "RL training complete: %d episodes in %.1f min, "
        "total_dismissals=%d, best_avg100=%.1f",
        episodes, elapsed / 60, total_dismissals, best_avg_reward,
    )

    return {
        "avg_reward": float(np.mean(list(reward_window))) if reward_window else 0.0,
        "best_avg_reward": best_avg_reward,
        "checkpoint_path": str(output_dir / "best"),
        "total_dismissals": total_dismissals,
    }

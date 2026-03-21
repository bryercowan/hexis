# CUA-SL: Self-Learning Computer Use Agent

A vision-only OS-level computer use agent that learns from a teacher (Claude) and distills repetitive skills into tiny expert heads. Over time, a learned router handles more actions autonomously while falling back to the teacher for novel situations.

## How It Works

```
Screenshot → Router → Expert (fast, free, vision-only)
                  ↓
              Unsure? → Claude decides (slow, $$, but records trajectory)
                            ↓
                    Pattern detector proposes new skill
                            ↓
                    SFT + RL autoresearch trains expert
                            ↓
                    Router retrains, handles it next time
```

**The flywheel**: Claude solves tasks. The system watches, detects repetitive motor patterns (scroll-click sequences, popup dismissals, form fills), and distills them into ~3M parameter expert heads. A router learns when to use each expert vs. when to ask Claude. Each run, the router handles more and Claude handles less.

## Architecture

**Frozen backbone**: Qwen3-VL-4B extracts 2560-dim vision/text features. Shared by all experts, never trained. Runs on MPS (Apple Silicon) or CUDA.

**Expert heads** (~3.3M params, ~13MB each): Cross-attention over vision+text tokens → spatial heatmap for click coordinates + action type classifier. Independently trainable, hot-loadable.

**MoE Router**: Learned query adapter maps pooled vision+text features to expert embeddings. Confidence threshold gates selection — below threshold means "I haven't seen this, ask the teacher." Retrains incrementally as new experts are added.

**Self-improvement loop**:
1. **Collect**: Claude solves benchmark, every action recorded with before/after screenshots + SSIM
2. **Detect**: Pattern detector finds scroll-click, dismiss, form-fill sequences across steps
3. **Train**: SFT on heatmap targets, then REINFORCE on live browser with SSIM-shaped rewards
4. **Deploy**: Expert hot-loaded into policy, router retrained on all trajectory data
5. **Repeat**: Next run, router handles known patterns, Claude only sees novel situations

## Quick Start

```bash
# Install
pip install -e .
playwright install chromium

# Run teacher loop (Claude collects trajectories, trains experts automatically)
python -m hexis \
    --version 1 \
    --max-steps 250 \
    --use-harness \
    --min-training-pairs 15 \
    --min-occurrences 2 \
    --min-sft-samples 10 \
    -v

# Run with trained experts (router-first mode)
python -m hexis \
    --version 1 \
    --max-steps 250 \
    --expert scroll_click_select=checkpoints/self_improve/scroll_click_select/rl/round_0/best \
    --expert click_cluster_dismiss=checkpoints/self_improve/click_cluster_dismiss/rl/round_0/best \
    --expert form_fill_submit=checkpoints/self_improve/form_fill_submit/rl/round_0/best \
    --router-first \
    --use-harness \
    -v
```

## Project Structure

```
cua_sl/
├── model/
│   ├── backbone.py       # Frozen Qwen3-VL-4B feature extractor
│   ├── expert_head.py    # ~3.3M param expert (cross-attn + heatmap + action head)
│   ├── router.py         # Learned MoE router with confidence gating
│   └── policy.py         # Wraps backbone + router + expert dict
├── training/
│   ├── sft.py            # Heatmap KL loss, feature caching, EMA
│   ├── rl.py             # REINFORCE + SSIM reward on live Playwright browser
│   └── router_sft.py     # Router query adapter cross-entropy training
├── teacher/
│   ├── loop.py           # Main loop: router-first dispatch, Claude fallback
│   ├── tools.py          # Dynamic tool/prompt generation from policy
│   ├── trajectory_logger.py
│   ├── window.py         # Sliding window conversation manager
│   └── stuck.py          # Screenshot hash + click cluster detection
├── self_improve/
│   ├── harness.py        # Top-level orchestrator
│   ├── pattern_detector.py  # Detects scroll-click, dismiss, form-fill patterns
│   ├── labeler.py        # Heuristic trajectory labeling
│   └── improvement_loop.py  # SFT→RL autoresearch with plateau detection
├── data/
│   ├── schemas.py        # TrajectoryRecord, ExpertStatus, CheckpointRecord
│   ├── trajectory_store.py  # JSONL-per-label with SHA-256 dedup
│   └── registry.py       # Expert lifecycle state machine
├── harness/
│   ├── ssim.py           # SSIM computation + reward shaping
│   └── page.py           # Async Playwright page adapter
└── util/
    └── image.py          # Base64, PIL, Anthropic content blocks
```

## Key Design Decisions

**Vision-only at inference**: No DOM queries, no text extraction. The expert sees a screenshot and outputs coordinates + action type. This generalizes across any GUI.

**Frozen backbone, tiny experts**: The 4B backbone loads once (~8GB). Each expert is 3.3M params (~13MB). You can have dozens of experts with negligible memory overhead.

**SSIM-based reward shaping**: RL reward is derived from visual change between before/after screenshots. Large change (popup dismissed) = +5.0, no change = -0.5. No task-specific reward engineering.

**Autoresearch training**: Trains in rounds (5 epochs SFT / 200 episodes RL per round). Detects plateaus and overfitting automatically. Requests more data when stuck. No human in the loop.

**Edge-first**: Qwen3-VL-4B chosen for edge deployment. The full inference path (backbone + router + expert) runs on a single device with no API calls.

## Requirements

- Python >= 3.11
- PyTorch >= 2.2
- transformers >= 4.45
- playwright >= 1.40
- anthropic >= 0.39 (for teacher mode only)

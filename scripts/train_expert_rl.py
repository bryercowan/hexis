#!/usr/bin/env python3
"""CLI: RL fine-tune an expert head on live browser.

Usage:
    python scripts/train_expert_rl.py \
        --checkpoint checkpoints/popup_sft/best \
        --subgoal "dismiss popup green button" \
        --episodes 2000 --output checkpoints/popup_rl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from cua_sl.model.backbone import VLMBackbone
from cua_sl.model.expert_head import EXPERT_DIM, NUM_ACTION_TYPES, ExpertActionHead
from cua_sl.training.rl import train_rl

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL fine-tune an expert head")
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint path")
    parser.add_argument("--subgoal", required=True, help="Expert subgoal text")
    parser.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", default="checkpoints/expert_rl")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--version", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    log.info("Loading backbone: %s", args.backbone)
    backbone = VLMBackbone(model_name=args.backbone)

    # Load expert from SFT checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path / "expert.pt" if ckpt_path.is_dir() else ckpt_path,
                      map_location=device, weights_only=True)
    expert = ExpertActionHead(
        backbone_dim=backbone.hidden_dim,
        expert_dim=EXPERT_DIM,
        num_action_types=NUM_ACTION_TYPES,
    ).to(device)
    expert.load_state_dict(ckpt["expert_state_dict"])
    log.info("Loaded SFT checkpoint from %s", args.checkpoint)

    result = train_rl(
        backbone=backbone,
        expert=expert,
        subgoal=args.subgoal,
        output_dir=args.output,
        episodes=args.episodes,
        max_actions=args.max_actions,
        lr=args.lr,
        headed=args.headed,
    )

    log.info("Result: %s", json.dumps(result, indent=2))
    # Print JSON for automation (improvement loop parses this)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

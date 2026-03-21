#!/usr/bin/env python3
"""CLI: retrain the MoE router query adapter.

Usage:
    python scripts/train_router.py \
        --data data/router_train.jsonl \
        --expert dismiss_popups="dismiss popup green button" \
        --expert solve_radio_modal="solve the radio form modal" \
        --output checkpoints/router
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cua_sl.model.backbone import VLMBackbone
from cua_sl.model.router import MoERouter
from cua_sl.training.router_sft import parse_expert_specs, train_router

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MoE router query adapter")
    parser.add_argument("--data", required=True, help="Training JSONL path")
    parser.add_argument(
        "--expert", action="append", default=[],
        help="Expert as name=subgoal (repeatable)",
    )
    parser.add_argument("--none-data", help="Negative examples JSONL path")
    parser.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", default="checkpoints/router")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.expert:
        log.error("At least one --expert name=subgoal required")
        sys.exit(1)

    expert_map = parse_expert_specs(args.expert)

    log.info("Loading backbone: %s", args.backbone)
    backbone = VLMBackbone(model_name=args.backbone)

    router = MoERouter(backbone)
    for name, subgoal in expert_map.items():
        router.register_expert(name, subgoal)
        log.info("Registered expert: %s → '%s'", name, subgoal)

    best_loss = train_router(
        backbone=backbone,
        router=router,
        expert_map=expert_map,
        data_path=args.data,
        output_dir=args.output,
        none_data_path=args.none_data,
        epochs=args.epochs,
        lr=args.lr,
    )

    log.info("Training complete. Best loss: %.4f", best_loss)
    log.info("Checkpoint saved to %s", args.output)


if __name__ == "__main__":
    main()

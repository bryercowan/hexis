#!/usr/bin/env python3
"""CLI: SFT an expert head.

Usage:
    python scripts/train_expert_sft.py \
        --data data/popup_train.jsonl \
        --val-data data/popup_val.jsonl \
        --expert dismiss_popups \
        --subgoal "dismiss popup green button" \
        --epochs 20 --output checkpoints/popup_sft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from cua_sl.model.backbone import VLMBackbone
from cua_sl.model.expert_head import EXPERT_DIM, NUM_ACTION_TYPES, ExpertActionHead
from cua_sl.training.sft import (
    EMAModel,
    ExpertSFTDataset,
    extract_features,
    run_epoch,
    save_expert,
)

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT train an expert head")
    parser.add_argument("--data", required=True, help="Training JSONL path")
    parser.add_argument("--val-data", help="Validation JSONL path")
    parser.add_argument("--expert", required=True, help="Expert name")
    parser.add_argument("--subgoal", required=True, help="Conditioning text")
    parser.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", default="checkpoints/expert_sft")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--centroid-weight", type=float, default=0.5)
    parser.add_argument("--resume", help="Resume from checkpoint path")
    parser.add_argument("--feature-cache", help="Path to cache extracted features")
    parser.add_argument("--json-output", action="store_true",
                        help="Print JSON result on last line for automation")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load backbone
    log.info("Loading backbone: %s", args.backbone)
    backbone = VLMBackbone(model_name=args.backbone)

    # Load data
    train_ds = ExpertSFTDataset(args.data, args.subgoal)
    log.info("Train samples: %d", len(train_ds))

    val_ds = None
    if args.val_data:
        val_ds = ExpertSFTDataset(args.val_data, args.subgoal)
        log.info("Val samples: %d", len(val_ds))

    # Extract features
    train_features, grid_h, grid_w = extract_features(
        train_ds, backbone, cache_path=args.feature_cache,
    )
    val_features = None
    if val_ds:
        val_cache = f"{args.feature_cache}_val" if args.feature_cache else None
        val_features, _, _ = extract_features(val_ds, backbone, cache_path=val_cache)

    # Create expert head
    expert = ExpertActionHead(
        backbone_dim=backbone.hidden_dim,
        expert_dim=EXPERT_DIM,
        num_action_types=NUM_ACTION_TYPES,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        expert.load_state_dict(ckpt["expert_state_dict"])
        log.info("Resumed from %s", args.resume)

    ema = EMAModel(expert)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_path = None

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            expert, train_features, grid_h, grid_w, device,
            batch_size=args.batch_size, sigma=args.sigma,
            centroid_weight=args.centroid_weight,
            optimizer=optimizer, ema=ema,
        )
        train_loss = train_metrics["loss"]
        train_px = train_metrics["px_error"]

        val_loss, val_px = float("inf"), float("inf")
        if val_features:
            val_metrics = run_epoch(
                ema.shadow, val_features, grid_h, grid_w, device,
                batch_size=args.batch_size, sigma=args.sigma,
                centroid_weight=args.centroid_weight,
            )
            val_loss = val_metrics["loss"]
            val_px = val_metrics["px_error"]

        log.info(
            "Epoch %d/%d: train_loss=%.4f train_px=%.1f val_loss=%.4f val_px=%.1f",
            epoch + 1, args.epochs, train_loss, train_px, val_loss, val_px,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best"
            save_expert(ema.shadow, ema, args.subgoal, best_path)
            log.info("  Saved best → %s", best_path)

    # Save final
    final_path = output_dir / "final"
    save_expert(expert, ema, args.subgoal, final_path)

    result = {
        "train_loss": round(train_loss, 6),
        "val_loss": round(best_val_loss, 6),
        "val_px_error": round(val_px, 1),
        "checkpoint_path": str(best_path or final_path),
        "epochs": args.epochs,
        "expert": args.expert,
    }
    if args.json_output:
        print(json.dumps(result))
    else:
        log.info("Result: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Hexis CLI entry point. Run with: python -m hexis"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from hexis.teacher.loop import run_benchmark

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hexis: self-learning computer use agent",
    )
    parser.add_argument("--version", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--expert", action="append", default=[],
        help="Expert as name=checkpoint_path (repeatable)",
    )
    parser.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--use-harness", action="store_true",
                        help="Enable self-improvement harness")
    parser.add_argument("--min-training-pairs", type=int, default=200,
                        help="Min training pairs before SFT triggers (default 200)")
    parser.add_argument("--min-occurrences", type=int, default=3,
                        help="Min pattern occurrences before proposing a skill (default 3)")
    parser.add_argument("--min-sft-samples", type=int, default=20,
                        help="Min samples to start SFT training (default 20)")
    parser.add_argument("--router-first", action="store_true",
                        help="Router-first mode: experts handle what they can, Claude fallback")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    policy = None
    if args.expert:
        from hexis.model.backbone import VLMBackbone
        from hexis.model.policy import MoEPolicy

        backbone = VLMBackbone(model_name=args.backbone)
        policy = MoEPolicy(backbone)

        for spec in args.expert:
            if "=" not in spec:
                log.error("Invalid --expert '%s', expected name=checkpoint_path", spec)
                sys.exit(1)
            name, ckpt_path = spec.split("=", 1)
            try:
                policy.load_expert(name, ckpt_path)
                log.info("Loaded expert '%s' from %s", name, ckpt_path)
            except FileNotFoundError as e:
                log.warning("Expert '%s' not found: %s — skipping", name, e)

    harness = None
    if args.use_harness:
        from hexis.self_improve.harness import SelfImprovementHarness
        harness = SelfImprovementHarness(
            backbone_name=args.backbone,
            min_sft_samples=args.min_sft_samples,
        )
        log.info("Self-improvement harness enabled (min_sft_samples=%d)", args.min_sft_samples)

    expert_names = policy.available_experts if policy else []
    log.info("Mode: %s", "Claude + experts" if expert_names else "Claude-only")
    if expert_names:
        log.info("Available experts: %s", expert_names)

    asyncio.run(run_benchmark(
        version=args.version,
        max_steps=args.max_steps,
        model=args.model,
        headed=args.headed,
        policy=policy,
        verbose=args.verbose,
        improvement_harness=harness,
        min_training_pairs=args.min_training_pairs,
        min_occurrences=args.min_occurrences,
        min_sft_samples=args.min_sft_samples,
        router_first=args.router_first,
    ))


if __name__ == "__main__":
    main()

"""Top-level orchestrator tying together collection, training, and deployment.

SelfImprovementHarness is the single entry point for:
- Collecting teacher trajectories (Claude solves, system labels and stores)
- Training experts (SFT autoresearch -> RL autoresearch)
- Deploying trained experts to the MoE policy
- Status reporting
"""

from __future__ import annotations

import base64
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from hexis.data.schemas import (
    CheckpointRecord,
    DataRequest,
    DataSource,
    ExpertStatus,
    TrajectoryRecord,
)
from hexis.data.registry import ExpertRegistry
from hexis.data.trajectory_store import TrajectoryStore
from hexis.self_improve.improvement_loop import ImprovementLoop
from hexis.self_improve.labeler import label_trajectory_entry

log = logging.getLogger(__name__)


class SelfImprovementHarness:
    """Top-level orchestrator for the self-improvement pipeline."""

    def __init__(
        self,
        registry: ExpertRegistry | None = None,
        store: TrajectoryStore | None = None,
        backbone_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        min_sft_samples: int = 20,
    ):
        self.registry = registry or ExpertRegistry()
        self.store = store or TrajectoryStore()
        self.backbone_name = backbone_name
        self._loop = ImprovementLoop(
            self.registry, self.store, backbone_name,
            min_sft_samples=min_sft_samples,
        )

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def store_trajectory_entry(self, entry: dict) -> bool:
        """Label and store a single trajectory entry. Returns True if stored."""
        screenshot_b64 = ""

        before = entry.get("screenshot_before", b"")
        if isinstance(before, bytes) and before:
            screenshot_b64 = base64.b64encode(before).decode("ascii")
        elif isinstance(before, str) and before:
            screenshot_b64 = before

        if not screenshot_b64:
            return False

        label = label_trajectory_entry(entry)
        record = TrajectoryRecord(
            screenshot_b64=screenshot_b64,
            action=entry.get("action", {}),
            expert_label=label,
            metadata={
                "step": entry.get("step", 0),
                "url": entry.get("url", ""),
                "tool": entry.get("tool", ""),
                "ssim": entry.get("ssim", 0.0),
                "reasoning_len": len(entry.get("reasoning", "")),
            },
        )
        return self.store.add(record)

    def collect_none_examples(
        self,
        trajectory_dir: str | Path,
        sample_rate: float = 0.3,
    ) -> int:
        """Sample non-expert screenshots from raw trajectories as 'none' class."""
        import json

        traj_dir = Path(trajectory_dir)
        actions_file = traj_dir / "actions.jsonl"
        shots_dir = traj_dir / "screenshots"

        if not actions_file.exists():
            log.warning("No actions.jsonl in %s", traj_dir)
            return 0

        added = 0
        for line in actions_file.read_text().splitlines():
            if not line.strip():
                continue
            if random.random() > sample_rate:
                continue

            entry = json.loads(line)
            tool = entry.get("tool", "")
            if tool in ("dismiss_popups", "solve_radio_modal"):
                continue

            screenshot_file = entry.get("screenshot_before", "")
            if not screenshot_file or not (shots_dir / screenshot_file).exists():
                continue

            jpeg_bytes = (shots_dir / screenshot_file).read_bytes()
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")

            record = TrajectoryRecord(
                screenshot_b64=b64,
                action=entry.get("action", {}),
                expert_label="none",
                metadata={"source": str(traj_dir), "tool": tool},
            )
            if self.store.add(record):
                added += 1

        log.info("Collected %d none examples from %s", added, traj_dir)
        return added

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_expert(
        self,
        expert_name: str,
    ) -> tuple[str | None, DataRequest | None]:
        """Run full SFT -> RL training for an expert."""
        ckpt, data_req = self._loop.run_full(expert_name)
        if ckpt is not None:
            self.registry.set_status(expert_name, ExpertStatus.VALIDATING)
            return ckpt.path, data_req
        return None, data_req

    def train_all(self) -> dict[str, Any]:
        """Train all experts that are in collecting/needs_data state with enough data."""
        results = {}
        for expert in self.registry.list_all():
            name = expert["name"]
            status = expert["status"]
            if status in (ExpertStatus.COLLECTING.value, ExpertStatus.NEEDS_DATA.value,
                          ExpertStatus.PROPOSED.value):
                n = self.store.count(name)
                if n >= 20:
                    log.info("Training '%s' (%d samples)", name, n)
                    path, req = self.train_expert(name)
                    results[name] = {"checkpoint": path, "data_request": req}
                else:
                    log.info("Skipping '%s': only %d samples", name, n)
                    results[name] = {"skipped": True, "samples": n}
        return results

    def train_router(
        self,
        output_dir: str | Path = "checkpoints/router",
        min_val_accuracy: float = 0.6,
    ) -> str | None:
        """Train the MoE router on all collected trajectory data.

        Returns checkpoint path if val accuracy >= min_val_accuracy, else None.
        """
        output_dir = Path(output_dir)

        # Need at least 1 expert label + none for the router to learn the boundary
        labels = [l for l in self.store.labels() if l != "none"]
        if len(labels) < 1:
            log.info("Router training skipped: no expert labels in trajectory store")
            return None
        if self.store.count("none") == 0:
            log.info("Router training skipped: no 'none' examples (needed for fallback class)")
            return None

        # Export router training data
        router_data = output_dir / "router_train.jsonl"
        self.store.export_for_router(router_data)

        # Build expert map from registry
        expert_map = {}
        for expert in self.registry.list_all():
            name = expert["name"]
            if self.store.count(name) > 0:
                expert_map[name] = expert["subgoal"]

        if len(expert_map) < 1:
            log.info("Router training skipped: no experts with data in registry")
            return None

        log.info("Training router with %d experts: %s", len(expert_map), list(expert_map.keys()))

        try:
            from hexis.model.backbone import VLMBackbone
            from hexis.model.router import MoERouter
            from hexis.training.router_sft import train_router

            backbone = VLMBackbone(model_name=self.backbone_name)
            router = MoERouter(backbone)
            for name, subgoal in expert_map.items():
                router.register_expert(name, subgoal)

            none_data = None
            if self.store.count("none") > 0:
                none_data = str(output_dir / "none_train.jsonl")
                # Export none data separately
                with open(none_data, "w") as f:
                    for rec in self.store.query("none"):
                        f.write(json.dumps({
                            "screenshot_b64": rec.screenshot_b64,
                            "task_text": "none",
                            "expert_name": "__none__",
                        }) + "\n")

            best_loss, val_acc = train_router(
                backbone=backbone,
                router=router,
                expert_map=expert_map,
                data_path=str(router_data),
                output_dir=str(output_dir),
                none_data_path=none_data,
                epochs=20,
            )
            if val_acc < min_val_accuracy:
                log.warning(
                    "Router val_acc=%.1f%% < %.0f%% threshold — not deploying. "
                    "Need more diverse training data.",
                    val_acc * 100, min_val_accuracy * 100,
                )
                return None
            self.registry.set_router_retrain_time(time.time())
            log.info("Router trained: val_acc=%.1f%%, checkpoint=%s", val_acc * 100, output_dir)
            return str(output_dir / "best")
        except Exception as e:
            log.error("Router training failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def deploy_expert(
        self,
        expert_name: str,
        checkpoint_path: str,
        policy: Any = None,
    ) -> None:
        """Mark expert as deployed and optionally hot-load into MoE policy."""
        self.registry.set_status(expert_name, ExpertStatus.DEPLOYED)
        self.registry.set_data_request(expert_name, None)

        if policy is not None:
            try:
                policy.load_expert(expert_name, checkpoint_path)
                log.info("Hot-loaded '%s' from %s", expert_name, checkpoint_path)
            except Exception as e:
                log.warning("Failed to hot-load '%s': %s", expert_name, e)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_existing_data(
        self,
        path: str | Path,
        expert_label: str,
        format: str = "jsonl",
    ) -> int:
        """Import training data from an existing file."""
        count = self.store.import_jsonl(path, expert_label)

        expert = self.registry.get(expert_label)
        if expert is None and expert_label != "none":
            self.registry.register(
                expert_label,
                subgoal=expert_label.replace("_", " "),
                action_types=["click"],
                status=ExpertStatus.COLLECTING,
            )
        if expert_label != "none":
            self.registry.set_trajectory_count(expert_label, self.store.count(expert_label))

        return count

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return full status of the self-improvement pipeline."""
        experts = {}
        for exp in self.registry.list_all():
            name = exp["name"]
            experts[name] = {
                "status": exp["status"],
                "subgoal": exp["subgoal"],
                "trajectory_count": self.store.count(name),
                "checkpoints": len(exp.get("checkpoints", [])),
                "best_sft": None,
                "best_rl": None,
                "data_request": exp.get("pending_data_request"),
            }
            best_sft = self.registry.best_checkpoint(name, "sft")
            if best_sft:
                experts[name]["best_sft"] = {
                    "val_loss": best_sft.val_loss,
                    "path": best_sft.path,
                }
            best_rl = self.registry.best_checkpoint(name, "rl")
            if best_rl:
                experts[name]["best_rl"] = {
                    "avg_reward": best_rl.avg_reward,
                    "path": best_rl.path,
                }

        return {
            "experts": experts,
            "trajectory_store": self.store.stats(),
            "global": self.registry.global_stats,
        }

    def print_status(self) -> None:
        """Print a human-readable status summary."""
        s = self.status()

        print("\n=== Self-Improvement Pipeline Status ===\n")

        if not s["experts"]:
            print("  No experts registered yet.")
        else:
            for name, info in s["experts"].items():
                status_badge = info["status"].upper()
                print(f"  [{status_badge}] {name}")
                print(f"    Subgoal: {info['subgoal']}")
                print(f"    Trajectories: {info['trajectory_count']}")
                print(f"    Checkpoints: {info['checkpoints']}")
                if info["best_sft"]:
                    print(f"    Best SFT: val_loss={info['best_sft']['val_loss']:.4f}")
                if info["best_rl"]:
                    print(f"    Best RL: avg_reward={info['best_rl']['avg_reward']:.2f}")
                if info["data_request"]:
                    req = info["data_request"]
                    print(f"    DATA NEEDED: {req['reason']} (min {req['min_additional']} more)")
                print()

        print("  Trajectory Store:")
        for label, count in s["trajectory_store"].items():
            print(f"    {label}: {count}")

        total = s["global"].get("total_trajectories", 0)
        print(f"\n  Total trajectories (registry): {total}")
        print()

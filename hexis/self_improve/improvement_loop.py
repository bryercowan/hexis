"""Autoresearch training loop: train in small rounds, validate, request more data if stuck.

Runs locally on MPS (M3 Ultra). No remote GPU orchestration needed.
SFT: ~5-10min per round with cached features. RL: ~30min per round with headless Playwright.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hexis.data.schemas import (
    CheckpointRecord,
    DataRequest,
    ExpertStatus,
)
from hexis.data.registry import ExpertRegistry
from hexis.data.trajectory_store import TrajectoryStore

log = logging.getLogger(__name__)


class ImprovementLoop:
    """Autoresearch pattern: train in small rounds, validate, request more data if stuck."""

    def __init__(
        self,
        registry: ExpertRegistry,
        trajectory_store: TrajectoryStore,
        backbone_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        sft_round_epochs: int = 5,
        sft_max_rounds: int = 20,
        sft_plateau_patience: int = 3,
        rl_round_episodes: int = 200,
        rl_max_rounds: int = 15,
        rl_plateau_patience: int = 3,
        min_sft_samples: int = 20,
        max_val_px_error: float = 50.0,
    ):
        self.registry = registry
        self.store = trajectory_store
        self.backbone_name = backbone_name

        self.sft_round_epochs = sft_round_epochs
        self.sft_max_rounds = sft_max_rounds
        self.sft_plateau_patience = sft_plateau_patience

        self.rl_round_episodes = rl_round_episodes
        self.rl_max_rounds = rl_max_rounds
        self.rl_plateau_patience = rl_plateau_patience
        self.min_sft_samples = min_sft_samples
        self.max_val_px_error = max_val_px_error

    def run_sft(
        self,
        expert_name: str,
    ) -> tuple[CheckpointRecord | None, DataRequest | None]:
        """Run SFT autoresearch: train in rounds, validate, stop or request data."""
        expert = self.registry.get(expert_name)
        if expert is None:
            log.error("Expert '%s' not registered", expert_name)
            return None, None

        n_train = self.store.count(expert_name)
        if n_train < self.min_sft_samples:
            return None, DataRequest(
                expert_name=expert_name,
                reason=f"Only {n_train} training samples, need at least {self.min_sft_samples}",
                min_additional=max(200 - n_train, self.min_sft_samples),
                priority=2,
            )

        self.registry.set_status(expert_name, ExpertStatus.TRAINING_SFT)
        subgoal = expert["subgoal"]

        export_dir = Path(f"/tmp/hexis_improve_{expert_name}")
        export_dir.mkdir(parents=True, exist_ok=True)
        train_path = export_dir / "train.jsonl"
        val_path = export_dir / "val.jsonl"

        train_records, val_records = self.store.train_val_split(expert_name, val_fraction=0.1)
        _write_records_jsonl(train_records, train_path, subgoal)
        _write_records_jsonl(val_records, val_path, subgoal)

        log.info("SFT autoresearch for '%s': %d train, %d val samples",
                 expert_name, len(train_records), len(val_records))

        output_base = Path(f"checkpoints/self_improve/{expert_name}/sft")
        output_base.mkdir(parents=True, exist_ok=True)

        feature_cache = str(export_dir / "features")

        best_val_loss = float("inf")
        best_checkpoint_path: str | None = None
        plateau_count = 0
        overfit_count = 0
        prev_train_loss = float("inf")
        checkpoint_version = len(expert.get("checkpoints", [])) + 1

        for round_idx in range(self.sft_max_rounds):
            round_dir = output_base / f"round_{round_idx}"
            resume_arg = best_checkpoint_path if best_checkpoint_path else None

            result = _run_sft_round(
                data_path=str(train_path),
                val_path=str(val_path),
                expert_name=expert_name,
                subgoal=subgoal,
                backbone=self.backbone_name,
                output_dir=str(round_dir),
                epochs=self.sft_round_epochs,
                resume=resume_arg,
                feature_cache=feature_cache,
            )

            if result is None:
                log.error("SFT round %d failed", round_idx + 1)
                break

            train_loss = result.get("train_loss", float("inf"))
            val_loss = result.get("val_loss", float("inf"))
            val_px_error = result.get("val_px_error", float("inf"))
            checkpoint_path = result.get("checkpoint_path", "")

            log.info("  Round %d: train_loss=%.4f val_loss=%.4f val_px_err=%.1f",
                     round_idx + 1, train_loss, val_loss, val_px_error)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_path
                plateau_count = 0
                overfit_count = 0

                record = CheckpointRecord(
                    version=checkpoint_version,
                    phase="sft",
                    path=checkpoint_path,
                    epoch=(round_idx + 1) * self.sft_round_epochs,
                    val_loss=val_loss,
                    val_px_error=val_px_error,
                    timestamp=time.time(),
                )
                self.registry.add_checkpoint(expert_name, record)
                checkpoint_version += 1
            else:
                plateau_count += 1

            if train_loss < prev_train_loss and val_loss > best_val_loss * 1.05:
                overfit_count += 1
                if overfit_count >= 2:
                    log.info("Overfitting detected, stopping SFT")
                    break
            else:
                overfit_count = 0

            prev_train_loss = train_loss

            if plateau_count >= self.sft_plateau_patience:
                log.info("SFT plateau for %d rounds, requesting more data", plateau_count)
                data_request = DataRequest(
                    expert_name=expert_name,
                    reason=f"SFT plateau after {round_idx + 1} rounds, "
                           f"best_val_loss={best_val_loss:.4f}",
                    min_additional=200,
                    priority=1,
                )
                self.registry.set_status(expert_name, ExpertStatus.NEEDS_DATA)
                self.registry.set_data_request(expert_name, data_request)
                if best_checkpoint_path:
                    return CheckpointRecord.from_dict({
                        "version": checkpoint_version - 1, "phase": "sft",
                        "path": best_checkpoint_path, "epoch": 0,
                        "val_loss": best_val_loss, "timestamp": time.time(),
                    }), data_request
                return None, data_request

        if best_checkpoint_path:
            log.info("SFT complete: best_val_loss=%.4f at %s", best_val_loss, best_checkpoint_path)
            return CheckpointRecord.from_dict({
                "version": checkpoint_version - 1, "phase": "sft",
                "path": best_checkpoint_path, "epoch": 0,
                "val_loss": best_val_loss, "timestamp": time.time(),
            }), None
        return None, None

    def run_rl(
        self,
        expert_name: str,
        sft_checkpoint: str | None = None,
    ) -> tuple[CheckpointRecord | None, DataRequest | None]:
        """Run RL autoresearch: REINFORCE on live benchmark."""
        expert = self.registry.get(expert_name)
        if expert is None:
            log.error("Expert '%s' not registered", expert_name)
            return None, None

        if sft_checkpoint is None:
            best_sft = self.registry.best_checkpoint(expert_name, phase="sft")
            if best_sft is None:
                log.error("No SFT checkpoint for '%s', run SFT first", expert_name)
                return None, None
            sft_checkpoint = best_sft.path

        self.registry.set_status(expert_name, ExpertStatus.TRAINING_RL)
        subgoal = expert["subgoal"]
        output_base = Path(f"checkpoints/self_improve/{expert_name}/rl")
        output_base.mkdir(parents=True, exist_ok=True)

        best_reward = -float("inf")
        best_checkpoint_path: str | None = None
        plateau_count = 0
        checkpoint_version = len(expert.get("checkpoints", [])) + 1

        for round_idx in range(self.rl_max_rounds):
            round_dir = output_base / f"round_{round_idx}"
            version = (round_idx % 3) + 1

            result = _run_rl_round(
                checkpoint=sft_checkpoint if round_idx == 0 and best_checkpoint_path is None
                           else (best_checkpoint_path or sft_checkpoint),
                subgoal=subgoal,
                episodes=self.rl_round_episodes,
                output_dir=str(round_dir),
                version=version,
            )

            if result is None:
                log.error("RL round %d failed", round_idx + 1)
                break

            avg_reward = result.get("avg_reward", -float("inf"))
            checkpoint_path = result.get("checkpoint_path", "")

            log.info("  Round %d: avg_reward=%.2f", round_idx + 1, avg_reward)

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_checkpoint_path = checkpoint_path
                plateau_count = 0

                record = CheckpointRecord(
                    version=checkpoint_version, phase="rl",
                    path=checkpoint_path,
                    epoch=(round_idx + 1) * self.rl_round_episodes,
                    avg_reward=avg_reward, timestamp=time.time(),
                )
                self.registry.add_checkpoint(expert_name, record)
                checkpoint_version += 1
            else:
                plateau_count += 1

            if avg_reward < best_reward * 0.8 and best_checkpoint_path:
                log.info("RL degrading, reverting")
                break

            if plateau_count >= self.rl_plateau_patience:
                log.info("RL plateau for %d rounds, requesting more data", plateau_count)
                data_request = DataRequest(
                    expert_name=expert_name,
                    reason=f"RL plateau after {round_idx + 1} rounds, best_reward={best_reward:.2f}",
                    min_additional=100, priority=1,
                )
                self.registry.set_status(expert_name, ExpertStatus.NEEDS_DATA)
                self.registry.set_data_request(expert_name, data_request)
                if best_checkpoint_path:
                    return CheckpointRecord.from_dict({
                        "version": checkpoint_version - 1, "phase": "rl",
                        "path": best_checkpoint_path, "epoch": 0,
                        "avg_reward": best_reward, "timestamp": time.time(),
                    }), data_request
                return None, data_request

        if best_checkpoint_path:
            return CheckpointRecord.from_dict({
                "version": checkpoint_version - 1, "phase": "rl",
                "path": best_checkpoint_path, "epoch": 0,
                "avg_reward": best_reward, "timestamp": time.time(),
            }), None
        return None, None

    def run_full(
        self,
        expert_name: str,
    ) -> tuple[CheckpointRecord | None, DataRequest | None]:
        """Run full SFT -> behavioral validation -> RL pipeline.

        After SFT, checks val_px_error against threshold. If the expert
        can't reproduce the teacher's clicks within max_val_px_error pixels,
        it requests more data instead of proceeding to RL.
        """
        sft_ckpt, sft_request = self.run_sft(expert_name)

        if sft_request is not None and sft_ckpt is None:
            return None, sft_request

        if sft_ckpt is None:
            log.error("SFT produced no checkpoint for '%s'", expert_name)
            return None, None

        # --- Behavioral validation gate ---
        val_px = sft_ckpt.val_px_error
        if val_px > self.max_val_px_error:
            log.warning(
                "BEHAVIORAL CHECK FAILED: '%s' val_px_error=%.1f > threshold=%.1f. "
                "Expert can't reproduce teacher's clicks. Requesting more data.",
                expert_name, val_px, self.max_val_px_error,
            )
            data_request = DataRequest(
                expert_name=expert_name,
                reason=f"Behavioral validation failed: val_px_error={val_px:.1f}px "
                       f"(threshold={self.max_val_px_error:.0f}px). "
                       f"Expert can't reproduce teacher clicks accurately enough.",
                min_additional=100,
                priority=1,
            )
            self.registry.set_status(expert_name, ExpertStatus.NEEDS_DATA)
            self.registry.set_data_request(expert_name, data_request)
            return sft_ckpt, data_request

        log.info(
            "BEHAVIORAL CHECK PASSED: '%s' val_px_error=%.1fpx (<= %.0fpx). Starting RL.",
            expert_name, val_px, self.max_val_px_error,
        )
        return self.run_rl(expert_name, sft_checkpoint=sft_ckpt.path)


# ---------------------------------------------------------------------------
# Subprocess wrappers — invoke training scripts
# ---------------------------------------------------------------------------

def _write_records_jsonl(records: list, path: Path, conditioning_text: str) -> None:
    with open(path, "w") as f:
        for rec in records:
            entry = {
                "screenshot_b64": rec.screenshot_b64,
                "action": rec.action,
                "conditioning_text": conditioning_text,
            }
            f.write(json.dumps(entry) + "\n")


def _run_sft_round(
    data_path: str, val_path: str, expert_name: str, subgoal: str,
    backbone: str, output_dir: str, epochs: int,
    resume: str | None = None, feature_cache: str | None = None,
) -> dict[str, Any] | None:
    cmd = [
        sys.executable, "-m", "hexis.training.sft_cli",
        "--data", data_path, "--val-data", val_path,
        "--expert", expert_name, "--subgoal", subgoal,
        "--backbone", backbone, "--output", output_dir,
        "--epochs", str(epochs), "--json-output",
    ]
    if resume:
        cmd.extend(["--resume", resume])
    if feature_cache:
        cmd.extend(["--feature-cache", feature_cache])

    log.info("Running SFT: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        output_lines: list[str] = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            if " INFO " in line or " WARNING " in line or " ERROR " in line:
                log.info("  [SFT] %s", line.split(" INFO ", 1)[-1] if " INFO " in line else line)
        proc.wait(timeout=3600)

        if proc.returncode != 0:
            log.error("SFT failed (rc=%d)", proc.returncode)
            return None

        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("SFT timed out after 1 hour")
        return None


def _run_rl_round(
    checkpoint: str, subgoal: str, episodes: int,
    output_dir: str, version: int = 1,
) -> dict[str, Any] | None:
    cmd = [
        sys.executable, "-m", "hexis.training.rl_cli",
        "--checkpoint", checkpoint,
        "--subgoal", subgoal,
        "--episodes", str(episodes),
        "--output", output_dir,
    ]

    log.info("Running RL: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        output_lines: list[str] = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            if " INFO " in line or " WARNING " in line or " ERROR " in line:
                log.info("  [RL] %s", line.split(" INFO ", 1)[-1] if " INFO " in line else line)
        proc.wait(timeout=7200)

        if proc.returncode != 0:
            log.error("RL failed (rc=%d)", proc.returncode)
            return None

        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        import re
        avg_reward = None
        rl_checkpoint_path = None
        for line in reversed(output_lines):
            if avg_reward is None:
                m = re.search(r"avg_reward[=:]?\s*([-\d.]+)", line)
                if m:
                    avg_reward = float(m.group(1))
            if rl_checkpoint_path is None and "Saved" in line and output_dir in line:
                m = re.search(r"Saved.*?(\S+/(?:best|latest|final)\S*)", line)
                if m:
                    rl_checkpoint_path = m.group(1)

        if avg_reward is not None:
            return {
                "avg_reward": avg_reward,
                "checkpoint_path": rl_checkpoint_path or str(Path(output_dir) / "best"),
            }
        return None
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("RL timed out after 2 hours")
        return None

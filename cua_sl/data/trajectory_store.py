"""JSONL-per-label trajectory storage for expert training data.

Stores trajectories at ~/.cua-sl/trajectories/ with one JSONL file per expert label
plus a `none.jsonl` for negative examples. Deduplicates via SHA-256 of screenshot bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from cua_sl.data.schemas import TrajectoryRecord

log = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path.home() / ".cua-sl" / "trajectories"


class TrajectoryStore:
    """JSONL-per-label trajectory storage with deduplication."""

    def __init__(self, base_dir: str | Path = DEFAULT_STORE_PATH):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._hash_path = self._base / "_hashes.json"
        self._hashes: set[str] = set()
        self._load_hashes()

    def _load_hashes(self) -> None:
        if self._hash_path.exists():
            try:
                data = json.loads(self._hash_path.read_text())
                self._hashes = set(data)
            except (json.JSONDecodeError, TypeError):
                self._hashes = set()

    def _save_hashes(self) -> None:
        self._hash_path.write_text(json.dumps(list(self._hashes)))

    def _label_path(self, label: str) -> Path:
        safe_label = label.replace("/", "_").replace("\\", "_")
        return self._base / f"{safe_label}.jsonl"

    @staticmethod
    def _compute_hash(screenshot_b64: str) -> str:
        raw = base64.b64decode(screenshot_b64)
        return hashlib.sha256(raw).hexdigest()

    def add(self, record: TrajectoryRecord, force: bool = False) -> bool:
        """Add a single trajectory. Returns False if duplicate (unless force=True).

        When force=True, the record is stored even if the screenshot hash
        already exists under a different label. Use this for relabeling
        (e.g., pattern detector overriding heuristic labels).
        """
        if not record.screenshot_hash:
            record.screenshot_hash = self._compute_hash(record.screenshot_b64)

        if not force and record.screenshot_hash in self._hashes:
            return False

        self._hashes.add(record.screenshot_hash)
        path = self._label_path(record.expert_label)
        with open(path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        self._save_hashes()
        return True

    def add_batch(self, records: list[TrajectoryRecord]) -> int:
        """Add multiple trajectories. Returns count of non-duplicates added."""
        added = 0
        label_buffers: dict[str, list[str]] = {}

        for record in records:
            if not record.screenshot_hash:
                record.screenshot_hash = self._compute_hash(record.screenshot_b64)
            if record.screenshot_hash in self._hashes:
                continue
            self._hashes.add(record.screenshot_hash)
            label = record.expert_label
            if label not in label_buffers:
                label_buffers[label] = []
            label_buffers[label].append(json.dumps(record.to_dict()))
            added += 1

        for label, lines in label_buffers.items():
            path = self._label_path(label)
            with open(path, "a") as f:
                f.write("\n".join(lines) + "\n")

        if added > 0:
            self._save_hashes()
        return added

    def import_jsonl(
        self,
        path: str | Path,
        expert_label: str,
        action_key: str = "action",
        screenshot_key: str = "screenshot_b64",
    ) -> int:
        """Import records from an existing JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No file at {path}")

        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            b64 = d.get(screenshot_key, "")
            if not b64:
                continue

            action = d.get(action_key, {})
            if isinstance(action, str):
                try:
                    action = json.loads(action)
                except json.JSONDecodeError:
                    action = {"raw": action}

            records.append(TrajectoryRecord(
                screenshot_b64=b64,
                action=action,
                expert_label=expert_label,
                metadata={k: v for k, v in d.items()
                          if k not in (screenshot_key, action_key)},
            ))

        added = self.add_batch(records)
        log.info("Imported %d/%d records from %s (label=%s)",
                 added, len(records), path, expert_label)
        return added

    def query(self, expert_label: str) -> list[TrajectoryRecord]:
        """Load all trajectories for a label."""
        path = self._label_path(expert_label)
        if not path.exists():
            return []

        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TrajectoryRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
        return records

    def count(self, expert_label: str | None = None) -> int:
        """Count trajectories, optionally filtered by label."""
        if expert_label is not None:
            path = self._label_path(expert_label)
            if not path.exists():
                return 0
            return sum(1 for line in path.read_text().splitlines() if line.strip())

        total = 0
        for p in self._base.glob("*.jsonl"):
            if p.name == "_hashes.json":
                continue
            total += sum(1 for line in p.read_text().splitlines() if line.strip())
        return total

    def labels(self) -> list[str]:
        """List all expert labels that have data."""
        result = []
        for p in sorted(self._base.glob("*.jsonl")):
            label = p.stem
            if label.startswith("_"):
                continue
            result.append(label)
        return result

    def train_val_split(
        self,
        expert_label: str,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
        """Deterministic train/val split for a label."""
        import random

        records = self.query(expert_label)
        if not records:
            return [], []

        rng = random.Random(seed)
        shuffled = list(records)
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_fraction))
        return shuffled[n_val:], shuffled[:n_val]

    def export_for_sft(
        self,
        expert_label: str,
        output_path: str | Path,
        conditioning_text: str,
    ) -> str:
        """Export trajectories as JSONL suitable for SFT training."""
        records = self.query(expert_label)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for rec in records:
                entry = {
                    "screenshot_b64": rec.screenshot_b64,
                    "action": rec.action,
                    "conditioning_text": conditioning_text,
                }
                f.write(json.dumps(entry) + "\n")

        log.info("Exported %d records to %s", len(records), output_path)
        return str(output_path)

    def export_for_router(self, output_path: str | Path) -> str:
        """Export all labels (including 'none') as JSONL for router training."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = 0
        with open(output_path, "w") as f:
            for label in self.labels():
                records = self.query(label)
                for rec in records:
                    entry = {
                        "screenshot_b64": rec.screenshot_b64,
                        "task_text": rec.metadata.get("subgoal", label),
                        "expert_name": label if label != "none" else "__none__",
                    }
                    f.write(json.dumps(entry) + "\n")
                    total += 1

        log.info("Exported %d records (%d labels) to %s",
                 total, len(self.labels()), output_path)
        return str(output_path)

    def stats(self) -> dict[str, int]:
        """Return {label: count} for all labels."""
        result = {}
        for label in self.labels():
            result[label] = self.count(label)
        return result

"""Persistent expert registry tracking lifecycle, checkpoints, and data sources.

Stored as JSON at ~/.cua-sl/expert_registry.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from hexis.data.schemas import (
    CheckpointRecord,
    DataRequest,
    DataSource,
    ExpertStatus,
)

log = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path.home() / ".cua-sl" / "expert_registry.json"


class ExpertRegistry:
    """Persistent registry of all experts and their lifecycle state."""

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH):
        self._path = Path(path)
        self._data: dict[str, Any] = {"version": 1, "experts": {}, "global_stats": {
            "total_trajectories": 0,
            "last_router_retrain": 0.0,
        }}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
                log.debug("Loaded registry from %s (%d experts)",
                          self._path, len(self._data.get("experts", {})))
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("Failed to load registry: %s — starting fresh", e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def register(
        self,
        name: str,
        subgoal: str,
        action_types: list[str],
        status: ExpertStatus = ExpertStatus.PROPOSED,
    ) -> None:
        """Register a new expert or update an existing one."""
        if name in self._data["experts"]:
            log.info("Expert '%s' already registered, updating subgoal/action_types", name)
            self._data["experts"][name]["subgoal"] = subgoal
            self._data["experts"][name]["action_types"] = action_types
        else:
            self._data["experts"][name] = {
                "name": name,
                "subgoal": subgoal,
                "status": status.value,
                "action_types": action_types,
                "data_sources": [],
                "checkpoints": [],
                "trajectory_count": 0,
                "pending_data_request": None,
                "created_at": time.time(),
                "validation_history": [],
                "retry_count": 0,
            }
            log.info("Registered new expert: '%s' (status=%s)", name, status.value)
        self._save()

    def get(self, name: str) -> dict | None:
        return self._data["experts"].get(name)

    def list_all(self) -> list[dict]:
        return list(self._data["experts"].values())

    def set_status(self, name: str, status: ExpertStatus) -> None:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        old = expert["status"]
        expert["status"] = status.value
        log.info("Expert '%s': %s -> %s", name, old, status.value)
        self._save()

    def add_checkpoint(self, name: str, checkpoint: CheckpointRecord) -> None:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        expert["checkpoints"].append(checkpoint.to_dict())
        self._save()

    def add_data_source(self, name: str, source: DataSource) -> None:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        expert["data_sources"].append(source.to_dict())
        expert["trajectory_count"] += source.num_trajectories
        self._data["global_stats"]["total_trajectories"] += source.num_trajectories
        self._save()

    def set_trajectory_count(self, name: str, count: int) -> None:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        expert["trajectory_count"] = count
        self._save()

    def set_data_request(self, name: str, request: DataRequest | None) -> None:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        expert["pending_data_request"] = request.to_dict() if request else None
        self._save()

    def latest_checkpoint(self, name: str, phase: str = "rl") -> CheckpointRecord | None:
        expert = self._data["experts"].get(name)
        if expert is None:
            return None
        ckpts = [c for c in expert.get("checkpoints", []) if c["phase"] == phase]
        if not ckpts:
            return None
        latest = max(ckpts, key=lambda c: c.get("timestamp", 0))
        return CheckpointRecord.from_dict(latest)

    def best_checkpoint(self, name: str, phase: str = "rl") -> CheckpointRecord | None:
        expert = self._data["experts"].get(name)
        if expert is None:
            return None
        ckpts = [c for c in expert.get("checkpoints", []) if c["phase"] == phase]
        if not ckpts:
            return None
        if phase == "rl":
            best = max(ckpts, key=lambda c: c.get("avg_reward", 0))
        else:
            best = min(ckpts, key=lambda c: c.get("val_loss", float("inf")))
        return CheckpointRecord.from_dict(best)

    def deployed_experts(self) -> dict[str, str]:
        """Return {name: checkpoint_path} for all deployed experts."""
        result = {}
        for name, expert in self._data["experts"].items():
            if expert["status"] == ExpertStatus.DEPLOYED.value:
                ckpts = expert.get("checkpoints", [])
                if ckpts:
                    rl = [c for c in ckpts if c["phase"] == "rl"]
                    sft = [c for c in ckpts if c["phase"] == "sft"]
                    best = rl[-1] if rl else (sft[-1] if sft else None)
                    if best:
                        result[name] = best["path"]
        return result

    def experts_needing_data(self) -> list[tuple[str, DataRequest]]:
        result = []
        for name, expert in self._data["experts"].items():
            req = expert.get("pending_data_request")
            if req is not None:
                result.append((name, DataRequest.from_dict(req)))
        return result

    def set_router_retrain_time(self, timestamp: float) -> None:
        self._data["global_stats"]["last_router_retrain"] = timestamp
        self._save()

    @property
    def global_stats(self) -> dict:
        return self._data.get("global_stats", {})

    def record_validation(self, name: str, success_rate: float, checkpoint: str) -> None:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        if "validation_history" not in expert:
            expert["validation_history"] = []
        expert["validation_history"].append({
            "success_rate": success_rate,
            "checkpoint": checkpoint,
            "timestamp": time.time(),
        })
        log.info("Expert '%s': validation %.0f%% (checkpoint=%s)",
                 name, success_rate * 100, checkpoint)
        self._save()

    def increment_retry(self, name: str) -> int:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        expert["retry_count"] = expert.get("retry_count", 0) + 1
        self._save()
        return expert["retry_count"]

    def reset_retry(self, name: str) -> None:
        expert = self._data["experts"].get(name)
        if expert is None:
            raise KeyError(f"Expert '{name}' not registered")
        expert["retry_count"] = 0
        self._save()

    @property
    def expert_names(self) -> list[str]:
        return list(self._data["experts"].keys())

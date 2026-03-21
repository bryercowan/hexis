"""Shared types for the self-improvement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExpertStatus(Enum):
    """Expert lifecycle state machine.

    [proposed] -> [collecting] -> [training_sft] -> [training_rl]
        -> [validating] -> [deployed]
              ^               |                |           |
              +-- [needs_data] <-- plateau ----+-----------+
    """
    PROPOSED = "proposed"
    COLLECTING = "collecting"
    TRAINING_SFT = "training_sft"
    TRAINING_RL = "training_rl"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    NEEDS_DATA = "needs_data"


@dataclass
class CheckpointRecord:
    """A single training checkpoint for an expert."""
    version: int
    phase: str  # "sft" or "rl"
    path: str
    epoch: int
    val_loss: float = 0.0
    val_px_error: float = 0.0
    avg_reward: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "phase": self.phase,
            "path": self.path,
            "epoch": self.epoch,
            "val_loss": self.val_loss,
            "val_px_error": self.val_px_error,
            "avg_reward": self.avg_reward,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointRecord:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class DataSource:
    """Metadata about a batch of collected trajectories."""
    benchmark_version: int
    step_range: list[int]  # [start, end]
    num_trajectories: int
    collected_at: float = 0.0
    run_id: str = ""

    def to_dict(self) -> dict:
        return {
            "benchmark_version": self.benchmark_version,
            "step_range": self.step_range,
            "num_trajectories": self.num_trajectories,
            "collected_at": self.collected_at,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataSource:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class DataRequest:
    """A request for more training data, emitted when training plateaus."""
    expert_name: str
    reason: str
    suggested_sources: list[str] = field(default_factory=list)
    min_additional: int = 200
    priority: int = 1

    def to_dict(self) -> dict:
        return {
            "expert_name": self.expert_name,
            "reason": self.reason,
            "suggested_sources": self.suggested_sources,
            "min_additional": self.min_additional,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataRequest:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class TrajectoryRecord:
    """A single (screenshot, action) pair for SFT training."""
    screenshot_b64: str
    action: dict[str, Any]
    expert_label: str  # e.g. "dismiss_popups", "none"
    screenshot_hash: str = ""  # SHA-256 for dedup
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "screenshot_b64": self.screenshot_b64,
            "action": self.action,
            "expert_label": self.expert_label,
            "screenshot_hash": self.screenshot_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryRecord:
        return cls(
            screenshot_b64=d["screenshot_b64"],
            action=d.get("action", {}),
            expert_label=d.get("expert_label", "none"),
            screenshot_hash=d.get("screenshot_hash", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ValidationResult:
    """Metrics from evaluating an expert on a held-out set."""
    loss: float
    accuracy: float
    px_error: float
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "px_error": self.px_error,
            "per_class_accuracy": self.per_class_accuracy,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ValidationResult:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

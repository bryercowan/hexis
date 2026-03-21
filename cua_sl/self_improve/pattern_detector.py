"""Real-time pattern detector for the self-improvement loop.

Runs after each step completion, analyzes the trajectory buffer,
and proposes new expert skills when repetitive motor patterns are detected.

Generic — works for any GUI, not just the benchmark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill proposal — output of pattern detection
# ---------------------------------------------------------------------------

@dataclass
class SkillProposal:
    """A proposed expert skill extracted from trajectory patterns."""

    name: str
    subgoal: str
    action_types: list[str]
    training_pairs: list[dict] = field(default_factory=list)
    occurrences: int = 0
    steps_seen: list[int] = field(default_factory=list)
    avg_reasoning_len: float = 0.0
    estimated_savings: int = 0
    min_pairs_for_training: int = 200
    collecting: bool = False

    @property
    def is_motor(self) -> bool:
        """Motor patterns have short, repetitive reasoning."""
        return self.avg_reasoning_len < 250

    @property
    def ready_for_training(self) -> bool:
        return len(self.training_pairs) >= self.min_pairs_for_training

    @property
    def collection_progress(self) -> float:
        return min(1.0, len(self.training_pairs) / self.min_pairs_for_training)

    def to_sft_dataset(self) -> list[dict]:
        """Convert to SFT training format."""
        return [
            {
                "screenshot_before": p["screenshot_before"],
                "action": p["action"],
                "subgoal": self.subgoal,
            }
            for p in self.training_pairs
        ]


# ---------------------------------------------------------------------------
# Pattern signatures
# ---------------------------------------------------------------------------

def _action_signature(action: dict) -> str:
    """Reduce an action to a canonical type string."""
    tool = action.get("tool", "")
    if tool and tool != "computer":
        return f"cnn:{tool}"

    a = action.get("action", {})
    act_type = a.get("action", "unknown")
    if act_type in ("left_click", "double_click", "right_click"):
        return "click"
    if act_type == "scroll":
        direction = a.get("scroll_direction", "down")
        return f"scroll_{direction}"
    if act_type == "type":
        return "type"
    if act_type == "key":
        return "key"
    return act_type


def _click_region(action: dict, grid_size: int = 200) -> tuple[int, int] | None:
    """Quantize click coordinates to a grid cell."""
    a = action.get("action", {})
    coord = a.get("coordinate", [])
    if len(coord) == 2 and a.get("action") in ("left_click", "double_click"):
        return (int(coord[0]) // grid_size, int(coord[1]) // grid_size)
    return None


# ---------------------------------------------------------------------------
# Pattern detector
# ---------------------------------------------------------------------------

class PatternDetector:
    """Detects repetitive motor patterns across benchmark steps.

    After each step, call `analyze_step()` with the actions from that step.
    When a pattern has been seen `min_occurrences` times, a SkillProposal
    is returned and enters collection mode.
    """

    def __init__(
        self,
        min_occurrences: int = 3,
        min_training_pairs: int = 200,
    ) -> None:
        self.min_occurrences = min_occurrences
        self.min_training_pairs = min_training_pairs
        self._pattern_counts: dict[str, dict[str, Any]] = {}
        self._all_proposals: list[SkillProposal] = []
        self._proposed_signatures: set[str] = set()
        self._collecting: dict[str, SkillProposal] = {}

    @property
    def proposals(self) -> list[SkillProposal]:
        return list(self._all_proposals)

    @property
    def collecting_skills(self) -> dict[str, SkillProposal]:
        return dict(self._collecting)

    def skills_ready_for_training(self) -> list[SkillProposal]:
        """Return proposals that have enough data to start training."""
        ready = []
        for name, proposal in list(self._collecting.items()):
            if proposal.ready_for_training:
                proposal.collecting = False
                del self._collecting[name]
                ready.append(proposal)
                log.info(
                    "COLLECTION COMPLETE: '%s' — %d pairs ready for SFT",
                    name, len(proposal.training_pairs),
                )
        return ready

    def analyze_step(self, step_num: int, step_actions: list[dict]) -> list[SkillProposal]:
        """Analyze actions from a completed step. Returns new proposals (if any)."""
        if len(step_actions) < 2:
            return []

        self._collect_for_existing(step_num, step_actions)

        new_proposals = []

        scroll_click = self._detect_scroll_click_pattern(step_num, step_actions)
        if scroll_click:
            new_proposals.append(scroll_click)

        click_cluster = self._detect_click_cluster_pattern(step_num, step_actions)
        if click_cluster:
            new_proposals.append(click_cluster)

        type_seq = self._detect_type_sequence_pattern(step_num, step_actions)
        if type_seq:
            new_proposals.append(type_seq)

        return new_proposals

    # ------------------------------------------------------------------
    # Pattern: scroll + click (modal interaction)
    # ------------------------------------------------------------------

    def _detect_scroll_click_pattern(
        self, step_num: int, actions: list[dict],
    ) -> SkillProposal | None:
        sig = "scroll_click_select"
        if sig in self._proposed_signatures:
            return None

        scroll_runs = []
        current_scrolls = []
        for a in actions:
            s = _action_signature(a)
            if s.startswith("scroll_"):
                current_scrolls.append(a)
            else:
                if len(current_scrolls) >= 2 and s == "click":
                    scroll_runs.append({
                        "scrolls": list(current_scrolls),
                        "click": a,
                        "total_actions": len(current_scrolls) + 1,
                    })
                current_scrolls = []

        if not scroll_runs:
            return None

        if sig not in self._pattern_counts:
            self._pattern_counts[sig] = {
                "occurrences": 0, "steps": [], "training_pairs": [],
                "reasoning_lens": [], "total_actions": 0,
            }

        pc = self._pattern_counts[sig]
        pc["occurrences"] += len(scroll_runs)
        pc["steps"].append(step_num)

        for run in scroll_runs:
            for scroll_action in run["scrolls"]:
                pc["training_pairs"].append(scroll_action)
                pc["reasoning_lens"].append(len(scroll_action.get("reasoning", "")))
            pc["training_pairs"].append(run["click"])
            pc["reasoning_lens"].append(len(run["click"].get("reasoning", "")))
            pc["total_actions"] += run["total_actions"]

        if pc["occurrences"] >= self.min_occurrences:
            avg_reasoning = (
                sum(pc["reasoning_lens"]) / len(pc["reasoning_lens"])
                if pc["reasoning_lens"] else 0
            )
            proposal = SkillProposal(
                name="scroll_click_select",
                subgoal="scroll to find correct option and click it",
                action_types=["click", "scroll_up", "scroll_down"],
                training_pairs=pc["training_pairs"],
                occurrences=pc["occurrences"],
                steps_seen=pc["steps"],
                avg_reasoning_len=avg_reasoning,
                estimated_savings=pc["total_actions"] // pc["occurrences"],
                min_pairs_for_training=self.min_training_pairs,
                collecting=True,
            )
            self._proposed_signatures.add(sig)
            self._all_proposals.append(proposal)
            self._collecting[sig] = proposal
            return proposal

        return None

    # ------------------------------------------------------------------
    # Pattern: click cluster (dismiss-like)
    # ------------------------------------------------------------------

    def _detect_click_cluster_pattern(
        self, step_num: int, actions: list[dict],
    ) -> SkillProposal | None:
        sig = "click_cluster_dismiss"
        if sig in self._proposed_signatures:
            return None

        effective_clicks = []
        for a in actions:
            if (a.get("tool") == "computer"
                    and a.get("action", {}).get("action") == "left_click"
                    and a.get("ssim", 1.0) < 0.95):
                effective_clicks.append(a)

        if len(effective_clicks) < 2:
            return None

        regions = [_click_region(a) for a in effective_clicks]
        regions = [r for r in regions if r is not None]
        if not regions:
            return None

        if sig not in self._pattern_counts:
            self._pattern_counts[sig] = {
                "occurrences": 0, "steps": [], "training_pairs": [],
                "reasoning_lens": [],
            }

        pc = self._pattern_counts[sig]
        pc["occurrences"] += 1
        pc["steps"].append(step_num)
        for a in effective_clicks:
            pc["training_pairs"].append(a)
            pc["reasoning_lens"].append(len(a.get("reasoning", "")))

        if pc["occurrences"] >= self.min_occurrences:
            avg_reasoning = (
                sum(pc["reasoning_lens"]) / len(pc["reasoning_lens"])
                if pc["reasoning_lens"] else 0
            )
            proposal = SkillProposal(
                name="click_cluster_dismiss",
                subgoal="click dismiss buttons to clear overlays",
                action_types=["click"],
                training_pairs=pc["training_pairs"],
                occurrences=pc["occurrences"],
                steps_seen=pc["steps"],
                avg_reasoning_len=avg_reasoning,
                estimated_savings=len(pc["training_pairs"]) // pc["occurrences"],
                min_pairs_for_training=self.min_training_pairs,
                collecting=True,
            )
            self._proposed_signatures.add(sig)
            self._all_proposals.append(proposal)
            self._collecting[sig] = proposal
            return proposal

        return None

    # ------------------------------------------------------------------
    # Pattern: click + type + key (form fill)
    # ------------------------------------------------------------------

    def _detect_type_sequence_pattern(
        self, step_num: int, actions: list[dict],
    ) -> SkillProposal | None:
        sig = "form_fill"
        if sig in self._proposed_signatures:
            return None

        sequences = []
        for i in range(len(actions) - 2):
            sigs = [_action_signature(actions[j]) for j in range(i, i + 3)]
            if sigs == ["click", "type", "key"]:
                sequences.append(actions[i:i + 3])

        if not sequences:
            return None

        if sig not in self._pattern_counts:
            self._pattern_counts[sig] = {
                "occurrences": 0, "steps": [], "training_pairs": [],
                "reasoning_lens": [],
            }

        pc = self._pattern_counts[sig]
        pc["occurrences"] += len(sequences)
        pc["steps"].append(step_num)
        for seq in sequences:
            for a in seq:
                pc["training_pairs"].append(a)
                pc["reasoning_lens"].append(len(a.get("reasoning", "")))

        if pc["occurrences"] >= self.min_occurrences:
            avg_reasoning = (
                sum(pc["reasoning_lens"]) / len(pc["reasoning_lens"])
                if pc["reasoning_lens"] else 0
            )
            proposal = SkillProposal(
                name="form_fill_submit",
                subgoal="click input field type text and submit",
                action_types=["click", "type", "key"],
                training_pairs=pc["training_pairs"],
                occurrences=pc["occurrences"],
                steps_seen=pc["steps"],
                avg_reasoning_len=avg_reasoning,
                estimated_savings=3,
                min_pairs_for_training=self.min_training_pairs,
                collecting=True,
            )
            self._proposed_signatures.add(sig)
            self._all_proposals.append(proposal)
            self._collecting[sig] = proposal
            return proposal

        return None

    # ------------------------------------------------------------------
    # Continue collecting data for proposed skills
    # ------------------------------------------------------------------

    def _collect_for_existing(
        self, step_num: int, step_actions: list[dict],
    ) -> None:
        if not self._collecting:
            return

        if "scroll_click_select" in self._collecting:
            proposal = self._collecting["scroll_click_select"]
            current_scrolls = []
            for a in step_actions:
                s = _action_signature(a)
                if s.startswith("scroll_"):
                    current_scrolls.append(a)
                else:
                    if len(current_scrolls) >= 2 and s == "click":
                        for scroll_a in current_scrolls:
                            proposal.training_pairs.append(scroll_a)
                        proposal.training_pairs.append(a)
                        proposal.occurrences += 1
                    current_scrolls = []

        if "click_cluster_dismiss" in self._collecting:
            proposal = self._collecting["click_cluster_dismiss"]
            for a in step_actions:
                if (a.get("tool") == "computer"
                        and a.get("action", {}).get("action") == "left_click"
                        and a.get("ssim", 1.0) < 0.95):
                    proposal.training_pairs.append(a)
                    proposal.occurrences += 1

        if "form_fill" in self._collecting:
            proposal = self._collecting["form_fill"]
            for i in range(len(step_actions) - 2):
                sigs = [_action_signature(step_actions[j]) for j in range(i, i + 3)]
                if sigs == ["click", "type", "key"]:
                    for a in step_actions[i:i + 3]:
                        proposal.training_pairs.append(a)
                    proposal.occurrences += 1

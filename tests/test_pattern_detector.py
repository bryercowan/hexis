"""Tests for PatternDetector: scroll+click, click cluster, form fill patterns."""

import pytest

from cua_sl.self_improve.pattern_detector import (
    PatternDetector,
    SkillProposal,
    _action_signature,
    _click_region,
)


def _make_action(action_type: str, coord=None, tool="computer", ssim=0.5, reasoning=""):
    a = {"tool": tool, "action": {"action": action_type}, "ssim": ssim, "reasoning": reasoning}
    if coord:
        a["action"]["coordinate"] = coord
    return a


def _scroll_action(direction="down"):
    return _make_action("scroll", ssim=0.8)


def _click_action(x=100, y=200, ssim=0.5):
    return _make_action("left_click", coord=[x, y], ssim=ssim)


def _type_action():
    return _make_action("type", ssim=0.8)


def _key_action():
    return _make_action("key", ssim=0.8)


class TestActionSignature:
    def test_click(self):
        assert _action_signature(_click_action()) == "click"

    def test_scroll(self):
        a = {"tool": "computer", "action": {"action": "scroll", "scroll_direction": "down"}}
        assert _action_signature(a) == "scroll_down"

    def test_type(self):
        assert _action_signature(_type_action()) == "type"

    def test_key(self):
        assert _action_signature(_key_action()) == "key"

    def test_cnn_tool(self):
        a = {"tool": "dismiss_popups", "action": {}}
        assert _action_signature(a) == "cnn:dismiss_popups"


class TestClickRegion:
    def test_valid_click(self):
        a = _click_action(150, 250)
        region = _click_region(a)
        assert region is not None
        assert region == (0, 1)  # 150//200=0, 250//200=1

    def test_non_click(self):
        a = _type_action()
        assert _click_region(a) is None


class TestSkillProposal:
    def test_is_motor(self):
        p = SkillProposal(
            name="test", subgoal="test", action_types=["click"],
            avg_reasoning_len=100,
        )
        assert p.is_motor is True

        p.avg_reasoning_len = 500
        assert p.is_motor is False

    def test_ready_for_training(self):
        p = SkillProposal(
            name="test", subgoal="test", action_types=["click"],
            min_pairs_for_training=10,
        )
        assert p.ready_for_training is False
        p.training_pairs = [{}] * 10
        assert p.ready_for_training is True

    def test_collection_progress(self):
        p = SkillProposal(
            name="test", subgoal="test", action_types=["click"],
            min_pairs_for_training=100,
            training_pairs=[{}] * 50,
        )
        assert p.collection_progress == pytest.approx(0.5)

    def test_to_sft_dataset(self):
        pairs = [
            {"screenshot_before": b"img1", "action": {"click": [1, 2]}},
            {"screenshot_before": b"img2", "action": {"click": [3, 4]}},
        ]
        p = SkillProposal(
            name="test", subgoal="do something", action_types=["click"],
            training_pairs=pairs,
        )
        ds = p.to_sft_dataset()
        assert len(ds) == 2
        assert ds[0]["subgoal"] == "do something"


class TestPatternDetector:
    def test_no_proposals_from_single_action(self):
        d = PatternDetector(min_occurrences=1)
        proposals = d.analyze_step(1, [_click_action()])
        assert proposals == []

    def test_scroll_click_pattern(self):
        d = PatternDetector(min_occurrences=2, min_training_pairs=5)

        # Make scroll_action properly: need scroll_direction in action
        def scroll_a():
            return {"tool": "computer", "action": {"action": "scroll", "scroll_direction": "down"},
                    "ssim": 0.8, "reasoning": "scroll"}

        actions = [scroll_a(), scroll_a(), _click_action()]

        # First step
        proposals = d.analyze_step(1, actions)
        assert len(proposals) == 0  # Need min_occurrences=2

        # Second step
        proposals = d.analyze_step(2, actions)
        assert len(proposals) == 1
        assert proposals[0].name == "scroll_click_select"

    def test_click_cluster_pattern(self):
        d = PatternDetector(min_occurrences=3, min_training_pairs=5)

        actions = [
            _click_action(100, 200, ssim=0.5),
            _click_action(120, 220, ssim=0.6),
            _click_action(300, 400, ssim=0.7),
        ]

        # Need 3 occurrences
        d.analyze_step(1, actions)
        d.analyze_step(2, actions)
        proposals = d.analyze_step(3, actions)
        assert len(proposals) == 1
        assert proposals[0].name == "click_cluster_dismiss"

    def test_form_fill_pattern(self):
        d = PatternDetector(min_occurrences=2, min_training_pairs=3)

        actions = [_click_action(), _type_action(), _key_action()]

        d.analyze_step(1, actions)
        proposals = d.analyze_step(2, actions)

        found = [p for p in proposals if p.name == "form_fill_submit"]
        assert len(found) == 1

    def test_no_duplicate_proposals(self):
        d = PatternDetector(min_occurrences=1, min_training_pairs=3)

        def scroll_a():
            return {"tool": "computer", "action": {"action": "scroll", "scroll_direction": "down"},
                    "ssim": 0.8, "reasoning": ""}

        actions = [scroll_a(), scroll_a(), _click_action()]
        p1 = d.analyze_step(1, actions)
        p2 = d.analyze_step(2, actions)
        # Second time should not re-propose the same pattern
        scroll_proposals = [p for p in p2 if p.name == "scroll_click_select"]
        assert len(scroll_proposals) == 0

    def test_skills_ready_for_training(self):
        d = PatternDetector(min_occurrences=1, min_training_pairs=100)

        def scroll_a():
            return {"tool": "computer", "action": {"action": "scroll", "scroll_direction": "down"},
                    "ssim": 0.8, "reasoning": ""}

        actions = [scroll_a(), scroll_a(), _click_action()]
        d.analyze_step(1, actions)

        # Initially not ready (only 3 pairs, need 100)
        assert d.skills_ready_for_training() == []

        # Force enough pairs
        for name, proposal in d.collecting_skills.items():
            proposal.training_pairs = [{}] * 200

        ready = d.skills_ready_for_training()
        assert len(ready) >= 1

    def test_proposals_property(self):
        d = PatternDetector(min_occurrences=1, min_training_pairs=5)

        def scroll_a():
            return {"tool": "computer", "action": {"action": "scroll", "scroll_direction": "down"},
                    "ssim": 0.8, "reasoning": ""}

        d.analyze_step(1, [scroll_a(), scroll_a(), _click_action()])
        assert len(d.proposals) >= 1

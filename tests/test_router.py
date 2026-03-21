"""Tests for MoERouter: registration, text routing, learned routing."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from cua_sl.model.router import MoERouter


class FakeBackbone:
    """Minimal backbone mock for router tests (no GPU/model needed)."""

    def __init__(self, hidden_dim=256):
        self.hidden_dim = hidden_dim
        self.device = torch.device("cpu")
        self._embeddings = {}

    def tokenize_subgoal(self, subgoal: str):
        input_ids = torch.ones(1, 10, dtype=torch.long)
        attn_mask = torch.ones(1, 10, dtype=torch.long)
        return input_ids, attn_mask

    def text_features(self, input_ids, attn_mask):
        # Return deterministic features based on text hash for consistent routing
        B = input_ids.shape[0]
        torch.manual_seed(hash(str(input_ids.shape)) % 2**31)
        return torch.randn(B, 10, self.hidden_dim)


@pytest.fixture
def backbone():
    return FakeBackbone(hidden_dim=256)


@pytest.fixture
def router(backbone):
    return MoERouter(backbone, confidence_threshold=0.5)


class TestMoERouter:
    def test_register_expert(self, router):
        router.register_expert("popup", "dismiss popup green button")
        assert router.has_expert("popup")
        assert not router.has_expert("nonexistent")

    def test_route_no_experts_raises(self, router):
        with pytest.raises(RuntimeError, match="No experts registered"):
            router.route("test")

    def test_route_returns_string_or_none(self, router):
        router.register_expert("popup", "dismiss popup")
        result = router.route("dismiss popup")
        assert result is None or isinstance(result, str)

    def test_route_best_match(self, router):
        """With a single expert, route always picks it (if above threshold)."""
        router.confidence_threshold = -1.0  # Accept anything
        router.register_expert("popup", "dismiss popup")
        result = router.route("dismiss popup")
        assert result == "popup"

    def test_route_below_threshold(self, router):
        router.confidence_threshold = 2.0  # Impossibly high
        router.register_expert("popup", "dismiss popup")
        result = router.route("dismiss popup")
        assert result is None

    def test_register_none_expert(self, router):
        router.register_none_expert()
        assert router.has_expert("__none__")

    def test_enable_learned_routing(self, router):
        assert not router.uses_learned_routing
        router.enable_learned_routing(True)
        assert router.uses_learned_routing
        router.enable_learned_routing(False)
        assert not router.uses_learned_routing

    def test_logits_from_features(self, router, backbone):
        router.register_expert("popup", "dismiss popup")
        router.register_expert("modal", "solve radio modal")

        vis = torch.randn(1, 10, backbone.hidden_dim)
        txt = torch.randn(1, 10, backbone.hidden_dim)

        names, logits = router.logits_from_features(vis, txt)
        assert len(names) == 2
        assert "popup" in names
        assert "modal" in names
        assert logits.shape == (1, 2)

    def test_route_from_features_fallback(self, router):
        """Without learned routing, falls back to text-only."""
        router.confidence_threshold = -1.0
        router.register_expert("popup", "dismiss popup")
        result = router.route_from_features(None, None, "dismiss popup")
        assert result == "popup"

    def test_save_load(self, router, backbone, tmp_path):
        router.register_expert("popup", "dismiss popup")
        router.enable_learned_routing(True)

        router.save(tmp_path / "router_ckpt")

        new_router = MoERouter(backbone, confidence_threshold=0.5)
        new_router.register_expert("popup", "dismiss popup")
        new_router.load(tmp_path / "router_ckpt")

        assert new_router.uses_learned_routing

    def test_load_nonexistent_raises(self, router, tmp_path):
        with pytest.raises(FileNotFoundError):
            router.load(tmp_path / "nonexistent")

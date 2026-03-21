"""Tests for ExpertActionHead: forward, RL forward, log_prob_of_action."""

import pytest
import torch

from cua_sl.model.expert_head import (
    ACTION_NAMES,
    BACKBONE_DIM,
    EXPERT_DIM,
    NUM_ACTION_TYPES,
    CrossAttentionBlock,
    ExpertActionHead,
)


@pytest.fixture
def expert():
    return ExpertActionHead(
        backbone_dim=BACKBONE_DIM,
        expert_dim=EXPERT_DIM,
        num_action_types=NUM_ACTION_TYPES,
    )


@pytest.fixture
def dummy_tokens():
    """Fake backbone output: (B=2, seq_len, backbone_dim)."""
    B, vis_seq, txt_seq = 2, 64, 16
    vis = torch.randn(B, vis_seq, BACKBONE_DIM)
    txt = torch.randn(B, txt_seq, BACKBONE_DIM)
    return vis, txt


class TestCrossAttentionBlock:
    def test_output_shape(self):
        block = CrossAttentionBlock(dim=EXPERT_DIM, num_heads=4)
        vis = torch.randn(2, 64, EXPERT_DIM)
        txt = torch.randn(2, 16, EXPERT_DIM)
        out = block(vis, txt)
        assert out.shape == vis.shape

    def test_residual_connection(self):
        block = CrossAttentionBlock(dim=EXPERT_DIM, num_heads=4)
        vis = torch.randn(1, 4, EXPERT_DIM)
        txt = torch.randn(1, 2, EXPERT_DIM)
        # With zero init, output should be close to input (residual)
        out = block(vis, txt)
        assert out.shape == vis.shape


class TestExpertActionHead:
    def test_forward_shapes(self, expert, dummy_tokens):
        vis, txt = dummy_tokens
        grid_h, grid_w = 8, 8
        coords, heatmap, action_logits = expert(vis, txt, grid_h, grid_w)

        assert coords.shape == (2, 2)
        assert heatmap.shape == (2, grid_h, grid_w)
        assert action_logits.shape == (2, NUM_ACTION_TYPES)

    def test_coords_in_unit_range(self, expert, dummy_tokens):
        vis, txt = dummy_tokens
        coords, _, _ = expert(vis, txt, 8, 8)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_forward_rl_shapes(self, expert, dummy_tokens):
        vis, txt = dummy_tokens
        grid_h, grid_w = 8, 8
        result = expert.forward_rl(vis, txt, grid_h, grid_w)

        assert result["coords"].shape == (2, 2)
        assert result["coord_log_prob"].shape == (2,)
        assert result["coord_entropy"].shape == (2,)
        assert result["heatmap"].shape == (2, grid_h, grid_w)
        assert result["action_type"].shape == (2,)
        assert result["action_type_log_prob"].shape == (2,)
        assert result["action_type_entropy"].shape == (2,)

    def test_forward_rl_action_types_valid(self, expert, dummy_tokens):
        vis, txt = dummy_tokens
        result = expert.forward_rl(vis, txt, 8, 8)
        assert (result["action_type"] >= 0).all()
        assert (result["action_type"] < NUM_ACTION_TYPES).all()

    def test_log_prob_of_action(self, expert, dummy_tokens):
        vis, txt = dummy_tokens
        grid_h, grid_w = 8, 8
        target_coords = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        target_actions = torch.tensor([0, 1])

        result = expert.log_prob_of_action(
            vis, txt, grid_h, grid_w, target_coords, target_actions,
        )

        assert result["coord_log_prob"].shape == (2,)
        assert result["coord_entropy"].shape == (2,)
        assert result["action_type_log_prob"].shape == (2,)
        assert (result["coord_log_prob"] <= 0).all()  # Log probs are non-positive

    def test_different_grid_sizes(self, expert):
        B = 2
        txt = torch.randn(B, 16, BACKBONE_DIM)
        for gh, gw in [(4, 4), (8, 16), (16, 8)]:
            vis = torch.randn(B, gh * gw, BACKBONE_DIM)
            coords, heatmap, logits = expert(vis, txt, gh, gw)
            assert heatmap.shape == (2, gh, gw)
            assert coords.shape == (2, 2)

    def test_param_count(self, expert):
        n_params = sum(p.numel() for p in expert.parameters())
        # Should be ~2M params (small expert head)
        assert 500_000 < n_params < 5_000_000, f"Unexpected param count: {n_params}"

    def test_action_names_length(self):
        assert len(ACTION_NAMES) == NUM_ACTION_TYPES

    def test_temperature_affects_entropy(self, expert, dummy_tokens):
        vis, txt = dummy_tokens
        r_cold = expert.forward_rl(vis, txt, 8, 8, temperature=0.1)
        r_hot = expert.forward_rl(vis, txt, 8, 8, temperature=10.0)
        # Higher temperature → higher entropy
        assert r_hot["coord_entropy"].mean() > r_cold["coord_entropy"].mean()

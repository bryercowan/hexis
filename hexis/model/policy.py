"""MoEPolicy: frozen backbone + router + N expert heads.

The shared backbone is frozen. Each expert head (~2M, ~8MB)
can be independently trained, saved, and hot-loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from hexis.model.backbone import VLMBackbone
from hexis.model.expert_head import (
    BACKBONE_DIM,
    EXPERT_DIM,
    NUM_ACTION_TYPES,
    ExpertActionHead,
)
from hexis.model.router import MoERouter

log = logging.getLogger(__name__)


class MoEPolicy(nn.Module):
    """MoE VLM Policy: frozen backbone + router + N expert heads."""

    def __init__(self, backbone: VLMBackbone, router_confidence: float = 0.85):
        super().__init__()
        self.backbone = backbone
        self.router = MoERouter(backbone, confidence_threshold=router_confidence)
        self.experts: nn.ModuleDict = nn.ModuleDict()
        self._expert_subgoals: dict[str, str] = {}

    def register_expert(
        self,
        name: str,
        subgoal: str,
        expert: ExpertActionHead | None = None,
    ):
        """Register a new expert head."""
        if expert is None:
            expert = ExpertActionHead(backbone_dim=self.backbone.hidden_dim)
            expert.to(self.backbone.device)
        self.experts[name] = expert
        self._expert_subgoals[name] = subgoal
        self.router.register_expert(name, subgoal)

        n_params = sum(p.numel() for p in expert.parameters())
        log.info(
            "Registered expert '%s' (subgoal='%s', params=%s)",
            name, subgoal, f"{n_params:,}",
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        subgoal: str,
        conditioning_text: str | None = None,
        expert_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Deterministic forward: backbone → expert → (coords, heatmap, action_logits).

        Returns None if the router declines.
        """
        prompt_text = conditioning_text or subgoal
        with torch.no_grad():
            vis_tokens, grid_h, grid_w = self.backbone.vision_features(
                pixel_values, image_grid_thw,
            )
            input_ids, attn_mask = self.backbone.tokenize_subgoal(prompt_text)
            text_tokens = self.backbone.text_features(input_ids, attn_mask)

        if expert_name is None:
            expert_name = self.router.route_from_features(vis_tokens, text_tokens, prompt_text)
        if expert_name is None:
            return None
        expert = self.experts[expert_name]
        return expert(vis_tokens, text_tokens, grid_h, grid_w)

    def forward_rl(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        subgoal: str,
        temperature: float = 1.0,
        action_temperature: float = 1.0,
        conditioning_text: str | None = None,
        expert_name: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full RL forward: backbone features → expert head → sampled action."""
        prompt_text = conditioning_text or subgoal
        with torch.no_grad():
            vis_tokens, grid_h, grid_w = self.backbone.vision_features(
                pixel_values, image_grid_thw,
            )
            input_ids, attn_mask = self.backbone.tokenize_subgoal(prompt_text)
            text_tokens = self.backbone.text_features(input_ids, attn_mask)

        if expert_name is None:
            expert_name = self.router.route_from_features(vis_tokens, text_tokens, prompt_text)

        if expert_name is None:
            return self._no_expert_output(vis_tokens.device)

        expert = self.experts[expert_name]
        return expert.forward_rl(
            vis_tokens, text_tokens, grid_h, grid_w,
            temperature=temperature,
            action_temperature=action_temperature,
        )

    def forward_rl_cached(
        self,
        vis_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
        expert_name: str,
        temperature: float = 1.0,
        action_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """RL forward with pre-extracted backbone features (for training)."""
        expert = self.experts[expert_name]
        return expert.forward_rl(
            vis_tokens, text_tokens, grid_h, grid_w,
            temperature=temperature,
            action_temperature=action_temperature,
        )

    def forward_from_bytes(
        self,
        screenshot_bytes: bytes,
        subgoal: str,
        temperature: float = 0.1,
        conditioning_text: str | None = None,
        expert_name: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convenience: JPEG bytes → action dict."""
        img_inputs = self.backbone.preprocess_image(screenshot_bytes)
        return self.forward_rl(
            img_inputs["pixel_values"],
            img_inputs["image_grid_thw"],
            subgoal,
            temperature=temperature,
            conditioning_text=conditioning_text,
            expert_name=expert_name,
        )

    @torch.no_grad()
    def route_screenshot(self, screenshot_bytes: bytes, subgoal: str = "act") -> str | None:
        """Route a screenshot to an expert or return None (fallback to Claude).

        This is the entry point for hybrid mode: router decides first,
        Claude only gets called if no expert is confident enough.
        """
        if not self.experts:
            return None
        img_inputs = self.backbone.preprocess_image(screenshot_bytes)
        vis_tokens, _, _ = self.backbone.vision_features(
            img_inputs["pixel_values"], img_inputs["image_grid_thw"],
        )
        input_ids, attn_mask = self.backbone.tokenize_subgoal(subgoal)
        text_tokens = self.backbone.text_features(input_ids, attn_mask)
        return self.router.route_from_features(vis_tokens, text_tokens, subgoal)

    @staticmethod
    def _no_expert_output(device: torch.device) -> dict[str, torch.Tensor]:
        """High-entropy output when no expert matches — triggers escalation."""
        return {
            "coords": torch.tensor([[0.5, 0.5]], device=device),
            "coord_log_prob": torch.tensor([0.0], device=device),
            "coord_entropy": torch.tensor([999.0], device=device),
            "heatmap": torch.zeros(1, 1, 1, device=device),
            "action_type": torch.tensor([6], device=device),  # wait
            "action_type_log_prob": torch.tensor([0.0], device=device),
            "action_type_entropy": torch.tensor([999.0], device=device),
        }

    # ------------------------------------------------------------------
    # Backbone feature caching (for RL training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_backbone_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        subgoal: str,
        conditioning_text: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Extract and return backbone features for offline use."""
        vis_tokens, grid_h, grid_w = self.backbone.vision_features(
            pixel_values, image_grid_thw,
        )
        input_ids, attn_mask = self.backbone.tokenize_subgoal(conditioning_text or subgoal)
        text_tokens = self.backbone.text_features(input_ids, attn_mask)
        return vis_tokens, text_tokens, grid_h, grid_w

    @torch.no_grad()
    def extract_features_from_bytes(
        self,
        screenshot_bytes: bytes,
        subgoal: str,
        conditioning_text: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Extract backbone features from JPEG bytes."""
        img_inputs = self.backbone.preprocess_image(screenshot_bytes)
        return self.extract_backbone_features(
            img_inputs["pixel_values"],
            img_inputs["image_grid_thw"],
            subgoal,
            conditioning_text=conditioning_text,
        )

    # ------------------------------------------------------------------
    # Save / load individual experts
    # ------------------------------------------------------------------

    def save_expert(self, name: str, path: str | Path):
        """Save a single expert's weights (~8MB)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        expert = self.experts[name]
        torch.save({
            "expert_state_dict": expert.state_dict(),
            "subgoal": self._expert_subgoals[name],
            "backbone_dim": expert.backbone_dim,
            "expert_dim": expert.expert_dim,
            "num_action_types": expert.num_action_types,
        }, path / "expert.pt")
        log.info("Saved expert '%s' to %s (%s)", name, path,
                 f"{(path / 'expert.pt').stat().st_size / 1e6:.1f}MB")

    def load_expert(self, name: str, path: str | Path):
        """Load a single expert's weights from disk."""
        path = Path(path)
        ckpt_file = path / "expert.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"No expert checkpoint at {ckpt_file}")

        data = torch.load(ckpt_file, map_location="cpu", weights_only=True)

        expert = ExpertActionHead(
            backbone_dim=data.get("backbone_dim", BACKBONE_DIM),
            expert_dim=data.get("expert_dim", EXPERT_DIM),
            num_action_types=data.get("num_action_types", NUM_ACTION_TYPES),
        )
        expert.load_state_dict(data["expert_state_dict"])
        expert.to(self.backbone.device)

        subgoal = data["subgoal"]
        self.register_expert(name, subgoal, expert)
        log.info("Loaded expert '%s' from %s (subgoal='%s')", name, path, subgoal)

    def save_router(self, path: str | Path) -> None:
        self.router.save(path)

    def load_router(self, path: str | Path) -> None:
        self.router.register_none_expert()
        self.router.load(path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def available_experts(self) -> list[str]:
        return list(self.experts.keys())

    def expert_param_count(self, name: str) -> int:
        return sum(p.numel() for p in self.experts[name].parameters())

    def total_param_count(self) -> dict[str, int]:
        counts = {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
        }
        for name in self.experts:
            counts[name] = self.expert_param_count(name)
        return counts

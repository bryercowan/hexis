"""MoERouter: routes conditioning requests to expert heads.

Supports two modes:
- Fallback text-only routing via cosine similarity over registered subgoals
- Learned routing via a tiny query adapter over pooled vision/text features
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexis.model.backbone import VLMBackbone

log = logging.getLogger(__name__)


class MoERouter(nn.Module):
    """Routes conditioning requests to experts."""

    def __init__(self, backbone: VLMBackbone, confidence_threshold: float = 0.4):
        super().__init__()
        self.backbone = backbone
        self.confidence_threshold = confidence_threshold
        self._expert_embeddings: dict[str, torch.Tensor] = {}
        self.query_adapter = nn.Sequential(
            nn.Linear(backbone.hidden_dim * 2, backbone.hidden_dim),
            nn.GELU(),
            nn.Linear(backbone.hidden_dim, backbone.hidden_dim),
        ).to(backbone.device)
        self._learned_enabled = False

    def register_expert(self, name: str, subgoal: str):
        """Cache the embedding for an expert's subgoal."""
        input_ids, attn_mask = self.backbone.tokenize_subgoal(subgoal)
        with torch.no_grad():
            text_emb = self.backbone.text_features(input_ids, attn_mask)
            pooled = text_emb.mean(dim=1).squeeze(0)
            self._expert_embeddings[name] = F.normalize(pooled, dim=0)

    def route(self, subgoal: str) -> str | None:
        """Find the best-matching expert for a query text (fallback mode).

        Returns None if below confidence_threshold.
        """
        if not self._expert_embeddings:
            raise RuntimeError("No experts registered")

        input_ids, attn_mask = self.backbone.tokenize_subgoal(subgoal)
        with torch.no_grad():
            text_emb = self.backbone.text_features(input_ids, attn_mask)
            query = F.normalize(text_emb.mean(dim=1).squeeze(0), dim=0)

        best_name = None
        best_sim = -float("inf")
        for name, emb in self._expert_embeddings.items():
            sim = (query @ emb).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < self.confidence_threshold:
            log.info(
                "MoERouter: best match '%s' (sim=%.3f) below threshold %.3f, declining.",
                best_name, best_sim, self.confidence_threshold,
            )
            return None

        log.debug("MoERouter: routed to '%s' (sim=%.3f)", best_name, best_sim)
        return best_name

    def register_none_expert(self):
        """Register the __none__ virtual expert for learned routing."""
        input_ids, attn_mask = self.backbone.tokenize_subgoal("no expert needed")
        with torch.no_grad():
            text_emb = self.backbone.text_features(input_ids, attn_mask)
            pooled = text_emb.mean(dim=1).squeeze(0)
            self._expert_embeddings["__none__"] = F.normalize(pooled, dim=0)

    def route_from_features(
        self,
        vis_tokens: torch.Tensor | None,
        text_tokens: torch.Tensor | None,
        fallback_text: str,
    ) -> str | None:
        """Route using learned adapter if available, else text-only fallback.

        Returns None if no expert is confident enough, or if __none__ wins.
        """
        if not self._expert_embeddings:
            raise RuntimeError("No experts registered")

        if self._learned_enabled and vis_tokens is not None and text_tokens is not None:
            expert_names, logits = self.logits_from_features(vis_tokens, text_tokens)
            probs = F.softmax(logits, dim=-1)
            best_idx = int(probs.argmax(dim=-1).item())
            best_name = expert_names[best_idx]
            best_conf = probs[0, best_idx].item()

            if best_name == "__none__" or best_conf < self.confidence_threshold:
                return None

            log.debug("MoERouter: learned routing → '%s' (conf=%.3f)", best_name, best_conf)
            return best_name

        return self.route(fallback_text)

    def logits_from_features(
        self,
        vis_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor]:
        """Return routing logits over registered experts for a feature batch."""
        if not self._expert_embeddings:
            raise RuntimeError("No experts registered")

        vis_pooled = vis_tokens.mean(dim=1).float()
        txt_pooled = text_tokens.mean(dim=1).float()
        query = self.query_adapter(torch.cat([vis_pooled, txt_pooled], dim=-1))
        query = F.normalize(query, dim=-1)

        expert_names = list(self._expert_embeddings.keys())
        expert_matrix = torch.stack(
            [self._expert_embeddings[name].to(query.device) for name in expert_names],
            dim=0,
        ).float()
        logits = query @ expert_matrix.T
        return expert_names, logits

    def has_expert(self, name: str) -> bool:
        return name in self._expert_embeddings

    def enable_learned_routing(self, enabled: bool = True) -> None:
        self._learned_enabled = enabled

    @property
    def uses_learned_routing(self) -> bool:
        return self._learned_enabled

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "query_adapter_state_dict": self.query_adapter.state_dict(),
                "learned_enabled": self._learned_enabled,
            },
            path / "router.pt",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = path / "router.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No router checkpoint at {ckpt}")
        data = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.query_adapter.load_state_dict(data["query_adapter_state_dict"])
        self.query_adapter.to(self.backbone.device)
        self._learned_enabled = bool(data.get("learned_enabled", True))

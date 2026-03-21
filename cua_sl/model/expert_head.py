"""ExpertActionHead: cross-attention + conv heatmap + action classifier (~2M params).

Takes projected vision and text tokens from the frozen backbone,
applies cross-attention for task-conditioned spatial reasoning,
then produces a spatial heatmap and action type logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (exported for use by training scripts)
# ---------------------------------------------------------------------------

BACKBONE_DIM = 2560       # Qwen3-VL-4B-Instruct hidden_size
EXPERT_DIM = 256           # Expert head internal dimension
NUM_ACTION_TYPES = 8
IMG_H, IMG_W = 720, 1280

ACTION_NAMES = [
    "click", "scroll_up", "scroll_down", "key_enter",
    "key_escape", "key_home", "wait", "done",
]


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: vision tokens attend to text tokens."""

    def __init__(self, dim: int = EXPERT_DIM, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        vis_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        residual = vis_tokens
        vis_normed = self.norm1(vis_tokens)
        attn_out, _ = self.attn(vis_normed, text_tokens, text_tokens)
        vis_tokens = residual + attn_out

        residual = vis_tokens
        vis_tokens = residual + self.ffn(self.norm2(vis_tokens))

        return vis_tokens


# ---------------------------------------------------------------------------
# Expert Action Head (~2M params)
# ---------------------------------------------------------------------------

class ExpertActionHead(nn.Module):
    """Tiny expert head: cross-attention + conv heatmap + action classifier.

    ~2M trainable parameters per expert.
    """

    def __init__(
        self,
        backbone_dim: int = BACKBONE_DIM,
        expert_dim: int = EXPERT_DIM,
        num_heads: int = 4,
        num_cross_attn_layers: int = 2,
        num_action_types: int = NUM_ACTION_TYPES,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.expert_dim = expert_dim
        self.num_action_types = num_action_types

        # Project backbone → expert dim
        self.vis_proj = nn.Linear(backbone_dim, expert_dim)
        self.txt_proj = nn.Linear(backbone_dim, expert_dim)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(expert_dim, num_heads)
            for _ in range(num_cross_attn_layers)
        ])

        # Conv heatmap head: 256 → 128 → 64 → 1
        self.conv_head = nn.Sequential(
            nn.Conv2d(expert_dim, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

        # Action type classifier from pooled features
        self.action_head = nn.Sequential(
            nn.Linear(expert_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_action_types),
        )

        # Dynamic coordinate grids (set on first forward)
        self._grid_h = 0
        self._grid_w = 0

    def _ensure_grids(self, grid_h: int, grid_w: int, device: torch.device):
        """Create/update coordinate grids when spatial dims change."""
        if self._grid_h == grid_h and self._grid_w == grid_w:
            return
        self._grid_h = grid_h
        self._grid_w = grid_w

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, grid_h),
            torch.linspace(0, 1, grid_w),
            indexing="ij",
        )
        self.register_buffer("grid_x", grid_x.flatten().to(device), persistent=False)
        self.register_buffer("grid_y", grid_y.flatten().to(device), persistent=False)
        self.register_buffer(
            "x_coords", torch.linspace(0, 1, grid_w).to(device), persistent=False,
        )
        self.register_buffer(
            "y_coords", torch.linspace(0, 1, grid_h).to(device), persistent=False,
        )

    def _project_and_attend(
        self,
        vis_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Project backbone features and run cross-attention."""
        vis = self.vis_proj(vis_tokens.float())
        txt = self.txt_proj(text_tokens.float())

        for layer in self.cross_attn_layers:
            vis = layer(vis, txt)

        return vis

    def forward(
        self,
        vis_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic forward: weighted centroid from heatmap.

        Returns:
            coords: (B, 2) predicted (x, y) in [0, 1].
            heatmap: (B, grid_h, grid_w) logit heatmap.
            action_logits: (B, num_action_types).
        """
        B = vis_tokens.shape[0]
        device = vis_tokens.device
        self._ensure_grids(grid_h, grid_w, device)

        vis = self._project_and_attend(vis_tokens, text_tokens)

        spatial_2d = vis.permute(0, 2, 1).reshape(
            B, self.expert_dim, grid_h, grid_w,
        )

        heatmap = self.conv_head(spatial_2d).squeeze(1)

        weights = F.softmax(heatmap.flatten(1), dim=-1)
        x = (weights * self.grid_x).sum(dim=-1)
        y = (weights * self.grid_y).sum(dim=-1)
        coords = torch.stack([x, y], dim=-1)

        vis_pooled = vis.mean(dim=1)
        action_logits = self.action_head(vis_pooled)

        return coords, heatmap, action_logits

    def forward_rl(
        self,
        vis_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
        temperature: float = 1.0,
        action_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """RL forward: factorized x/y sampling from heatmap.

        Returns dict with coords, coord_log_prob, coord_entropy, heatmap,
        action_type, action_type_log_prob, action_type_entropy.
        """
        from torch.distributions import Categorical

        B = vis_tokens.shape[0]
        device = vis_tokens.device
        self._ensure_grids(grid_h, grid_w, device)

        vis = self._project_and_attend(vis_tokens, text_tokens)

        spatial_2d = vis.permute(0, 2, 1).reshape(
            B, self.expert_dim, grid_h, grid_w,
        )
        heatmap = self.conv_head(spatial_2d).squeeze(1)

        # Factorized x/y sampling
        x_logits = torch.logsumexp(heatmap, dim=1) / temperature
        y_logits = torch.logsumexp(heatmap, dim=2) / temperature

        x_dist = Categorical(logits=x_logits)
        y_dist = Categorical(logits=y_logits)

        x_idx = x_dist.sample()
        y_idx = y_dist.sample()

        coords = torch.stack([
            self.x_coords[x_idx],
            self.y_coords[y_idx],
        ], dim=-1)

        coord_log_prob = x_dist.log_prob(x_idx) + y_dist.log_prob(y_idx)
        coord_entropy = x_dist.entropy() + y_dist.entropy()

        # Action type
        vis_pooled = vis.mean(dim=1)
        action_logits = self.action_head(vis_pooled) / action_temperature
        action_dist = Categorical(logits=action_logits)
        action_type = action_dist.sample()

        return {
            "coords": coords,
            "coord_log_prob": coord_log_prob,
            "coord_entropy": coord_entropy,
            "heatmap": heatmap,
            "action_type": action_type,
            "action_type_log_prob": action_dist.log_prob(action_type),
            "action_type_entropy": action_dist.entropy(),
        }

    def log_prob_of_action(
        self,
        vis_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
        expert_coords: torch.Tensor,
        expert_action_type: torch.Tensor,
        temperature: float = 1.0,
        action_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute log probability of given expert actions under current policy."""
        from torch.distributions import Categorical

        B = vis_tokens.shape[0]
        device = vis_tokens.device
        self._ensure_grids(grid_h, grid_w, device)

        vis = self._project_and_attend(vis_tokens, text_tokens)

        spatial_2d = vis.permute(0, 2, 1).reshape(
            B, self.expert_dim, grid_h, grid_w,
        )
        heatmap = self.conv_head(spatial_2d).squeeze(1)

        x_logits = torch.logsumexp(heatmap, dim=1) / temperature
        y_logits = torch.logsumexp(heatmap, dim=2) / temperature

        x_dist = Categorical(logits=x_logits)
        y_dist = Categorical(logits=y_logits)

        ex = expert_coords[:, 0]
        ey = expert_coords[:, 1]
        x_idx = (ex * (grid_w - 1)).round().long().clamp(0, grid_w - 1)
        y_idx = (ey * (grid_h - 1)).round().long().clamp(0, grid_h - 1)

        coord_log_prob = x_dist.log_prob(x_idx) + y_dist.log_prob(y_idx)
        coord_entropy = x_dist.entropy() + y_dist.entropy()

        vis_pooled = vis.mean(dim=1)
        action_logits = self.action_head(vis_pooled) / action_temperature
        action_dist = Categorical(logits=action_logits)

        return {
            "coord_log_prob": coord_log_prob,
            "coord_entropy": coord_entropy,
            "action_type_log_prob": action_dist.log_prob(expert_action_type),
            "action_type_entropy": action_dist.entropy(),
        }

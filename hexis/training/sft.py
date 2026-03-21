"""Expert SFT: feature caching, heatmap KL loss.

Trains a tiny ExpertActionHead (~2M params) using supervised heatmap loss.
The shared Qwen backbone stays frozen; only the expert head updates.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from hexis.model.backbone import IMG_H, IMG_W, VLMBackbone
from hexis.model.expert_head import (
    EXPERT_DIM,
    NUM_ACTION_TYPES,
    ExpertActionHead,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ExpertSFTDataset(Dataset):
    """Loads SFT supervision from shards or JSONL."""

    def __init__(self, data_path: str | Path, default_conditioning_text: str):
        self.data_path = Path(data_path)
        self.default_conditioning_text = default_conditioning_text
        self._mode = "jsonl" if self.data_path.is_file() else "shards"
        self._cached_shard_idx = -1
        self._cached_data = None

        if self._mode == "jsonl":
            self.records = [
                json.loads(line)
                for line in self.data_path.read_text().splitlines()
                if line.strip()
            ]
            if not self.records:
                raise FileNotFoundError(f"No JSONL records in {data_path}")
            self.total = len(self.records)
            log.info("Loaded %d JSONL records from %s", self.total, data_path)
            return

        self.shards = sorted(self.data_path.glob("shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.pt files in {data_path}")

        self.shard_sizes = []
        self.cumulative = [0]
        for shard_path in self.shards:
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            n = len(data.get("screenshot_jpeg", data.get("screenshots", [])))
            self.shard_sizes.append(n)
            self.cumulative.append(self.cumulative[-1] + n)

        self.total = self.cumulative[-1]
        log.info("Loaded %d shards, %d total samples from %s",
                 len(self.shards), self.total, data_path)

    def __len__(self):
        return self.total

    def _load_shard(self, shard_idx: int):
        if shard_idx == self._cached_shard_idx:
            return self._cached_data
        self._cached_data = torch.load(
            self.shards[shard_idx], map_location="cpu", weights_only=False,
        )
        self._cached_shard_idx = shard_idx
        return self._cached_data

    def __getitem__(self, idx):
        if self._mode == "jsonl":
            return self._getitem_jsonl(idx)

        shard_idx = 0
        for i, cum in enumerate(self.cumulative[1:], 0):
            if idx < cum:
                shard_idx = i
                break

        local_idx = idx - self.cumulative[shard_idx]
        data = self._load_shard(shard_idx)

        screenshots = data.get("screenshot_jpeg", data.get("screenshots", []))
        jpeg_bytes = screenshots[local_idx]
        if isinstance(jpeg_bytes, torch.Tensor):
            jpeg_bytes = jpeg_bytes.numpy().tobytes()

        if "target_x" in data:
            tx = float(data["target_x"][local_idx])
            ty = float(data["target_y"][local_idx])
        elif "actions" in data:
            action = data["actions"][local_idx]
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            tx = float(action[8] if len(action) > 8 else 0.5)
            ty = float(action[9] if len(action) > 9 else 0.5)
        else:
            tx, ty = 0.5, 0.5

        if "action_type" in data:
            at = int(data["action_type"][local_idx])
        elif "actions" in data:
            action = data["actions"][local_idx]
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            at = int(np.argmax(action[:NUM_ACTION_TYPES]))
        else:
            at = 0

        return jpeg_bytes, tx, ty, at, self.default_conditioning_text

    def _getitem_jsonl(self, idx: int):
        record = self.records[idx]
        screenshot_b64 = record.get("screenshot_b64", "")
        if not screenshot_b64:
            raise ValueError(f"Record {idx} missing screenshot_b64")
        jpeg_bytes = base64.b64decode(screenshot_b64)
        tx, ty, at = extract_action_supervision(record.get("action", {}))
        conditioning_text = (
            record.get("conditioning_text")
            or record.get("subgoal")
            or self.default_conditioning_text
        )
        return jpeg_bytes, tx, ty, at, conditioning_text


# ---------------------------------------------------------------------------
# Action supervision extraction
# ---------------------------------------------------------------------------

def _normalize_coord(value: float, size: int) -> float:
    value_f = float(value)
    if 0.0 <= value_f <= 1.0:
        return value_f
    return max(0.0, min(1.0, value_f / size))


def extract_action_supervision(action: dict) -> tuple[float, float, int]:
    """Map a recorded GUI action into expert coordinate + action-type targets."""
    if not isinstance(action, dict):
        return 0.5, 0.5, 0

    action_name = str(action.get("action", "")).lower()
    coord = action.get("coordinate") or action.get("coords") or []
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        coord = [IMG_W * 0.5, IMG_H * 0.5]

    tx = _normalize_coord(coord[0], IMG_W)
    ty = _normalize_coord(coord[1], IMG_H)

    if action_name in ("left_click", "right_click", "double_click", "click"):
        return tx, ty, 0
    if action_name == "scroll":
        direction = str(action.get("scroll_direction", "down")).lower()
        return tx, ty, 1 if direction == "up" else 2
    if action_name == "key":
        key = str(action.get("text", "")).strip().lower()
        key_map = {"enter": 3, "escape": 4, "esc": 4, "home": 5}
        return tx, ty, key_map.get(key, 6)
    if action_name == "wait":
        return tx, ty, 6
    if action_name == "done":
        return tx, ty, 7
    return tx, ty, 0


# ---------------------------------------------------------------------------
# Heatmap target generation
# ---------------------------------------------------------------------------

def make_heatmap_target(
    x: torch.Tensor, y: torch.Tensor,
    grid_h: int, grid_w: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Generate Gaussian heatmap target as a probability distribution."""
    device = x.device
    B = x.shape[0]

    gy = torch.linspace(0, 1, grid_h, device=device)
    gx = torch.linspace(0, 1, grid_w, device=device)
    grid_yy, grid_xx = torch.meshgrid(gy, gx, indexing="ij")

    cell_w = 1.0 / max(grid_w - 1, 1)
    cell_h = 1.0 / max(grid_h - 1, 1)
    sigma_x = sigma * cell_w
    sigma_y = sigma * cell_h

    dx = grid_xx.unsqueeze(0) - x.reshape(B, 1, 1)
    dy = grid_yy.unsqueeze(0) - y.reshape(B, 1, 1)
    heatmap = torch.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2)))

    heatmap_flat = heatmap.reshape(B, -1)
    heatmap_flat = heatmap_flat / (heatmap_flat.sum(dim=-1, keepdim=True) + 1e-8)

    return heatmap_flat


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    dataset: ExpertSFTDataset,
    backbone: VLMBackbone,
    cache_path: str | Path | None = None,
) -> tuple[list[dict], int, int]:
    """Pre-extract backbone features for all samples."""
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            log.info("Loading cached features from %s ...", cache_file)
            load_start = time.time()
            cache_data = torch.load(cache_file, map_location="cpu", weights_only=False)
            if cache_data["n_samples"] == len(dataset):
                load_elapsed = time.time() - load_start
                log.info(
                    "Loaded %d cached features in %.1fs (grid=%dx%d)",
                    cache_data["n_samples"], load_elapsed,
                    cache_data["grid_h"], cache_data["grid_w"],
                )
                return cache_data["features"], cache_data["grid_h"], cache_data["grid_w"]

    cached_features = []
    cache_grid_h = cache_grid_w = 0
    cache_start = time.time()

    for i in range(len(dataset)):
        jpeg_bytes, tx, ty, at, conditioning_text = dataset[i]

        with torch.no_grad():
            img_inputs = backbone.preprocess_image(jpeg_bytes)
            vis, gh, gw = backbone.vision_features(
                img_inputs["pixel_values"], img_inputs["image_grid_thw"],
            )
            input_ids, attn_mask = backbone.tokenize_subgoal(conditioning_text)
            txt = backbone.text_features(input_ids, attn_mask)

        cached_features.append({
            "vis": vis.cpu(), "txt": txt.cpu(),
            "tx": tx, "ty": ty, "at": at,
        })
        cache_grid_h, cache_grid_w = gh, gw

        if (i + 1) % 100 == 0 or i == len(dataset) - 1:
            elapsed_cache = time.time() - cache_start
            rate = (i + 1) / elapsed_cache
            eta = (len(dataset) - i - 1) / rate if rate > 0 else 0
            log.info("  cached %d/%d (%.1f/s, ETA %.0fs)", i + 1, len(dataset), rate, eta)

    # Pad text tokens to uniform length
    max_txt_len = max(f["txt"].shape[1] for f in cached_features)
    for f in cached_features:
        txt = f["txt"]
        if txt.shape[1] < max_txt_len:
            pad = torch.zeros(1, max_txt_len - txt.shape[1], txt.shape[2], dtype=txt.dtype)
            f["txt"] = torch.cat([txt, pad], dim=1)

    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "features": cached_features,
            "grid_h": cache_grid_h,
            "grid_w": cache_grid_w,
            "n_samples": len(cached_features),
        }, cache_file)

    return cached_features, cache_grid_h, cache_grid_w


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def run_epoch(
    expert: ExpertActionHead,
    features: list[dict],
    grid_h: int,
    grid_w: int,
    device: torch.device,
    batch_size: int,
    sigma: float,
    centroid_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMAModel | None = None,
) -> dict[str, float]:
    """Run one epoch of training or validation."""
    is_train = optimizer is not None
    if is_train:
        expert.train()
    else:
        expert.eval()

    epoch_losses = []
    epoch_coord_losses = []
    epoch_type_losses = []
    epoch_px_errors = []

    indices = np.random.permutation(len(features)) if is_train else np.arange(len(features))

    from contextlib import nullcontext
    ctx = torch.no_grad() if not is_train else nullcontext()
    with ctx:
        for batch_start in range(0, len(features), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            if len(batch_indices) == 0:
                continue

            batch_vis, batch_txt = [], []
            batch_tx, batch_ty, batch_at = [], [], []

            for idx in batch_indices:
                feat = features[int(idx)]
                batch_vis.append(feat["vis"])
                batch_txt.append(feat["txt"])
                batch_tx.append(feat["tx"])
                batch_ty.append(feat["ty"])
                batch_at.append(feat["at"])

            vis_batch = torch.cat(batch_vis, dim=0).to(device)
            txt_batch = torch.cat(batch_txt, dim=0).to(device)
            tx_t = torch.tensor(batch_tx, device=device)
            ty_t = torch.tensor(batch_ty, device=device)
            at_t = torch.tensor(batch_at, dtype=torch.long, device=device)

            coords, heatmap, action_logits = expert(vis_batch, txt_batch, grid_h, grid_w)

            target_dist = make_heatmap_target(tx_t, ty_t, grid_h, grid_w, sigma=sigma)
            pred_log_probs = F.log_softmax(heatmap.flatten(1), dim=-1)
            kl_loss = F.kl_div(pred_log_probs, target_dist, reduction="batchmean")

            target_coords = torch.stack([tx_t, ty_t], dim=-1)
            centroid_loss = F.l1_loss(coords, target_coords)
            coord_loss = kl_loss + centroid_weight * centroid_loss
            type_loss = F.cross_entropy(action_logits, at_t)
            loss = coord_loss + 0.1 * type_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(expert)

            with torch.no_grad():
                pred_px_x = coords[:, 0] * IMG_W
                pred_px_y = coords[:, 1] * IMG_H
                tgt_px_x = tx_t * IMG_W
                tgt_px_y = ty_t * IMG_H
                px_err = torch.sqrt((pred_px_x - tgt_px_x)**2 + (pred_px_y - tgt_px_y)**2)

            epoch_losses.append(loss.item())
            epoch_coord_losses.append(coord_loss.item())
            epoch_type_losses.append(type_loss.item())
            epoch_px_errors.append(px_err.mean().item())

    return {
        "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
        "coord_loss": float(np.mean(epoch_coord_losses)) if epoch_coord_losses else 0.0,
        "type_loss": float(np.mean(epoch_type_losses)) if epoch_type_losses else 0.0,
        "px_error": float(np.mean(epoch_px_errors)) if epoch_px_errors else 0.0,
    }


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_expert(
    expert: ExpertActionHead,
    ema: EMAModel,
    subgoal: str,
    path: Path,
):
    """Save expert + EMA checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "expert_state_dict": expert.state_dict(),
        "subgoal": subgoal,
        "backbone_dim": expert.backbone_dim,
        "expert_dim": expert.expert_dim,
        "num_action_types": expert.num_action_types,
    }, path / "expert.pt")
    torch.save(ema.state_dict(), path / "ema_expert.pt")
    log.info("Saved expert to %s", path)

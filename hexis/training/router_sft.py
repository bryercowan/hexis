"""Router query_adapter cross-entropy training.

Trains only the tiny router adapter; the shared Qwen-VL backbone stays frozen.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from hexis.model.backbone import VLMBackbone
from hexis.model.router import MoERouter

log = logging.getLogger(__name__)


def load_records(path: str | Path) -> list[dict]:
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"No records found in {path}")
    return records


def parse_expert_specs(specs: list[str]) -> dict[str, str]:
    expert_map: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid expert spec '{spec}', expected name=subgoal")
        name, subgoal = spec.split("=", 1)
        name = name.strip()
        subgoal = subgoal.strip()
        if not name or not subgoal:
            raise ValueError(f"Invalid expert spec '{spec}', expected name=subgoal")
        expert_map[name] = subgoal
    if not expert_map:
        raise ValueError("At least one expert spec is required")
    return expert_map


def train_router(
    backbone: VLMBackbone,
    router: MoERouter,
    expert_map: dict[str, str],
    data_path: str | Path,
    output_dir: str | Path,
    none_data_path: str | Path | None = None,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 16,
) -> float:
    """Train the router query adapter with cached features. Returns best loss."""
    import random
    import time

    import numpy as np

    for name, subgoal in expert_map.items():
        if not router.has_expert(name):
            router.register_expert(name, subgoal)

    has_none = none_data_path is not None
    if has_none:
        router.register_none_expert()

    router.enable_learned_routing(True)
    router.query_adapter.train()

    expert_names = list(expert_map.keys())
    if has_none:
        expert_names.append("__none__")
    label_map = {name: idx for idx, name in enumerate(expert_names)}

    records = load_records(data_path)
    if has_none:
        none_records = load_records(none_data_path)
        expert_counts = {}
        for r in records:
            en = r.get("expert_name", "")
            expert_counts[en] = expert_counts.get(en, 0) + 1
        if expert_counts:
            median_count = sorted(expert_counts.values())[len(expert_counts) // 2]
            max_none = median_count * 2
            if len(none_records) > max_none:
                random.seed(42)
                none_records = random.sample(none_records, max_none)
        records.extend(none_records)

    # --- Phase 1: Extract and cache all backbone features (one-time cost) ---
    log.info("Caching backbone features for %d records...", len(records))
    device = backbone.device
    cached_features: list[dict] = []
    t0 = time.time()

    for i, record in enumerate(records):
        expert_name = record.get("expert_name")
        if expert_name not in label_map:
            continue

        screenshot_b64 = record.get("screenshot_b64", "")
        task_text = record.get("task_text", "")
        if not screenshot_b64 or not task_text:
            continue

        screenshot_bytes = base64.b64decode(screenshot_b64)
        img_inputs = backbone.preprocess_image(screenshot_bytes)
        with torch.no_grad():
            vis_tokens, _, _ = backbone.vision_features(
                img_inputs["pixel_values"], img_inputs["image_grid_thw"],
            )
            input_ids, attn_mask = backbone.tokenize_subgoal(task_text)
            text_tokens = backbone.text_features(input_ids, attn_mask)

        # Pool and concat now (what the router adapter expects)
        vis_pooled = vis_tokens.mean(dim=1).cpu()
        txt_pooled = text_tokens.mean(dim=1).cpu()

        cached_features.append({
            "vis_pooled": vis_pooled,
            "txt_pooled": txt_pooled,
            "label": label_map[expert_name],
        })

        if (i + 1) % 50 == 0 or i == len(records) - 1:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(records) - i - 1) / rate if rate > 0 else 0
            log.info("  cached %d/%d (%.1f/s, ETA %.0fs)", i + 1, len(records), rate, eta)

    log.info("Cached %d features in %.1fs", len(cached_features), time.time() - t0)

    # --- Phase 2: Train on cached features with mini-batches ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build expert embedding matrix (fixed order)
    expert_matrix = torch.stack(
        [router._expert_embeddings[name].to(device) for name in expert_names],
        dim=0,
    ).float()

    optimizer = torch.optim.AdamW(router.query_adapter.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")

    for epoch in range(epochs):
        indices = np.random.permutation(len(cached_features))
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]

            vis_batch = torch.cat([cached_features[i]["vis_pooled"] for i in batch_idx], dim=0).to(device)
            txt_batch = torch.cat([cached_features[i]["txt_pooled"] for i in batch_idx], dim=0).to(device)
            labels = torch.tensor([cached_features[i]["label"] for i in batch_idx], device=device)

            query_input = torch.cat([vis_batch.float(), txt_batch.float()], dim=-1)
            query = F.normalize(router.query_adapter(query_input), dim=-1)
            logits = query @ expert_matrix.T

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.query_adapter.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)

        # Compute accuracy
        with torch.no_grad():
            all_vis = torch.cat([f["vis_pooled"] for f in cached_features], dim=0).to(device)
            all_txt = torch.cat([f["txt_pooled"] for f in cached_features], dim=0).to(device)
            all_labels = torch.tensor([f["label"] for f in cached_features], device=device)
            q = F.normalize(router.query_adapter(torch.cat([all_vis.float(), all_txt.float()], dim=-1)), dim=-1)
            preds = (q @ expert_matrix.T).argmax(dim=-1)
            acc = (preds == all_labels).float().mean().item()

        log.info("epoch=%d  loss=%.4f  acc=%.1f%%  samples=%d", epoch, mean_loss, acc * 100, len(cached_features))

        if mean_loss < best_loss:
            best_loss = mean_loss
            router.save(output_dir / "best")

    router.save(output_dir / "final")
    log.info("Router training complete: best_loss=%.4f", best_loss)
    return best_loss

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
) -> float:
    """Train the router query adapter. Returns best loss.

    Args:
        backbone: Frozen VLM backbone.
        router: MoERouter instance with experts registered.
        expert_map: {name: subgoal} for registered experts.
        data_path: JSONL with screenshot_b64/task_text/expert_name.
        output_dir: Where to save router checkpoints.
        none_data_path: Optional JSONL for __none__ class.
        epochs: Training epochs.
        lr: Learning rate.
    """
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
        # Balance: downsample none to 2x the median expert count
        expert_counts = {}
        for r in records:
            en = r.get("expert_name", "")
            expert_counts[en] = expert_counts.get(en, 0) + 1
        if expert_counts:
            median_count = sorted(expert_counts.values())[len(expert_counts) // 2]
            max_none = median_count * 2
            if len(none_records) > max_none:
                import random
                random.seed(42)
                none_records = random.sample(none_records, max_none)
        records.extend(none_records)

    optimizer = torch.optim.AdamW(router.query_adapter.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        seen = 0

        for record in records:
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
                    img_inputs["pixel_values"],
                    img_inputs["image_grid_thw"],
                )
                input_ids, attn_mask = backbone.tokenize_subgoal(task_text)
                text_tokens = backbone.text_features(input_ids, attn_mask)

            expert_order, logits = router.logits_from_features(vis_tokens, text_tokens)
            if expert_order != expert_names:
                reorder = torch.tensor(
                    [expert_order.index(name) for name in expert_names if name in expert_order],
                    device=logits.device,
                )
                logits = logits[:, reorder]

            target = torch.tensor([label_map[expert_name]], device=logits.device)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            seen += 1

        mean_loss = total_loss / max(seen, 1)
        log.info("epoch=%d  loss=%.4f  samples=%d", epoch, mean_loss, seen)
        if mean_loss < best_loss:
            best_loss = mean_loss
            router.save(output_dir / "best")

    router.save(output_dir / "final")
    log.info("Router training complete: best_loss=%.4f", best_loss)
    return best_loss

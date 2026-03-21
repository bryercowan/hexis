"""VLMBackbone: frozen Qwen3-VL feature extractor (dense or MoE).

Exposes vision_features() and text_features() without trainable params.
The backbone stays in memory once, shared by all expert heads.
"""

from __future__ import annotations

import io
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 720, 1280
DEFAULT_BACKBONE = "Qwen/Qwen3-VL-4B-Instruct"


# ---------------------------------------------------------------------------
# Model class resolver
# ---------------------------------------------------------------------------

def _resolve_vl_model_class(model_name: str):
    """Pick the correct transformers class for the given model name."""
    name_lower = model_name.lower()
    if "qwen3" in name_lower and ("moe" in name_lower or "a3b" in name_lower):
        from transformers import Qwen3VLMoeForConditionalGeneration
        return Qwen3VLMoeForConditionalGeneration
    elif "qwen3" in name_lower:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# VLM Backbone (frozen)
# ---------------------------------------------------------------------------

class VLMBackbone(nn.Module):
    """Frozen Qwen3-VL feature extractor (dense or MoE).

    Exposes vision_features() and text_features() without trainable params.
    The backbone stays in memory once, shared by all expert heads.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "auto",
        max_pixels: int = 1280 * 720,
        min_pixels: int = 256 * 256,
    ):
        super().__init__()
        from transformers import AutoProcessor

        VLModelClass = _resolve_vl_model_class(model_name)
        log.info("Loading VLM backbone: %s (%s)", model_name, VLModelClass.__name__)

        # Resolve device — MPS doesn't support device_map
        self._target_device = device
        if device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
            elif torch.backends.mps.is_available():
                device_map = "cpu"
                self._target_device = "mps"
            else:
                device_map = "cpu"
        elif device == "mps":
            device_map = "cpu"
        else:
            device_map = device

        self.model = VLModelClass.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if self._target_device == "mps":
            log.info("Moving model to MPS...")
            self.model = self.model.to("mps")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # MoE models nest hidden_size under text_config
        cfg = self.model.config
        self.hidden_dim = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        self._torch_dtype = torch_dtype

        self.spatial_merge_size = getattr(
            self.model.config.vision_config, "spatial_merge_size", 2
        )

        # Resolve vision encoder and text embedder paths (differ between dense and MoE)
        if hasattr(self.model, "visual"):
            self._visual = self.model.visual
        else:
            self._visual = self.model.model.visual

        inner = self.model.model
        if hasattr(inner, "embed_tokens"):
            self._embed_tokens = inner.embed_tokens
        else:
            self._embed_tokens = inner.language_model.embed_tokens

        total_params = sum(p.numel() for p in self.model.parameters())
        log.info(
            "VLM backbone loaded: hidden_dim=%d, merge_size=%d, dtype=%s, params=%s",
            self.hidden_dim, self.spatial_merge_size, torch_dtype, f"{total_params:,}",
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def vision_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Extract spatial vision tokens from the VLM's vision encoder.

        Returns:
            vis_tokens: (B, N_vis, hidden_dim) spatial vision features.
            grid_h: spatial grid height after merger.
            grid_w: spatial grid width after merger.
        """
        vis_out = self._visual(
            pixel_values.to(dtype=self._torch_dtype, device=self.device),
            grid_thw=image_grid_thw.to(self.device),
        )
        if hasattr(vis_out, "pooler_output") and vis_out.pooler_output is not None:
            vis_tokens = vis_out.pooler_output
        elif hasattr(vis_out, "last_hidden_state"):
            vis_tokens = vis_out.last_hidden_state
        else:
            vis_tokens = vis_out

        t, h, w = image_grid_thw[0].tolist()
        m = self.spatial_merge_size
        grid_h = int(h) // m
        grid_w = int(w) // m

        if vis_tokens.dim() == 2:
            vis_tokens = vis_tokens.unsqueeze(0)

        return vis_tokens, grid_h, grid_w

    @torch.no_grad()
    def text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract text embeddings from the LLM's embedding layer."""
        return self._embed_tokens(input_ids.to(self.device))

    def preprocess_image(self, image) -> dict[str, torch.Tensor]:
        """Process a PIL Image or JPEG bytes into model inputs."""
        from PIL import Image as PILImage

        if isinstance(image, bytes):
            image = PILImage.open(io.BytesIO(image)).convert("RGB")
        if image.size != (IMG_W, IMG_H):
            image = image.resize((IMG_W, IMG_H), PILImage.BILINEAR)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "act"},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
        )

        return {
            "pixel_values": inputs["pixel_values"].to(self.device),
            "image_grid_thw": inputs["image_grid_thw"].to(self.device),
        }

    def tokenize_subgoal(self, subgoal: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize one or more conditioning strings for text_features()."""
        encoded = self.processor.tokenizer(
            subgoal,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )

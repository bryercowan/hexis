"""Image utility helpers: base64 encoding, PIL conversion, Anthropic content blocks."""

from __future__ import annotations

import base64
import io

from PIL import Image


def screenshot_to_content_block(screenshot_bytes: bytes) -> dict:
    """Convert screenshot bytes to Anthropic image content block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(screenshot_bytes).decode("ascii"),
        },
    }


def b64_to_content_block(b64_str: str) -> dict:
    """Convert base64 screenshot string to Anthropic image content block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": b64_str,
        },
    }


def bytes_to_b64(data: bytes) -> str:
    """Encode raw bytes to base64 ASCII string."""
    return base64.b64encode(data).decode("ascii")


def b64_to_bytes(b64_str: str) -> bytes:
    """Decode base64 ASCII string to raw bytes."""
    return base64.b64decode(b64_str)


def bytes_to_pil(data: bytes) -> Image.Image:
    """Convert JPEG/PNG bytes to PIL Image (RGB)."""
    return Image.open(io.BytesIO(data)).convert("RGB")

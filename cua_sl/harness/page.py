"""Async Playwright page adapter for benchmark automation.

Translates computer_20250124 tool actions into Playwright calls.
Pure vision — no DOM queries at inference time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from playwright.async_api import Page

log = logging.getLogger(__name__)

# Mapping from computer_20250124 key names to Playwright key names
KEY_MAP = {
    "Return": "Enter",
    "BackSpace": "Backspace",
    "space": " ",
    "Tab": "Tab",
    "Escape": "Escape",
    "Up": "ArrowUp",
    "Down": "ArrowDown",
    "Left": "ArrowLeft",
    "Right": "ArrowRight",
    "Home": "Home",
    "End": "End",
    "Page_Up": "PageUp",
    "Page_Down": "PageDown",
    "Delete": "Delete",
    "Super_L": "Meta",
    "ctrl+a": "Control+a",
    "ctrl+c": "Control+c",
    "ctrl+v": "Control+v",
    "ctrl+x": "Control+x",
    "ctrl+z": "Control+z",
}

# Settle delay after actions (ms)
SETTLE_MS = 150


class BenchmarkPage:
    """Async Playwright wrapper for benchmark page interaction.

    All coordinates are pixel values in the viewport coordinate space
    (default 1280x720).
    """

    def __init__(
        self,
        playwright_page: Page,
        viewport_w: int = 1280,
        viewport_h: int = 720,
    ) -> None:
        self._page = playwright_page
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h

    @property
    def url(self) -> str:
        return self._page.url

    @property
    def raw(self) -> Page:
        """Access the underlying Playwright page (for setup/navigation only)."""
        return self._page

    async def start_cdp_session(self):
        """Create CDP session for screencast."""
        return await self._page.context.new_cdp_session(self._page)

    async def screenshot(self, **kwargs) -> bytes:
        """Capture viewport as JPEG bytes."""
        kwargs.setdefault("type", "jpeg")
        kwargs.setdefault("quality", 80)
        kwargs.setdefault("full_page", False)
        return await self._page.screenshot(**kwargs)

    async def execute_computer_action(self, action_input: dict[str, Any]) -> None:
        """Dispatch a computer_20250124 tool action to Playwright.

        Handles: left_click, right_click, double_click, type, key,
                 scroll, mouse_move, wait.
        """
        action = action_input.get("action", "")
        x = action_input.get("coordinate", [None, None])[0] if action_input.get("coordinate") else None
        y = action_input.get("coordinate", [None, None])[1] if action_input.get("coordinate") else None

        if action == "left_click":
            await self.click_px(x, y)

        elif action == "right_click":
            if x is not None and y is not None:
                await self._page.mouse.click(x, y, button="right")
                await self._settle()

        elif action == "double_click":
            if x is not None and y is not None:
                await self._page.mouse.dblclick(x, y)
                await self._settle()

        elif action == "type":
            text = action_input.get("text", "")
            if text:
                await self._page.keyboard.type(text, delay=20)
                await self._settle()

        elif action == "key":
            key_str = action_input.get("text", "")
            await self._press_key(key_str)

        elif action == "scroll":
            direction = action_input.get("coordinate", [640, 360])
            scroll_x = direction[0] if direction else 640
            scroll_y = direction[1] if direction else 360
            scroll_dir = action_input.get("scroll_direction", "down")
            scroll_amount = action_input.get("scroll_amount", 3)
            delta = scroll_amount * 100
            if scroll_dir == "up":
                delta = -delta
            await self.scroll_px(scroll_x, scroll_y, delta)

        elif action == "mouse_move":
            if x is not None and y is not None:
                await self._page.mouse.move(x, y)

        elif action == "wait":
            await asyncio.sleep(1.0)

        elif action == "screenshot":
            pass  # No-op — screenshot is handled by the caller

        else:
            log.warning("Unknown action: %s", action)

    async def click_px(self, x: int | float, y: int | float) -> None:
        """Click at pixel coordinates."""
        await self._page.mouse.click(float(x), float(y))
        await self._settle()

    async def scroll_px(
        self, x: int | float, y: int | float, delta_y: int | float,
    ) -> None:
        """Scroll at pixel coordinates. Positive delta_y = scroll down."""
        await self._page.mouse.move(float(x), float(y))
        await self._page.mouse.wheel(0, float(delta_y))
        await self._settle()

    async def _press_key(self, key_str: str) -> None:
        """Press a key or key combo, mapping names as needed."""
        if " " in key_str and "+" not in key_str:
            for k in key_str.split():
                await self._press_key(k.strip())
            return
        mapped = KEY_MAP.get(key_str, key_str)
        try:
            return await self._press_key_inner(mapped)
        except Exception as e:
            log.warning("Key press failed for '%s' (mapped='%s'): %s", key_str, mapped, e)

    async def _press_key_inner(self, mapped: str) -> None:
        if "+" in mapped and len(mapped) > 1:
            parts = mapped.split("+")
            modifier_map = {
                "ctrl": "Control", "control": "Control",
                "alt": "Alt", "shift": "Shift",
                "meta": "Meta", "cmd": "Meta", "command": "Meta",
                "super": "Meta",
            }
            parts = [modifier_map.get(p.lower(), p) for p in parts]
            for p in parts[:-1]:
                await self._page.keyboard.down(p)
            await self._page.keyboard.press(parts[-1])
            for p in reversed(parts[:-1]):
                await self._page.keyboard.up(p)
        else:
            await self._page.keyboard.press(mapped)
        await self._settle()

    async def _settle(self) -> None:
        """Wait for UI to settle after an action."""
        await asyncio.sleep(SETTLE_MS / 1000)

"""Sliding window conversation manager for Claude API calls."""

from __future__ import annotations

# Default: keep first message + last N pairs
DEFAULT_WINDOW_PAIRS = 5


class SlidingWindow:
    """Manages conversation messages with sliding window trimming."""

    def __init__(self, max_pairs: int = DEFAULT_WINDOW_PAIRS) -> None:
        self.messages: list[dict] = []
        self.long_term_memory: list[str] = []
        self._max_pairs = max_pairs

    def append(self, msg: dict) -> None:
        self.messages.append(msg)
        self._trim()

    @property
    def context(self) -> list[dict]:
        """Return messages for API call, with memory prefix on first message."""
        if not self.messages:
            return []
        msgs = list(self.messages)
        if self.long_term_memory:
            memory_text = "PREVIOUS ACTIONS:\n" + "\n".join(
                f"- {m}" for m in self.long_term_memory[-20:]
            )
            first = msgs[0]
            if isinstance(first.get("content"), list):
                msgs[0] = {
                    **first,
                    "content": [{"type": "text", "text": memory_text}] + first["content"],
                }
            elif isinstance(first.get("content"), str):
                msgs[0] = {**first, "content": memory_text + "\n\n" + first["content"]}
        return msgs

    def _trim(self) -> None:
        """Trim to first message + last max_pairs assistant/user pairs."""
        if len(self.messages) <= 1 + self._max_pairs * 2:
            return

        keep_first = self.messages[0]
        keep_tail = self.messages[-(self._max_pairs * 2):]
        dropped = self.messages[1:-(self._max_pairs * 2)]

        for msg in dropped:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    summary = "; ".join(t[:100] for t in texts if t)
                elif isinstance(content, str):
                    summary = content[:150]
                else:
                    continue
                if summary:
                    self.long_term_memory.append(summary)

        self.messages = [keep_first] + keep_tail

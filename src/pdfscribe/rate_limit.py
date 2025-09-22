# src/pdfscribe/rate_limit.py
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Deque, Tuple, Literal

# Per-model iamge token costs (BASE COST, COST_PER_1K_TOKENS)
IMG_COSTS = {
    "gpt-4o-mini": (2833, 5667),
    "gpt-4o": (85, 170),
    "gpt-4.1": (85, 170),
    # fallback/default
    "*": (2833, 5667),
}

Detail = Literal["high", "low"]


def image_token_cost(
    width: int, height: int, model: str = "gpt-4o-mini", detail: Detail = "high"
) -> int:
    """
    Compute image tokens following OpenAI's spec:
      - if detail=low: base-only
      - if detail=high:
          1) scale to fit inside 2048x2048
          2) then scale so shortest side is 768
          3) tiles = ceil(w/512) * ceil(h/512)
          4) cost = base + tiles * tile

    NOTE: This project uses gpt-4o-mini model.
    REF: https://platform.openai.com/docs/guides/images-vision?api-mode=responses#calculating-costs
    """
    base, tile = IMG_COSTS.get(model, IMG_COSTS["*"])
    if detail == "low":
        return base

    # 1. Scale to fit in 2048 square
    w, h = float(width), float(height)
    if w > 2048 or h > 2048:
        s = min(2048 / w, 2048 / h)  # scale factor
        w *= s
        h *= s

    # 2. Scale to shortest side is 768
    short = min(w, h)
    if short > 0 and short != 768:
        s = 768 / short  # scale factor
        w *= s
        h *= s

    # 3. tiles
    tiles = ceil(w / 512) * ceil(h / 512)
    return int(base + tiles * tile)


def approx_text_tokens(s: str) -> int:
    """
    Cheap text-token estimator (safe-side): ~4 chars/token is standard lore.
    Use 3.6 to bias higher (safer).
    """
    if not s:
        return 0
    return int(len(s) / 3.6) + 1


@dataclass
class TokenLimiter:
    """
    Sliding 60s window token budgeter.
    - capacity: tokens per minute (TPM) that the user allows, and cap 80% of them.
    - window: deque of (timestamp, token_count) for the last 60s
    """

    capacity: int  # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window # The maximum number of tokens allowed in the window
    window_seconds: float = 60.0

    def __post_init__(self) -> None:
        self._events: Deque[Tuple[float, int]] = deque()
        self._sum: int = (
            0  # The running total of tokens spent within the current window.
        )

    def _gc(self, now: float) -> None:
        """
        Garbage collect old events outside the window.
        """
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            # pop old events
            _, tks = self._events.popleft()
            self._sum -= tks
            if self._sum < 0:
                self._sum = 0  # safety

    def can_spend(self, tokens: int, now: float | None = None) -> bool:
        """
        Can we spend these tokens now?
        Or... we should wait a bit?
        """
        now = now or time.monotonic()
        self._gc(now)  # clean up old events
        return (self._sum + max(0, tokens)) <= self.capacity

    def wait_for_budget(self, tokens: int) -> None:
        """
        Block until there is enough headroom to spend tokens.
        """
        tokens = max(0, tokens)
        while True:
            now = time.monotonic()
            self._gc(now)
            if (self._sum + tokens) <= self.capacity:
                # Enough budget, we can spend now!
                return

            # Sleep until earliest event falls out of the wnidow
            if not self._events:
                time.sleep(0.05)
                continue
            oldest_ts, _ = self._events[0]
            sleep_for = max(0.0, (oldest_ts + self.window_seconds) - now)
            time.sleep(min(0.25, sleep_for if sleep_for > 0 else 0.05))

    def spend(self, tokens: int, now: float | None = None) -> None:
        """
        Record spending these tokens now.
        """
        now = now or time.monotonic()
        tokens = max(0, tokens)
        self._events.append((now, tokens))
        self._sum += tokens

"""Picking which word to play next. No lookahead: refills are unpredictable."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .solver import Board, Candidate, Path

STRATEGIES = ("score", "rate", "keep-good-letters")
HARD_LETTERS = set("QZXJKV")
COMMON_LETTERS = set("EARST")


@dataclass
class Timing:
    """Rough cost of playing one word, used by the `rate` strategy."""
    segment_time: float = 0.06  # seconds per tile-to-tile move
    overhead: float = 0.45      # press/release + post-word delay + re-reading tiles

    def drag_time(self, path: Path) -> float:
        return (len(path) - 1) * self.segment_time + self.overhead


def letter_value(board: Board, path: Path) -> int:
    """+1 for each hard letter used up, -1 for each common letter consumed."""
    value = 0
    for cell in path:
        ch = board[cell][:1]  # "QU" counts as Q
        if ch in HARD_LETTERS:
            value += 1
        elif ch in COMMON_LETTERS:
            value -= 1
    return value


def choose(
    candidates: Sequence[Candidate],
    strategy: str = "score",
    board: Optional[Board] = None,
    timing: Optional[Timing] = None,
    window: float = 0.10,
) -> Optional[Candidate]:
    """Return the candidate to play, or None if there are none.

    Ties break toward the shorter path (faster to drag), then alphabetically,
    so the choice is deterministic.
    """
    if not candidates:
        return None
    if strategy == "score":
        return min(candidates, key=lambda c: (-c.score, len(c.path), c.word))
    if strategy == "rate":
        timing = timing or Timing()
        return min(candidates, key=lambda c: (-c.score / timing.drag_time(c.path), -c.score, c.word))
    if strategy == "keep-good-letters":
        if board is None:
            raise ValueError("keep-good-letters needs the board")
        top = max(c.score for c in candidates)
        pool: List[Candidate] = [c for c in candidates if c.score >= top * (1 - window)]
        return min(pool, key=lambda c: (-letter_value(board, c.path), -c.score, len(c.path), c.word))
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")

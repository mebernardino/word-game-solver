"""Remembering words the game refused, so later games skip them.

rejected.txt holds one word per line with how many separate games rejected it:

    QOPH 1
    ZA 1

A word is skipped once its count reaches the threshold (default 1). A word on
a line by itself is skipped right away. Delete a line to allow a word again.
"""
from __future__ import annotations

import os
from typing import Dict, List, Set

HEADER = (
    "# Words the game rejected, with how many games rejected them.\n"
    "# A word is skipped once its count reaches the --reject-after threshold.\n"
    "# Add a word on its own line to skip it right away; delete a line to allow it again.\n"
)


class RejectLog:
    def __init__(self, path: str, threshold: int = 1):
        self.path = path
        self.threshold = threshold
        self.counts: Dict[str, int] = {}
        self.pending: List[str] = []  # rejected this game, not yet confirmed
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                parts = line.split("#", 1)[0].split()
                if not parts or not parts[0].isalpha():
                    continue
                word = parts[0].upper()
                count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else self.threshold
                self.counts[word] = max(self.counts.get(word, 0), count)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(HEADER)
            for word in sorted(self.counts):
                f.write(f"{word} {self.counts[word]}\n")

    @property
    def blocked(self) -> Set[str]:
        if self.threshold <= 0:
            return set()
        return {w for w, n in self.counts.items() if n >= self.threshold}

    def suspect(self, word: str) -> None:
        """The game seemed to reject word. Held until a later word is accepted,
        which proves drags are registering (a mis-focused window would make
        every word look rejected)."""
        self.pending.append(word)

    def confirm_pending(self) -> List[str]:
        """A word was just accepted: the pending rejections were real. Returns
        words that are now blocked for future games."""
        newly_blocked = []
        for word in self.pending:
            self.counts[word] = self.counts.get(word, 0) + 1
            if self.threshold > 0 and self.counts[word] == self.threshold:
                newly_blocked.append(word)
        if self.pending:
            self.save()
        self.pending = []
        return newly_blocked

    def discard_pending(self) -> None:
        self.pending = []

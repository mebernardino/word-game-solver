"""Board model, dictionary trie, and DFS word finder.

Cells are (row, col) tuples with (0, 0) at the top-left. Internally the DFS
uses a flat index i = row * SIZE + col so "visited" can be a 16-bit mask.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

SIZE = 4
Cell = Tuple[int, int]
Path = Tuple[Cell, ...]

DEFAULT_SCORES: Dict[int, int] = {3: 100, 4: 400, 5: 800, 6: 1400, 7: 1800, 8: 2200}
# 16 tiles, each at most two letters ("QU"), so no formable word is longer than 32.
MAX_WORD_LEN = SIZE * SIZE * 2


def _build_neighbors() -> List[List[int]]:
    """For each flat index, the flat indices of its (up to 8) adjacent tiles."""
    out = []
    for i in range(SIZE * SIZE):
        r, c = divmod(i, SIZE)
        out.append([
            nr * SIZE + nc
            for nr in range(r - 1, r + 2)
            for nc in range(c - 1, c + 2)
            if (nr, nc) != (r, c) and 0 <= nr < SIZE and 0 <= nc < SIZE
        ])
    return out


NEIGHBORS = _build_neighbors()


# --------------------------------------------------------------------------- scoring

def parse_scores(spec: str) -> Dict[int, int]:
    """Parse "3:100,4:400,8:2200" into {3: 100, 4: 400, 8: 2200}."""
    table = {}
    for part in spec.split(","):
        length, points = part.split(":")
        table[int(length)] = int(points)
    return table


def score_word(word: str, table: Dict[int, int] = DEFAULT_SCORES) -> int:
    """Score by letter count. Lengths past the largest key use that key's score."""
    n = len(word)
    if n in table:
        return table[n]
    top = max(table)
    return table[top] if n > top else 0


# --------------------------------------------------------------------------- board

UNKNOWN = "?"  # a tile OCR couldn't read; matches nothing, so words avoid it
QU = "QU"      # the two-letter "Qu" tile. A plain "Q" tile is just Q.


def _valid_tile(tile: str) -> bool:
    return tile in (UNKNOWN, QU) or (len(tile) == 1 and "A" <= tile <= "Z")


def normalize_tile(text: str) -> Optional[str]:
    """User/OCR text for one tile -> "A".."Z" or "QU", or None if invalid."""
    tile = text.strip().upper()
    return tile if tile != UNKNOWN and _valid_tile(tile) else None


def split_tiles(text: str, count: int) -> List[str]:
    """Split typed letters into `count` tiles, where "Qu" may be one tile.

    "QU" is ambiguous (a Qu tile, or a Q tile then a U tile), so pairs are only
    merged as needed to reach `count`, preferring ones typed as "Qu".
    """
    letters = [ch for ch in text if ch.isalpha()]
    extra = len(letters) - count
    if extra < 0:
        raise ValueError(f"need {count} tiles, got {len(letters)} letters")
    pairs = [i for i in range(len(letters) - 1)
             if letters[i].upper() == "Q" and letters[i + 1].upper() == "U"]
    pairs.sort(key=lambda i: (letters[i + 1] != "u", i))  # "Qu" first, then left to right
    merge = set()
    for i in pairs:
        if len(merge) == extra:
            break
        if i - 1 not in merge and i + 1 not in merge:
            merge.add(i)
    if len(merge) != extra:
        raise ValueError(f"need {count} tiles, got {len(letters)} letters")
    tiles, i = [], 0
    while i < len(letters):
        if i in merge:
            tiles.append(QU)
            i += 2
        else:
            tiles.append(letters[i].upper())
            i += 1
    return tiles


class Board:
    """A mutable 4x4 grid. Each tile is one letter "A".."Z", the two-letter
    Qu tile "QU", or UNKNOWN."""

    def __init__(self, tiles: Iterable[str]):
        flat = [t.upper() for t in tiles]
        if len(flat) != SIZE * SIZE or not all(_valid_tile(t) for t in flat):
            raise ValueError(f"need {SIZE * SIZE} tiles (A-Z or Qu), got {flat!r}")
        self.letters = flat

    @classmethod
    def parse(cls, text: str) -> "Board":
        """Accept "CATS ODOG RENT SXQZ", "catsodogrentsxqz", or with Qu tiles,
        e.g. "QuIT ..." (5 letters in a 4-tile row means one of them is Qu)."""
        groups = text.split()
        if len(groups) == SIZE:
            return cls(t for g in groups for t in split_tiles(g, SIZE))
        return cls(split_tiles("".join(groups), SIZE * SIZE))

    def copy(self) -> "Board":
        return Board(self.letters)

    def __getitem__(self, cell: Cell) -> str:
        r, c = cell
        return self.letters[r * SIZE + c]

    def __setitem__(self, cell: Cell, tile: str) -> None:
        tile = tile.upper()
        if not _valid_tile(tile):
            raise ValueError(f"bad tile {tile!r}")
        r, c = cell
        self.letters[r * SIZE + c] = tile

    def tile_text(self, i: int) -> str:
        """What tile i (flat index) contributes to a word."""
        return self.letters[i]

    def unknown_cells(self) -> List[Cell]:
        return [divmod(i, SIZE) for i, t in enumerate(self.letters) if t == UNKNOWN]

    def diff(self, other: "Board") -> List[Cell]:
        return [divmod(i, SIZE) for i in range(SIZE * SIZE) if self.letters[i] != other.letters[i]]

    def __str__(self) -> str:
        rows = []
        for r in range(SIZE):
            cells = [self.tile_text(r * SIZE + c).capitalize().ljust(2) for c in range(SIZE)]
            rows.append("  ".join(cells))
        return "\n".join(rows)


# --------------------------------------------------------------------------- trie

class TrieNode:
    __slots__ = ("children", "word")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.word: Optional[str] = None  # set on the node that ends a dictionary word


class Trie:
    """Prefix tree over the dictionary. The DFS walks it in lockstep with the board,
    so any board path whose letters aren't a prefix of some word is cut off at once."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self.size = 0

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        if node.word is None:
            node.word = word
            self.size += 1

    @classmethod
    def from_words(cls, words: Iterable[str], min_len: int = 3) -> "Trie":
        trie = cls()
        for w in words:
            w = w.strip().upper()
            if not (min_len <= len(w) <= MAX_WORD_LEN and w.isascii() and w.isalpha()):
                continue
            trie.insert(w)
        return trie


def load_trie(path: str, min_len: int = 3) -> Trie:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return Trie.from_words(f, min_len)


# --------------------------------------------------------------------------- solving

@dataclass(frozen=True)
class Candidate:
    word: str
    path: Path
    score: int


class Solver:
    def __init__(self, trie: Trie, min_len: int = 3, scores: Optional[Dict[int, int]] = None):
        self.trie = trie
        self.min_len = min_len
        self.scores = scores or DEFAULT_SCORES

    def solve(self, board: Board) -> Dict[str, Path]:
        """Return {word: one valid path} for every dictionary word on the board."""
        found: Dict[str, Path] = {}
        texts = [board.tile_text(i) for i in range(SIZE * SIZE)]
        min_len = self.min_len

        def dfs(i: int, node: TrieNode, visited: int, path: List[int]) -> None:
            # Step the trie through this tile's text ("QU" is two steps).
            for ch in texts[i]:
                node = node.children.get(ch)
                if node is None:
                    return  # no word starts with this prefix: prune
            path.append(i)
            if node.word is not None and len(node.word) >= min_len and node.word not in found:
                found[node.word] = tuple(divmod(j, SIZE) for j in path)
            if node.children:
                visited |= 1 << i  # bit i set = tile i already used in this word
                for j in NEIGHBORS[i]:
                    if not (visited >> j) & 1:
                        dfs(j, node, visited, path)
            path.pop()

        for start in range(SIZE * SIZE):
            dfs(start, self.trie.root, 0, [])
        return found


class CandidateSet:
    """The words currently on the board, minus ones excluded this game (mark_played)
    or blocked (e.g. rejected in past games)."""

    def __init__(self, solver: Solver, blocked: Iterable[str] = ()):
        self.solver = solver
        self.paths: Dict[str, Path] = {}
        self.played: Set[str] = set()
        self.blocked: Set[str] = set(blocked)  # e.g. words the game rejected in past games
        self.last_solve_ms = 0.0

    def refresh(self, board: Board, changed: Optional[Iterable[Cell]] = None) -> None:
        """Re-solve after tiles change.

        A full DFS takes only milliseconds on a 4x4 board, so we always run one
        (it finds words through the new tiles, and words that survive via a
        different path). For words whose stored path avoids every changed tile
        the old path is still valid, so we keep it rather than swapping in
        whichever path the new DFS happened to find first.
        """
        t0 = time.perf_counter()
        fresh = self.solver.solve(board)
        if changed is not None:
            changed = set(changed)
            for word, old_path in self.paths.items():
                if word in fresh and changed.isdisjoint(old_path):
                    fresh[word] = old_path
        self.paths = fresh
        self.last_solve_ms = (time.perf_counter() - t0) * 1000

    def mark_played(self, word: str) -> None:
        self.played.add(word)

    def available(self) -> List[Candidate]:
        return [
            Candidate(w, p, score_word(w, self.solver.scores))
            for w, p in self.paths.items()
            if w not in self.played and w not in self.blocked
        ]

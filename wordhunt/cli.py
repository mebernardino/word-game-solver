"""Command-line entry point and the main play loop."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

from .solver import (
    DEFAULT_SCORES, SIZE, UNKNOWN, Board, CandidateSet, Cell, Solver, load_trie, normalize_tile,
    parse_scores, split_tiles,
)
from .dragger import Aborted
from .strategy import STRATEGIES, Timing, choose

DICT_HELP = (
    "Plain-text word list, one word per line. ENABLE is a good free choice:\n"
    "  curl -O https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wordhunt",
        description="Auto-play a 4x4 word-search puzzle on screen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("setup")
    g.add_argument("--dict", metavar="PATH", help=DICT_HELP)
    g.add_argument("--calibrate", action="store_true", help="redo screen calibration")
    g.add_argument("--advanced", action="store_true", help="calibrate all 16 tiles individually")
    g.add_argument("--calibration", default="calibration.json", metavar="PATH")
    g.add_argument("--box", type=float, default=0.7, metavar="FRAC",
                   help="size of the area read on each tile, as a fraction of tile spacing (default 0.7)")
    g.add_argument("--calib-key", default="f8")
    g.add_argument("--abort-key", default="esc")
    g.add_argument("--pause-key", default="f9")

    g = p.add_argument_group("reading the board")
    g.add_argument("--manual", action="store_true", help="type letters instead of using OCR")
    g.add_argument("--min-conf", type=float, default=60, help="OCR confidence (0-100) below which a tile is re-shot once")
    g.add_argument("--retry-delay", type=float, default=0.25, help="wait before re-shooting an unclear tile")
    g.add_argument("--timing", default="timing.json", metavar="PATH",
                   help="where the learned animation timing is remembered between games")
    g.add_argument("--settle-max", type=float, default=1.0, metavar="S",
                   help="after the post-word delay, keep re-reading up to S seconds until tiles stop moving")
    g.add_argument("--rejected", default="rejected.txt", metavar="PATH",
                   help="file of words the game rejected; skipped in later games")
    g.add_argument("--reject-after", type=int, default=1, metavar="N",
                   help="skip a word once N separate games rejected it (default 1; 0 = never skip)")
    g.add_argument("--tesseract-cmd", metavar="PATH", help="path to the tesseract binary if not on PATH")
    g.add_argument("--save-tiles", metavar="DIR", help="save raw and preprocessed tile images for debugging")
    g.add_argument("--glyphs", default="glyphs", metavar="DIR",
                   help="letter templates folder; boards you confirm are added to it (default glyphs)")
    g.add_argument("--no-learn", action="store_true", help="don't add confirmed tiles to --glyphs")
    g.add_argument("--font", metavar="PATH", help="font file to render letter templates from")

    g = p.add_argument_group("solving")
    g.add_argument("--min-len", type=int, default=3)
    g.add_argument("--scores", type=parse_scores, default=DEFAULT_SCORES, metavar="SPEC",
                   help='e.g. "3:100,4:400,5:800,6:1400,7:1800,8:2200" (longer words use the last entry)')
    g.add_argument("--strategy", choices=STRATEGIES, default="score")
    g.add_argument("--keep-window", type=float, default=0.10,
                   help="keep-good-letters: consider words within this fraction of the top score")

    g = p.add_argument_group("playing")
    g.add_argument("--segment-time", type=float, default=0.11, help="seconds per tile-to-tile move")
    g.add_argument("--press-time", type=float, default=0.025,
                   help="seconds to hold at the first and last tile so the game registers them")
    g.add_argument("--drag-steps", type=int, default=7,
                   help="mouse events per tile-to-tile move (more = smoother but slower)")
    g.add_argument("--post-word-delay", type=float, default=0.62,
                   help="wait after a word for the tile-replacement animation (it takes ~0.62s)")
    g.add_argument("--countdown", type=int, default=3, help="seconds to focus the game window")
    g.add_argument("--max-words", type=int, default=0, help="stop after N words (0 = no limit)")
    g.add_argument("--time-limit", type=float, default=0, help="stop after S seconds (0 = no limit)")
    g.add_argument("--dry-run", action="store_true", help="hover the path without pressing the mouse")
    g.add_argument("--no-repeats", action="store_true",
                   help="never play a word twice in one game (by default accepted words may be replayed)")
    return p


# --------------------------------------------------------------------------- console helpers

def show_board(board: Board) -> None:
    print("       " + "   ".join(str(c + 1) for c in range(SIZE)))
    for r in range(SIZE):
        cells = [board.tile_text(r * SIZE + c).capitalize().ljust(2) for c in range(SIZE)]
        print(f"    {r + 1}  " + "  ".join(cells))


def edit_board(board: Board) -> Board:
    """Let the user confirm the board or correct it. Returns the (possibly new) board."""
    board = board.copy()
    while True:
        show_board(board)
        s = input("  Enter = looks right | 16 letters = replace all | 'row col letter' = fix one "
                  "(Q and Qu are different tiles): ").strip()
        if not s:
            return board
        parts = s.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            r, c = int(parts[0]) - 1, int(parts[1]) - 1
            letter = normalize_tile(parts[2])
            if 0 <= r < SIZE and 0 <= c < SIZE and letter:
                board[(r, c)] = letter
            else:
                print("  Use row and col 1-4 and a single letter (or Qu).")
            continue
        try:
            board = Board.parse(s)
        except ValueError:
            print("  Couldn't parse that: expected 16 tiles, e.g. CATS ODOG RENT SXQZ (QuIT = Qu, I, T)")


def type_board() -> Board:
    while True:
        s = input("Type the 16 tiles row by row (e.g. CATS ODOG RENT SXQZ; write a Qu tile as Qu): ")
        try:
            return edit_board(Board.parse(s))
        except ValueError:
            print("  Need exactly 16 tiles.")


def load_anim_start(path: str) -> Optional[float]:
    """How soon after a word tiles start moving, as learned in earlier games."""
    try:
        with open(path) as f:
            return float(json.load(f)["animation_start"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_anim_start(path: str, seconds: float) -> None:
    try:
        with open(path, "w") as f:
            json.dump({"animation_start": round(seconds, 3)}, f)
    except OSError:
        pass


def fmt_path(path) -> str:
    return " ".join(f"{r + 1}{c + 1}" for r, c in path)


def sleep_unless(event: threading.Event, seconds: float) -> bool:
    """Sleep, waking early if event is set. Returns True if it was set."""
    return event.wait(seconds)


# --------------------------------------------------------------------------- game

class Game:
    REJECT_WARNING = 5  # warn (once per streak) after this many rejections in a row
    POLL_INTERVAL = 0.03  # seconds between looks at a word's tiles while waiting
    REJECT_MARGIN = 0.12  # extra wait past the latest "tiles started moving" time seen
    SETTLE_GAP = 0.03     # seconds between board reads while waiting for tiles to land

    def __init__(self, args, solver: Solver, reader, dragger, abort, pause, rejects=None):
        self.args = args
        self.rejects = rejects  # RejectLog, or None to not track rejections across games
        self.cands = CandidateSet(solver, rejects.blocked if rejects else ())
        self.reader = reader  # None in --manual mode
        self.dragger = dragger
        self.abort = abort
        self.pause = pause
        self.timing = Timing(args.segment_time, overhead=args.post_word_delay + 0.15)
        self.board: Optional[Board] = None
        self.words = 0      # accepted words
        self.attempts = 0   # words dragged, accepted or not
        self.rejected = 0
        self.reject_streak = 0
        self.total = 0
        # Seconds after release at which an accepted word's tiles first visibly
        # changed (the latest seen). Once known, a word whose tiles still show
        # their old letters a bit after that was rejected: no need to wait longer.
        # Remembered between games in --timing.
        self.anim_start: Optional[float] = load_anim_start(args.timing)
        self.accepted_words: set = set()
        # Replaying accepted words is allowed until the game shows it doesn't
        # accept repeats (or --no-repeats / --dry-run, where the board never changes).
        self.repeats = not (args.no_repeats or args.dry_run)

    # ---- reading

    def read_cells(self, cells: Iterable[Cell]) -> Dict[Cell, str]:
        """OCR some tiles, retrying unclear ones once. Never stops to ask: a
        low-confidence guess is used as-is, and a blank read becomes UNKNOWN
        (words avoid it, and it's re-read after the next word)."""
        cells = list(cells)
        reads = self.reader.read_cells(cells)
        unclear = [c for c in cells if reads[c].confidence < self.args.min_conf]
        if unclear:
            time.sleep(self.args.retry_delay)  # animation may not have finished
            retry = self.reader.read_cells(unclear)
            for c in unclear:
                if retry[c].confidence > reads[c].confidence:
                    reads[c] = retry[c]
        blank = [c for c in cells if reads[c].letter == UNKNOWN]
        if blank:
            print(f"  couldn't read tile(s) {fmt_path(blank)}; avoiding them for now")
        return {c: reads[c].letter for c in cells}

    def read_full_board(self) -> Board:
        cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        letters = self.read_cells(cells)
        return Board(letters[c] for c in cells)

    def initial_board(self) -> Board:
        if self.reader is None:
            return type_board()
        # Read only once the game is in front: anything covering a tile
        # (like this terminal) would be read instead of the letter.
        if not self.countdown("Click the game window; reading the board in"):
            raise Aborted()
        board = self.read_full_board()
        print("Detected (unreadable tiles show as ?):")
        board = edit_board(board)
        self.learn(board)
        return board

    def learn(self, board: Board) -> None:
        """Teach the letter matcher from a board the user just confirmed."""
        if self.reader is not None and not self.args.no_learn:
            added = self.reader.learn(board)
            if added:
                print(f"  Learned {added} new letter template(s).")

    def wait_for_animation(self, used: List[Cell], released: float) -> Tuple[bool, Optional[float]]:
        """Wait out the --post-word-delay, watching the word's tiles.

        Returns (rejected, seconds after release when the tiles first changed).
        On an accepted word the tiles vanish almost at once. So a word is
        rejected if its tiles still show their old letters a little after the
        latest start time seen on accepted words, or if they never changed at
        all during the whole delay.
        """
        before = {c: self.board[c] for c in used}
        done = released + self.args.post_word_delay
        give_up = released + self.anim_start + self.REJECT_MARGIN if self.anim_start is not None else None
        while time.monotonic() < done:
            reads = {c: r.letter for c, r in self.reader.read_cells(used).items()}
            if reads != before:  # a tile vanished, moved, or changed: animation under way
                started = time.monotonic() - released
                sleep_unless(self.abort, done - time.monotonic())
                return False, started
            if give_up is not None and time.monotonic() >= give_up:
                return True, None
            if sleep_unless(self.abort, min(self.POLL_INTERVAL, max(0.0, done - time.monotonic()))):
                break
        return True, None  # nothing moved during the whole delay

    def read_settled(self, deadline: float) -> Dict[Cell, str]:
        """Read the whole board until it has stopped moving.

        In this game used tiles disappear and the tiles above drop down their
        column, so some can still be falling when the delay ends: a read then
        sees an empty spot ("?") or a tile part-way down. The board counts as
        settled when two reads in a row agree and there are no more "?" than
        before the word. Gives up at the deadline and returns the last read.
        """
        cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        allowed_unknown = len(self.board.unknown_cells())

        def read():
            return {c: r.letter for c, r in self.reader.read_cells(cells).items()}

        prev = read()
        while time.monotonic() < deadline:
            if sleep_unless(self.abort, self.SETTLE_GAP):
                break
            cur = read()
            if cur == prev and list(cur.values()).count(UNKNOWN) <= allowed_unknown:
                return cur
            prev = cur
        return prev

    @staticmethod
    def falling_zone(used: Iterable[Cell]) -> List[Cell]:
        """Tiles expected to change after a word: in each column the word used,
        every tile from the top down to its lowest used tile (they drop down)."""
        lowest: Dict[int, int] = {}
        for r, c in used:
            lowest[c] = max(lowest.get(c, -1), r)
        return [(r, c) for c, bottom in lowest.items() for r in range(bottom + 1)]

    def update_after_word(self, path, released: Optional[float] = None) -> Tuple[List[Cell], bool]:
        """Wait for the board to settle after a word, then re-read all of it.

        Returns (cells whose letters may have changed, whether the game accepted
        the word). If nothing where the word was has changed, it was rejected.
        """
        used = list(dict.fromkeys(path))
        if self.reader is None:
            print("  Type the new board, or press Enter if the word was rejected.")
            s = input("  ").strip()
            if not s:
                return used, False
            try:
                new = Board.parse(s)
            except ValueError:
                new = edit_board(self.board)
            changed = self.board.diff(new)
            self.board = new
            return changed, True

        released = time.monotonic() if released is None else released
        rejected, started = self.wait_for_animation(used, released)
        zone = self.falling_zone(used)
        everything = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        if rejected:
            letters = {c: r.letter for c, r in self.reader.read_cells(everything).items()}
            if any(letters[c] != self.board[c] for c in zone):
                # Safety net: it moved after all (a late start). Treat as accepted.
                letters = self.read_settled(time.monotonic() + self.args.post_word_delay
                                            + self.args.settle_max)
        else:
            letters = self.read_settled(released + self.args.post_word_delay + self.args.settle_max)
        accepted = any(letters[c] != self.board[c] for c in zone)
        if accepted and started is not None and (self.anim_start is None or started > self.anim_start):
            self.anim_start = started
            save_anim_start(self.args.timing, started)

        # Tiles outside the falling zone shouldn't change; if they read differently
        # now, an earlier read was wrong and this corrects it.
        fixed = [c for c in letters if c not in zone and letters[c] != self.board[c]
                 and self.board[c] != UNKNOWN]
        if fixed:
            print(f"  Re-read corrected {len(fixed)} tile(s): "
                  + ", ".join(f"{fmt_path([c])} {self.board[c].capitalize()}->{letters[c].capitalize()}"
                              for c in fixed))
        blank = [c for c in letters if letters[c] == UNKNOWN]
        if blank:
            print(f"  couldn't read tile(s) {fmt_path(blank)}; avoiding them for now")
        changed = sorted(set(zone) | {c for c in letters if letters[c] != self.board[c]})
        for cell, tile in letters.items():
            self.board[cell] = tile
        return changed, accepted

    # ---- loop

    def countdown(self, message: str = "Click the game window; starting in") -> bool:
        for n in range(self.args.countdown, 0, -1):
            print(f"  {message} {n}...   ", end="\r", flush=True)
            if sleep_unless(self.abort, 1):
                return False
        print(" " * 50, end="\r")
        return True

    def handle_pause(self) -> None:
        print("\nPaused. Correct any tiles, then press Enter to resume.")
        old = self.board
        self.board = edit_board(self.board)
        self.learn(self.board)
        self.cands.refresh(self.board, changed=old.diff(self.board))
        self.pause.clear()
        self.countdown()

    def run(self) -> None:
        a = self.args
        self.board = self.initial_board()
        self.cands.refresh(self.board)
        print(f"{len(self.cands.paths)} words on the board (solved in {self.cands.last_solve_ms:.1f} ms)")
        print(f"Hotkeys: {a.abort_key.upper()} = stop, {a.pause_key.upper()} = pause & correct. "
              "Or slam the mouse into a screen corner.")
        if not self.countdown():
            return
        start = time.monotonic()
        while not self.abort.is_set():
            elapsed = time.monotonic() - start
            if a.time_limit and elapsed >= a.time_limit:
                print("Time limit reached.")
                break
            if a.max_words and self.words >= a.max_words:
                print("Max words reached.")
                break
            if self.pause.is_set():
                self.handle_pause()
                continue

            pick = choose(self.cands.available(), a.strategy, self.board, self.timing, a.keep_window)
            if pick is None and self.reader is not None and not a.dry_run:
                # A misread tile could be hiding words: re-read everything once.
                new = self.read_full_board()
                if new.letters != self.board.letters:
                    print("  No words left; full re-read changed the board, retrying.")
                    changed = self.board.diff(new)
                    self.board = new
                    self.cands.refresh(self.board, changed=changed)
                    continue
            if pick is None:
                print("No playable words left on this board.")
                break

            self.dragger.drag(pick.path)
            self.attempts += 1
            stats = f"path {fmt_path(pick.path):<24} [{elapsed:5.1f}s, solve {self.cands.last_solve_ms:.1f} ms]"

            if a.dry_run:
                self.cands.mark_played(pick.word)
                self.words += 1
                self.total += pick.score
                print(f"#{self.attempts:<3} {pick.word:<12} +{pick.score:<5} total {self.total:<6} {stats}")
                self.cands.refresh(self.board, changed=[])  # nothing really changed
                continue
            released = time.monotonic()
            changed, accepted = self.update_after_word(pick.path, released)
            if self.abort.is_set():
                break
            self.cands.refresh(self.board, changed=changed)

            if accepted:
                self.words += 1
                self.total += pick.score
                self.reject_streak = 0
                repeat = " (repeat)" if pick.word in self.accepted_words else ""
                self.accepted_words.add(pick.word)
                if not self.repeats:
                    self.cands.mark_played(pick.word)
                print(f"#{self.attempts:<3} {pick.word:<12} +{pick.score:<5} total {self.total:<6} {stats}{repeat}")
                if self.rejects:
                    newly = self.rejects.confirm_pending()
                    if newly:
                        print(f"  Will skip in future games: {', '.join(newly)} (see {a.rejected})")
            else:
                self.rejected += 1
                self.reject_streak += 1
                self.cands.mark_played(pick.word)  # don't retry it this game
                if pick.word in self.accepted_words:
                    # The game took this word before, so it's valid: it refused the
                    # repeat. Don't record it, and stop replaying words this game.
                    print(f"#{self.attempts:<3} {pick.word:<12} rejected as a repeat; "
                          "no more repeats this game")
                    self.repeats = False
                    for word in self.accepted_words:
                        self.cands.mark_played(word)
                else:
                    print(f"#{self.attempts:<3} {pick.word:<12} rejected (not counted)")
                    if self.rejects:
                        self.rejects.suspect(pick.word)
                if self.reject_streak == self.REJECT_WARNING:
                    print(f"  ({self.reject_streak} words in a row didn't change the board. Is the game "
                          "window in front, and are the drags landing on the tiles?)")

        if self.rejects:
            # Rejections not followed by an accepted word can't be trusted.
            self.rejects.discard_pending()
        extra = f" ({self.rejected} rejected)" if self.rejected else ""
        print(f"\nDone: {self.words} words{extra}, {self.total} points.")


# --------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from .calibration import Calibration, CalibrationAborted, build_regions, preview, run_calibration

    if sys.platform == "darwin":
        print("macOS: your terminal app needs Accessibility, Input Monitoring and Screen Recording "
              "permissions (System Settings > Privacy & Security).")

    # -- calibration
    if args.calibrate or not os.path.exists(args.calibration):
        if not args.calibrate:
            print(f"No {args.calibration} found, so let's calibrate first.")
        try:
            calib = run_calibration(args.advanced, args.calib_key, args.abort_key, args.box)
        except CalibrationAborted:
            print("Calibration cancelled.")
            return 1
        calib.save(args.calibration)
        print(f"Saved {args.calibration}")
        preview(calib)
        if not args.dict:
            return 0
    else:
        calib = Calibration.load(args.calibration)
        print(f"Loaded {args.calibration} ({calib.mode} mode)")
    calib.regions = build_regions(calib.centers, args.box)

    if not args.dict:
        print("--dict is required to play.\n" + DICT_HELP)
        return 2

    # -- dictionary
    t0 = time.perf_counter()
    trie = load_trie(args.dict, args.min_len)
    print(f"Loaded {trie.size:,} words in {time.perf_counter() - t0:.2f}s")
    solver = Solver(trie, args.min_len, args.scores)

    # -- devices
    import pyautogui
    from .dragger import Dragger
    from .hotkeys import Hotkeys

    abort, pause = threading.Event(), threading.Event()
    reader = None
    if not args.manual:
        from .ocr import TileReader, default_matcher
        from .screen import detect_scales

        scales = detect_scales()
        print(f"Display scaling: {scales}")
        reader = TileReader(calib, scales.coord_scale, args.tesseract_cmd, args.save_tiles,
                            matcher=default_matcher(args.glyphs, args.font),
                            glyph_dir=None if args.no_learn else args.glyphs)
    dragger = Dragger(calib.centers, args.segment_time, args.press_time, args.drag_steps,
                      dry_run=args.dry_run, abort=abort)
    if args.dry_run:
        print("DRY RUN: the mouse will hover over each word without clicking.")

    from .rejects import RejectLog

    rejects = RejectLog(args.rejected, args.reject_after)
    if rejects.blocked:
        print(f"Skipping {len(rejects.blocked)} word(s) the game rejected before ({args.rejected})")
    game = Game(args, solver, reader, dragger, abort, pause, rejects)
    with Hotkeys({args.abort_key: abort.set, args.pause_key: pause.set}):
        try:
            game.run()
        except Aborted:
            print(f"\nStopped by {args.abort_key.upper()}. {game.words} words, {game.total} points.")
        except pyautogui.FailSafeException:
            print(f"\nFail-safe triggered (mouse in a corner). {game.words} words, {game.total} points.")
        except (KeyboardInterrupt, EOFError):
            print(f"\nInterrupted. {game.words} words, {game.total} points.")
    return 0

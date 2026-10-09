"""Entering a word by dragging the mouse through its tiles."""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from .calibration import Point
from .solver import Path


class Aborted(Exception):
    pass


class Dragger:
    def __init__(
        self,
        centers: List[List[Point]],
        segment_time: float = 0.11,
        press_time: float = 0.025,
        steps: int = 7,
        dry_run: bool = False,
        abort: Optional[threading.Event] = None,
        pag=None,  # pyautogui, or a stand-in for tests
    ):
        if pag is None:
            import pyautogui as pag
        self.pag = pag
        pag.FAILSAFE = True   # slam the mouse into a screen corner to abort
        pag.PAUSE = 0         # no implicit sleep after every call; we time things ourselves
        # On macOS pyautogui sleeps DARWIN_CATCH_UP_TIME (0.01s) after every mouse
        # event. Its own animated moves send ~6 events per step, so a "0.06s" step
        # really took ~0.13s. We send a few events ourselves with a much shorter pause.
        pag.DARWIN_CATCH_UP_TIME = 0.002
        self.centers = centers
        self.segment_time = segment_time
        self.press_time = press_time
        self.steps = max(1, steps)
        self.dry_run = dry_run
        self.abort = abort or threading.Event()

    def _check_abort(self) -> None:
        if self.abort.is_set():
            raise Aborted()

    def _glide(self, start: Point, end: Point, move) -> None:
        """Move from start to end in `steps` jumps spread over segment_time.

        With 3 steps the points land 1/3 and 2/3 of the way across: each one is
        well inside either the tile we're leaving or the one we're entering (a
        tile is ~0.9 of the spacing wide), so a diagonal never grazes the corner
        between tiles the way a smooth line through it would.
        """
        (x0, y0), (x1, y1) = start, end
        t0 = time.perf_counter()
        for i in range(1, self.steps + 1):
            self._check_abort()
            f = i / self.steps
            move(round(x0 + (x1 - x0) * f), round(y0 + (y1 - y0) * f))
            # Pace the jumps evenly; skip sleeping if the event itself took longer.
            ahead = t0 + self.segment_time * f - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)

    def drag(self, path: Path) -> None:
        pts = [self.centers[r][c] for r, c in path]
        self._check_abort()
        self.pag.moveTo(round(pts[0][0]), round(pts[0][1]))

        if self.dry_run:
            # Slow enough to watch which tiles it would visit.
            for a, b in zip(pts, pts[1:]):
                self._check_abort()
                self.pag.moveTo(round(b[0]), round(b[1]), duration=0.15)
            return

        self.pag.mouseDown()
        try:
            time.sleep(self.press_time)  # let the game register the touch-down
            for a, b in zip(pts, pts[1:]):
                # dragTo (not moveTo) so macOS gets "mouse dragged" events with the
                # button held; duration=0 sends a single event per call.
                self._glide(a, b, lambda x, y: self.pag.dragTo(
                    x, y, duration=0, button="left", mouseDownUp=False))
            time.sleep(self.press_time)  # let it register the last tile before release
        finally:
            # Always release, even on abort / fail-safe. mouseUp itself would raise
            # FailSafeException with the cursor in a corner, so suspend that check.
            failsafe, self.pag.FAILSAFE = self.pag.FAILSAFE, False
            try:
                self.pag.mouseUp()
            finally:
                self.pag.FAILSAFE = failsafe

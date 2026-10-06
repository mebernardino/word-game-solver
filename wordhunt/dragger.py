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
        segment_time: float = 0.06,
        press_time: float = 0.03,
        dry_run: bool = False,
        abort: Optional[threading.Event] = None,
    ):
        import pyautogui

        self.pag = pyautogui
        pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
        pyautogui.PAUSE = 0         # no implicit sleep after every call; we time things ourselves
        # pyautogui teleports instead of animating moves shorter than MINIMUM_DURATION
        # (default 0.1s), and caps the number of intermediate steps via MINIMUM_SLEEP.
        # Lower both so short segments still produce a smooth stream of drag events.
        pyautogui.MINIMUM_DURATION = 0.01
        pyautogui.MINIMUM_SLEEP = 0.01
        self.centers = centers
        self.segment_time = segment_time
        self.press_time = press_time
        self.dry_run = dry_run
        self.abort = abort or threading.Event()

    def _check_abort(self) -> None:
        if self.abort.is_set():
            raise Aborted()

    def drag(self, path: Path) -> None:
        pts = [self.centers[r][c] for r, c in path]
        self._check_abort()
        self.pag.moveTo(*pts[0])

        if self.dry_run:
            for x, y in pts[1:]:
                self._check_abort()
                self.pag.moveTo(x, y, duration=max(self.segment_time, 0.15))
            return

        self.pag.mouseDown()
        try:
            time.sleep(self.press_time)  # let the game register the touch-down
            for x, y in pts[1:]:
                self._check_abort()
                # dragTo (not moveTo) so macOS gets "mouse dragged" events with the
                # button held; mouseDownUp=False keeps one continuous press.
                self.pag.dragTo(x, y, duration=self.segment_time, button="left", mouseDownUp=False)
            time.sleep(self.press_time)
        finally:
            # Always release, even on abort / fail-safe. mouseUp itself would raise
            # FailSafeException with the cursor in a corner, so suspend that check.
            failsafe, self.pag.FAILSAFE = self.pag.FAILSAFE, False
            try:
                self.pag.mouseUp()
            finally:
                self.pag.FAILSAFE = failsafe

"""Dragging, with a fake pyautogui that records events instead of moving the mouse."""
import threading
import time

import pytest

from wordhunt.calibration import interpolate_centers
from wordhunt.dragger import Aborted, Dragger

CENTERS = interpolate_centers((100, 100), (400, 400))  # tiles 100 px apart


class FailSafe(Exception):
    pass


class FakePag:
    def __init__(self, fail_after=None):
        self.events = []
        self.FAILSAFE = True
        self.fail_after = fail_after  # raise like the fail-safe after this many drags

    def moveTo(self, x, y, duration=0):
        self.events.append(("move", x, y))

    def mouseDown(self):
        self.events.append(("down",))

    def mouseUp(self):
        if self.FAILSAFE and self.fail_after is not None:
            raise AssertionError("mouseUp must run with the fail-safe suspended")
        self.events.append(("up",))

    def dragTo(self, x, y, duration=0, button="left", mouseDownUp=True):
        assert duration == 0 and not mouseDownUp
        self.events.append(("drag", x, y))
        if self.fail_after is not None and sum(e[0] == "drag" for e in self.events) >= self.fail_after:
            raise FailSafe()


def test_event_sequence_and_points():
    pag = FakePag()
    d = Dragger(CENTERS, segment_time=0, press_time=0, steps=3, pag=pag)
    d.drag(((0, 0), (1, 1), (1, 2)))  # a diagonal, then a step right
    assert pag.events == [
        ("move", 100, 100), ("down",),
        ("drag", 133, 133), ("drag", 167, 167), ("drag", 200, 200),  # 1/3, 2/3, center
        ("drag", 233, 200), ("drag", 267, 200), ("drag", 300, 200),
        ("up",),
    ]


def test_diagonal_points_stay_inside_tiles():
    """With 3 steps, 1/3 and 2/3 of the way across a diagonal are each within a tile."""
    pag = FakePag()
    Dragger(CENTERS, segment_time=0, press_time=0, steps=3, pag=pag).drag(((0, 0), (1, 1)))
    for _, x, y in [e for e in pag.events if e[0] == "drag"]:
        # distance from the corner point (150, 150) between the four tiles
        assert max(abs(x - 150), abs(y - 150)) >= 16


def test_timing():
    pag = FakePag()
    d = Dragger(CENTERS, pag=pag)  # defaults: ~1.2x the original pyautogui speed
    start = time.perf_counter()
    d.drag(((0, 0), (0, 1), (0, 2), (0, 3), (1, 3)))  # 5 tiles, 4 moves
    took = time.perf_counter() - start
    assert 0.49 <= took < 0.6  # 4 * 0.11 + 2 * 0.025 = 0.49 (original was ~0.58)
    assert sum(e[0] == "drag" for e in pag.events) == 4 * 7

    pag = FakePag()
    start = time.perf_counter()
    Dragger(CENTERS, segment_time=0.03, press_time=0.015, steps=3, pag=pag).drag(
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3)))
    assert 0.15 <= time.perf_counter() - start < 0.25  # the settings are honored


def test_abort_releases_the_button():
    pag = FakePag()
    abort = threading.Event()
    abort.set()
    d = Dragger(CENTERS, segment_time=0, press_time=0, abort=abort, pag=pag)
    with pytest.raises(Aborted):
        d.drag(((0, 0), (0, 1)))
    abort.clear()
    d.abort = abort
    pag.events.clear()

    abort_mid = threading.Event()
    d = Dragger(CENTERS, segment_time=0, press_time=0, abort=abort_mid, pag=pag)
    original = pag.dragTo

    def drag_then_abort(*a, **k):
        original(*a, **k)
        abort_mid.set()

    pag.dragTo = drag_then_abort
    with pytest.raises(Aborted):
        d.drag(((0, 0), (0, 1), (0, 2)))
    assert pag.events[-1] == ("up",)


def test_failsafe_releases_the_button():
    pag = FakePag(fail_after=2)
    d = Dragger(CENTERS, segment_time=0, press_time=0, pag=pag)
    with pytest.raises(FailSafe):
        d.drag(((0, 0), (0, 1), (0, 2)))
    assert pag.events[-1] == ("up",)
    assert pag.FAILSAFE is True  # restored afterwards


def test_dry_run_never_presses():
    pag = FakePag()
    Dragger(CENTERS, dry_run=True, pag=pag).drag(((0, 0), (0, 1)))
    assert all(e[0] == "move" for e in pag.events)

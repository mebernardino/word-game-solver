"""Recording tile positions on screen and saving them to calibration.json."""
from __future__ import annotations

import json
import math
import queue
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

from .solver import SIZE

Point = Tuple[float, float]


@dataclass
class Calibration:
    centers: List[List[Point]]   # [row][col] -> (x, y), mouse coordinates
    regions: List[List[dict]]    # [row][col] -> {left, top, width, height}, mouse coordinates
    mode: str = "simple"
    coord_scale: float = 1.0     # informational; re-detected at runtime
    pixel_scale: float = 1.0

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            data = json.load(f)
        data["centers"] = [[tuple(p) for p in row] for row in data["centers"]]
        return cls(**data)


# --------------------------------------------------------------------------- geometry

def interpolate_centers(top_left: Point, bottom_right: Point) -> List[List[Point]]:
    """Fill in all 16 centers from the two corner tiles.

    The grid is evenly spaced, so tile (r, c) sits a fraction r/3 of the way
    down and c/3 of the way across between the two corner centers:
        x = x_tl + (x_br - x_tl) * c / 3
        y = y_tl + (y_br - y_tl) * r / 3
    """
    (x0, y0), (x1, y1) = top_left, bottom_right
    n = SIZE - 1
    return [
        [(x0 + (x1 - x0) * c / n, y0 + (y1 - y0) * r / n) for c in range(SIZE)]
        for r in range(SIZE)
    ]


def tile_pitch(centers: List[List[Point]]) -> Tuple[float, float]:
    """Average center-to-center spacing horizontally and vertically."""
    dx = [math.dist(centers[r][c], centers[r][c + 1]) for r in range(SIZE) for c in range(SIZE - 1)]
    dy = [math.dist(centers[r][c], centers[r + 1][c]) for r in range(SIZE - 1) for c in range(SIZE)]
    return sum(dx) / len(dx), sum(dy) / len(dy)


def build_regions(centers: List[List[Point]], box_frac: float) -> List[List[dict]]:
    """OCR box for each tile: a square of box_frac * pitch around the center.

    Staying well inside the tile keeps tile borders/gaps out of the OCR image.
    """
    px, py = tile_pitch(centers)
    w, h = px * box_frac, py * box_frac
    return [
        [{"left": x - w / 2, "top": y - h / 2, "width": w, "height": h} for (x, y) in row]
        for row in centers
    ]


# --------------------------------------------------------------------------- interactive

class CalibrationAborted(Exception):
    pass


def capture_points(labels: List[str], key: str, abort_key: str) -> List[Point]:
    """For each label, wait for the hotkey and record the mouse position."""
    import pyautogui
    from .hotkeys import Hotkeys

    events: "queue.Queue[str]" = queue.Queue()
    points: List[Point] = []
    with Hotkeys({key: lambda: events.put("capture"), abort_key: lambda: events.put("abort")}):
        for label in labels:
            print(f"  Hover over the CENTER of {label} and press {key.upper()}  ({abort_key} to cancel)")
            if events.get() == "abort":
                raise CalibrationAborted()
            x, y = pyautogui.position()
            points.append((float(x), float(y)))
            print(f"    recorded ({x}, {y})")
    return points


def run_calibration(advanced: bool, key: str, abort_key: str, box_frac: float) -> Calibration:
    from .screen import detect_scales

    scales = detect_scales()
    print(f"Detected display scaling: {scales}")
    if advanced:
        labels = [f"tile row {r + 1}, col {c + 1}" for r in range(SIZE) for c in range(SIZE)]
        pts = capture_points(labels, key, abort_key)
        centers = [pts[r * SIZE:(r + 1) * SIZE] for r in range(SIZE)]
    else:
        tl, br = capture_points(["the TOP-LEFT tile", "the BOTTOM-RIGHT tile"], key, abort_key)
        centers = interpolate_centers(tl, br)
    px, py = tile_pitch(centers)
    print(f"Tile spacing: {px:.1f} x {py:.1f} (mouse units)")
    return Calibration(
        centers=centers,
        regions=build_regions(centers, box_frac),
        mode="advanced" if advanced else "simple",
        coord_scale=scales.coord_scale,
        pixel_scale=scales.pixel_scale,
    )


def preview(calib: Calibration, dwell: float = 0.25) -> None:
    """Hover (no clicking) over every tile center in reading order."""
    import pyautogui

    print("Previewing tile centers (mouse moves, no clicks)...")
    for r, row in enumerate(calib.centers):
        for c, (x, y) in enumerate(row):
            pyautogui.moveTo(x, y, duration=0.12)
            time.sleep(dwell)
    print("Preview done. If any position looked off, rerun with --calibrate (or --calibrate --advanced).")

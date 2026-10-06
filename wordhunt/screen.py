"""Coordinate spaces and high-DPI handling.

There are three coordinate spaces in play:

  * mouse  - what pyautogui.position()/moveTo use. On macOS these are logical
             "points"; on Windows they're physical pixels if the process is
             DPI-aware (pyautogui makes it so), else scaled pixels.
  * grab   - the coordinates mss expects for a capture region.
  * image  - pixels in the captured image. On a Retina Mac a 50x50-point
             region comes back as a 100x100 image.

Everything we store (tile centers, OCR boxes) is in mouse space. To capture we
convert mouse -> grab with `coord_scale` (usually 1.0, but differs on Windows
if one library is DPI-aware and the other isn't). `pixel_scale` (image/grab)
doesn't need converting - the bigger image just gives OCR more to work with -
but we report it so you can see what was detected.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scales:
    coord_scale: float  # grab units per mouse unit
    pixel_scale: float  # image pixels per grab unit

    def __str__(self) -> str:
        return f"mouse->capture x{self.coord_scale:g}, capture->image x{self.pixel_scale:g}"


def open_capture():
    """An mss screen grabber at the display's full resolution.

    On macOS, mss captures at "nominal" (1x) resolution by default, which halves
    the detail on a Retina screen. Dropping that flag gives 2x images.
    """
    import sys

    import mss

    if sys.platform == "darwin":
        import mss.darwin as darwin

        darwin.IMAGE_OPTIONS = darwin.kCGWindowImageBoundsIgnoreFraming | darwin.kCGWindowImageShouldBeOpaque
    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


def detect_scales() -> Scales:
    import pyautogui

    with open_capture() as sct:
        mon = sct.monitors[1]  # primary monitor
        mouse_w, _ = pyautogui.size()
        coord_scale = mon["width"] / mouse_w
        probe = sct.grab({"left": mon["left"], "top": mon["top"], "width": 20, "height": 20})
        pixel_scale = probe.size.width / 20
    return Scales(round(coord_scale, 3), round(pixel_scale, 3))


def to_grab_region(region: dict, coord_scale: float) -> dict:
    """Convert a {left, top, width, height} box from mouse space to mss space."""
    return {k: int(round(region[k] * coord_scale)) for k in ("left", "top", "width", "height")}

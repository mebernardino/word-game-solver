"""Reading tile letters from the screen.

Pipeline per tile: screenshot (mss) -> isolate the letter (threshold, keep the
largest blob so the point-value dots are dropped) -> template match against
the game's font (glyphs.py) -> if that isn't confident, ask Tesseract too.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .calibration import Calibration
from .glyphs import ALPHABET, GlyphMatcher, crop_to_ink, has_trailing_letter, label_components
from .screen import open_capture, to_grab_region
from .solver import QU, SIZE, Board, Cell

WHITELIST = ALPHABET
TARGET_HEIGHT = 90  # px; Tesseract does best with glyphs ~30-100 px tall
# Tesseract is flaky on a lone glyph: with the whitelist it usually returns
# nothing rather than a wrong letter, and which letters it drops depends on the
# mode and margin (e.g. --psm 10 drops B/O/P/Q/R, --psm 8 drops I). So try
# these (psm, margin px) settings in order and keep the first non-empty read.
OCR_ATTEMPTS = ((10, 30), (8, 10), (13, 30), (10, 10))
TEMPLATE_TRUST = 70  # template confidence at which we skip Tesseract
ROUND_LETTERS = set("QOCG")  # what the Q of a Qu tile can match as
MIN_INK_PIXELS = 20      # less than this is noise / an empty tile mid-animation
MIN_GLYPH_HEIGHT = 0.25  # a letter is at least this fraction of the read area's height
MIN_TILE_LIGHT = 0.5     # at least this much of the read area must be light tile color


@dataclass
class TileRead:
    letter: str        # "A".."Z", "QU" for a Qu tile, or "?" if nothing was recognized
    confidence: float  # 0-100
    source: str = ""   # "template", "tesseract", "both"


def otsu_threshold(gray) -> int:
    """Pick the gray level that best separates two pixel populations (letter vs tile)."""
    import numpy as np

    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    levels = np.arange(256)
    w0 = np.cumsum(hist)                 # pixels at or below each level
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    mean_all = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean_all * w0 / total - m0) ** 2 / (w0 * w1 / total)
    between[~np.isfinite(between)] = 0
    return int(between.argmax())


def extract_glyph(img):
    """Tile screenshot -> cropped boolean ink mask of just the letter, or None."""
    found = _extract(img)
    return found[0] if found else None


def _extract(img):
    """Tile screenshot -> (cropped letter mask, is it followed by a "u"), or None.

    Returns None rather than guessing when the area doesn't look like a tile
    with a dark letter on it, e.g. a window is covering the game, or the tile
    is mid-animation. Callers treat None as an unknown tile.
    """
    import numpy as np

    gray = np.asarray(img.convert("L"))
    rgb = np.asarray(img.convert("RGB")).astype(int)
    # Tiles are light and nearly gray (beige); a covering window, the board's
    # blue background, or a dark app is not.
    light_tile = (gray > 140) & (rgb.max(axis=2) - rgb.min(axis=2) < 60)
    if light_tile.mean() < MIN_TILE_LIGHT:
        return None
    ink = gray < otsu_threshold(gray)
    if not ink.any() or gray[ink].mean() > 110:  # the "ink" must be genuinely dark
        return None
    glyph_ink = letter_component(ink)
    if glyph_ink is None or glyph_ink.sum() < MIN_INK_PIXELS:
        return None
    glyph = crop_to_ink(glyph_ink)
    if glyph.shape[0] < MIN_GLYPH_HEIGHT * gray.shape[0]:
        return None  # a speck or a stray line, not a letter
    return glyph, has_trailing_letter(ink, glyph_ink)


def letter_component(ink):
    """The letter's blob: the largest one, ignoring line-like blobs along the
    edge of the read area (a tile border, a window edge). A letter that's merely
    clipped by the edge (calibration a little off) is kept."""
    import numpy as np

    h, w = ink.shape
    labels, sizes = label_components(ink)
    best, best_size = None, 0
    for label, size in sizes.items():
        ys, xs = np.nonzero(labels == label)
        on_edge = ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
        span_h, span_w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        line_like = span_w > 0.8 * w or span_h > 0.8 * h or min(span_h, span_w) < 0.12 * max(h, w)
        if on_edge and line_like:
            continue
        if size > best_size:
            best, best_size = label, size
    return labels == best if best is not None else None


def glyph_to_image(glyph, border: int = 30):
    """Ink mask -> black-on-white image, upscaled and padded the way Tesseract likes."""
    import numpy as np
    from PIL import Image, ImageOps

    out = Image.fromarray(np.where(glyph, 0, 255).astype("uint8"))
    scale = TARGET_HEIGHT / max(out.height, 1)
    out = out.resize((max(1, round(out.width * scale)), TARGET_HEIGHT), Image.LANCZOS)
    return ImageOps.expand(out, border=border, fill=255)


def preprocess(img, border: int = 30):
    """Tile screenshot -> image ready for Tesseract (blank white if no letter found)."""
    import numpy as np

    glyph = extract_glyph(img)
    if glyph is None:
        glyph = np.zeros((TARGET_HEIGHT, TARGET_HEIGHT // 2), dtype=bool)
    return glyph_to_image(glyph, border)


def default_matcher(glyph_dir: Optional[str], font: Optional[str] = None) -> GlyphMatcher:
    matcher = GlyphMatcher()
    used = matcher.add_font(font)
    loaded = matcher.load_dir(glyph_dir) if glyph_dir else 0
    print(f"Letter templates: font {os.path.basename(used) if used else 'none found'}, "
          f"{loaded} from {glyph_dir}/")
    return matcher


class TileReader:
    def __init__(
        self,
        calib: Calibration,
        coord_scale: float,
        tesseract_cmd: Optional[str] = None,
        debug_dir: Optional[str] = None,
        matcher: Optional[GlyphMatcher] = None,
        glyph_dir: Optional[str] = None,
    ):
        self.tess = None
        try:
            import pytesseract

            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            pytesseract.get_tesseract_version()
            self.tess = pytesseract
        except Exception:
            print("Tesseract not available; using template matching only.")
        self.sct = open_capture()
        self.calib = calib
        self.coord_scale = coord_scale
        self.debug_dir = debug_dir
        self.matcher = matcher or GlyphMatcher()
        self.glyph_dir = glyph_dir
        self.last_glyphs: Dict[Cell, object] = {}  # most recent ink mask per tile, for learning
        self._shots = 0
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    def grab_many(self, cells: Iterable[Cell]) -> dict:
        """Screenshot each tile's read area. Takes ONE screenshot spanning all of
        them and crops it: each screenshot costs ~15 ms, so 16 separate ones
        made a full-board read ~15x slower."""
        from PIL import Image

        cells = list(cells)
        boxes = {c: to_grab_region(self.calib.regions[c[0]][c[1]], self.coord_scale) for c in cells}
        left = min(b["left"] for b in boxes.values())
        top = min(b["top"] for b in boxes.values())
        right = max(b["left"] + b["width"] for b in boxes.values())
        bottom = max(b["top"] + b["height"] for b in boxes.values())
        shot = self.sct.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        px = shot.size.width / (right - left)  # image pixels per capture unit (2 on Retina)
        return {
            c: img.crop((round((b["left"] - left) * px), round((b["top"] - top) * px),
                         round((b["left"] - left + b["width"]) * px), round((b["top"] - top + b["height"]) * px)))
            for c, b in boxes.items()
        }

    def _ocr(self, img, psm: int) -> TileRead:
        cfg = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
        data = self.tess.image_to_data(img, config=cfg, output_type=self.tess.Output.DICT)
        best = TileRead("?", 0.0, "tesseract")
        for text, conf in zip(data["text"], data["conf"]):
            text = "".join(ch for ch in str(text).upper() if ch in WHITELIST)
            conf = float(conf)
            if text and conf > best.confidence:
                # Tesseract only sees the main shape; Qu is decided separately.
                best = TileRead(text[0], conf, "tesseract")
        return best

    def _tesseract(self, glyph) -> TileRead:
        for psm, margin in OCR_ATTEMPTS:
            result = self._ocr(glyph_to_image(glyph, margin), psm)
            if result.letter != "?":
                return result
        return TileRead("?", 0.0, "tesseract")

    def read(self, cell: Cell) -> TileRead:
        return self.read_cells([cell])[cell]

    def read_cells(self, cells: Iterable[Cell]) -> dict:
        images = self.grab_many(cells)
        return {cell: self._read_image(cell, raw) for cell, raw in images.items()}

    def _read_image(self, cell: Cell, raw) -> TileRead:
        found = _extract(raw)
        glyph = found[0] if found else None
        if glyph is None:
            result = TileRead("?", 0.0)
        else:
            self.last_glyphs[cell] = glyph
            letter, conf = self.matcher.classify(glyph)
            result = TileRead(letter, conf, "template")
            if conf < TEMPLATE_TRUST and self.tess is not None:
                tess = self._tesseract(glyph)
                if tess.letter == letter:
                    result = TileRead(letter, max(conf, tess.confidence, TEMPLATE_TRUST), "both")
                elif tess.confidence > conf:
                    result = tess
            if found[1]:
                # A second letter after the first: in this game only the Qu tile
                # has two letters. Its Q has a short tail and often matches O
                # better than Q, so a round first letter is enough to call it Qu.
                if result.letter in ROUND_LETTERS:
                    result = TileRead(QU, max(result.confidence, TEMPLATE_TRUST), result.source)
                else:
                    result = TileRead(QU, 40, result.source)  # doubtful: retried once
        if self.debug_dir:
            self._shots += 1
            stem = os.path.join(self.debug_dir, f"{self._shots:05d}_r{cell[0]}c{cell[1]}_{result.letter}")
            raw.save(stem + "_raw.png")
            if glyph is not None:
                glyph_to_image(glyph).save(stem + "_glyph.png")
        return result

    def read_all(self) -> dict:
        return self.read_cells((r, c) for r in range(SIZE) for c in range(SIZE))

    def learn(self, board: Board) -> int:
        """Add the latest image of each tile as a template for the letter the user
        confirmed. Returns how many new templates were kept."""
        added = 0
        for cell, glyph in self.last_glyphs.items():
            letter = "Q" if board[cell] == QU else board[cell]  # a Qu tile's main shape is its Q
            if letter in ALPHABET and self.matcher.learn(letter, glyph, self.glyph_dir):
                added += 1
        return added

"""Letter recognition by template matching.

The game draws every letter in one clean font, so comparing a tile's letter
against reference images ("templates") beats general-purpose OCR. Templates
come from three places, in increasing order of trust:

  1. letters rendered from a system font that resembles the game's,
  2. PNGs in the glyphs/ folder (seeded from a real screenshot),
  3. tiles you confirm at the start of a game, which are added to glyphs/,
     so recognition keeps improving as you play.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

GLYPH_SIZE = 32  # templates and tiles are compared as 32x32 images
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_PER_LETTER = 8  # learned templates kept per letter
LEARN_MIN_FONT_MATCH = 0.6  # a tile must resemble the font's letter this much to be learned

# Fonts resembling the game's. The first one found is used. Measured against
# a real board, SF Pro Semibold (the iOS system font) matched best, then
# Helvetica Neue Medium. (path, face index in a .ttc, variable-font weight or None)
FONT_CANDIDATES = [
    ("/System/Library/Fonts/SFNS.ttf", 0, 600),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 10, None),  # Medium
    ("/System/Library/Fonts/Helvetica.ttc", 1, None),       # Bold
    ("C:/Windows/Fonts/seguisb.ttf", 0, None),              # Segoe UI Semibold
    ("C:/Windows/Fonts/arialbd.ttf", 0, None),
]


# --------------------------------------------------------------------------- glyph extraction

def label_components(ink):
    """8-connected blobs of ink. Returns (labels array, {label: pixel count})."""
    import numpy as np

    h, w = ink.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes = {}
    current = 0
    ys, xs = np.nonzero(ink)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0, x0]:
            continue
        current += 1
        labels[y0, x0] = current
        stack, size = [(y0, x0)], 0
        while stack:  # flood fill
            y, x = stack.pop()
            size += 1
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < h and 0 <= nx < w and ink[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
        sizes[current] = size
    return labels, sizes


def largest_component(ink):
    """Keep only the biggest 8-connected blob of ink pixels.

    Capital letters in this font are each one connected shape, so this drops
    the point-value dots under the letter, specks of the tile edge, etc.
    (On a "Qu" tile it keeps the Q; the u is detected separately.)
    """
    labels, sizes = label_components(ink)
    if not sizes:
        return ink
    return labels == max(sizes, key=sizes.get)


def has_trailing_letter(ink, main) -> bool:
    """True if another letter-sized shape sits right of the main letter, at
    letter height: the "u" of a "Qu" tile. The point-value dots don't count,
    since they sit below the letter."""
    import numpy as np

    rows, cols = np.nonzero(main)
    top, bottom, center_x = rows.min(), rows.max(), cols.mean()
    main_h, main_w = bottom - top + 1, cols.max() - cols.min() + 1
    labels, sizes = label_components(ink & ~main)
    for label, size in sizes.items():
        if size < 0.08 * main.sum():
            continue
        ys, xs = np.nonzero(labels == label)
        if (top <= ys.mean() <= bottom and xs.min() > center_x
                and ys.max() - ys.min() + 1 <= 1.1 * main_h and xs.max() - xs.min() + 1 <= 1.5 * main_w):
            return True
    return False


def crop_to_ink(ink):
    import numpy as np

    rows, cols = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
    return ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def normalize(glyph):
    """Cropped ink mask -> GLYPH_SIZE x GLYPH_SIZE float array (1.0 = ink).

    The glyph is centered on a square canvas before resizing, so its aspect
    ratio is kept (a thin I stays thin) while its size on screen stops mattering.
    """
    import numpy as np
    from PIL import Image

    h, w = glyph.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = glyph.astype(np.uint8) * 255
    small = Image.fromarray(canvas).resize((GLYPH_SIZE, GLYPH_SIZE), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


def _correlation(a, b) -> float:
    """Normalized cross-correlation: 1.0 = identical shape, ~0 = unrelated."""
    import numpy as np

    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / denom if denom else 0.0


# --------------------------------------------------------------------------- matcher

class GlyphMatcher:
    def __init__(self) -> None:
        self.templates: List[Tuple[str, object, str]] = []  # (letter, 32x32 array, source)

    def add(self, letter: str, glyph, source: str = "learned") -> None:
        self.templates.append((letter, normalize(glyph), source))

    def classify(self, glyph) -> Tuple[str, float]:
        """Return (letter, confidence 0-100) for a cropped ink mask."""
        if not self.templates:
            return "?", 0.0
        vec = normalize(glyph)
        best = {}
        for letter, tmpl, _ in self.templates:
            score = _correlation(vec, tmpl)
            if score > best.get(letter, -1.0):
                best[letter] = score
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        letter, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        # Confident when the match is close AND clearly better than any other letter.
        # On a real board, correct font-only matches scored ~0.9 with the nearest
        # other letter (P/R, O/C, G/C) ~0.075 behind, which maps to ~90 here.
        confidence = max(0.0, min(1.0, (top - 0.55) / 0.3)) * max(0.0, min(1.0, (top - runner_up) / 0.08))
        return letter, round(confidence * 100, 1)

    # ---- sources

    def add_font(self, path: Optional[str] = None) -> Optional[str]:
        """Render A-Z from the first available font. Returns the font used."""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        candidates = [(path, 0, None)] if path else FONT_CANDIDATES
        for font_path, index, weight in candidates:
            if not os.path.exists(font_path):
                continue
            try:
                font = ImageFont.truetype(font_path, 120, index=index)
                if weight is not None:
                    # Set the Weight axis by name; leave Width, Optical Size, etc. at default.
                    axes = font.get_variation_axes()
                    font.set_variation_by_axes([
                        weight if a["name"] in (b"Weight", "Weight") else a["default"] for a in axes
                    ])
            except (OSError, ValueError):
                continue
            for letter in ALPHABET:
                img = Image.new("L", (200, 200), 0)
                ImageDraw.Draw(img).text((100, 100), letter, fill=255, font=font, anchor="mm")
                ink = np.asarray(img) > 127
                self.add(letter, crop_to_ink(largest_component(ink)), source="font")
            return font_path
        return None

    def load_dir(self, folder: str) -> int:
        """Load <LETTER>_*.png ink masks (white ink on black) from folder."""
        import numpy as np
        from PIL import Image

        if not os.path.isdir(folder):
            return 0
        n = 0
        for name in sorted(os.listdir(folder)):
            letter = name[:1].upper()
            if name.lower().endswith(".png") and letter in ALPHABET and name[1:2] == "_":
                ink = np.asarray(Image.open(os.path.join(folder, name)).convert("L")) > 127
                if ink.any():
                    self.add(letter, crop_to_ink(ink), source="file")
                    n += 1
        return n

    def learn(self, letter: str, glyph, folder: Optional[str]) -> bool:
        """Remember a confirmed tile. Skips near-duplicates of what we already have.
        Returns True if a new template was added."""
        import numpy as np
        from PIL import Image

        vec = normalize(glyph)
        # Sanity check against the font, so a wrongly confirmed tile (or a
        # window edge read as a letter) doesn't become a bad template.
        font = [t for l, t, src in self.templates if l == letter and src == "font"]
        if font and max(_correlation(vec, t) for t in font) < LEARN_MIN_FONT_MATCH:
            return False
        own = [t for l, t, src in self.templates if l == letter and src != "font"]
        if any(_correlation(vec, t) > 0.97 for t in own) or len(own) >= MAX_PER_LETTER:
            return False
        self.templates.append((letter, vec, "learned"))
        if folder:
            os.makedirs(folder, exist_ok=True)
            i = 0
            while os.path.exists(os.path.join(folder, f"{letter}_{i:02d}.png")):
                i += 1
            Image.fromarray(glyph.astype(np.uint8) * 255).save(os.path.join(folder, f"{letter}_{i:02d}.png"))
        return True

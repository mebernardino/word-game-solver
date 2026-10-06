import shutil

import pytest

from wordhunt.calibration import Calibration, build_regions, interpolate_centers, tile_pitch
from wordhunt.screen import to_grab_region


def test_interpolate_centers():
    centers = interpolate_centers((100, 200), (400, 500))
    assert centers[0][0] == (100, 200)
    assert centers[3][3] == (400, 500)
    assert centers[0][3] == (400, 200)
    assert centers[2][1] == (200, 400)
    assert tile_pitch(centers) == pytest.approx((100, 100))


def test_regions_and_round_trip(tmp_path):
    centers = interpolate_centers((100, 200), (400, 500))
    regions = build_regions(centers, 0.5)
    assert regions[0][0] == {"left": 75, "top": 175, "width": 50, "height": 50}
    path = tmp_path / "calibration.json"
    Calibration(centers, regions).save(str(path))
    loaded = Calibration.load(str(path))
    assert loaded.centers == centers and loaded.regions == regions


def test_grab_region_scaling():
    region = {"left": 75, "top": 175, "width": 50, "height": 50}
    assert to_grab_region(region, 1.5) == {"left": 112, "top": 262, "width": 75, "height": 75}


def _letter_image(letter, size=60):
    """A dark letter on the game's light gray-beige tile color."""
    from PIL import Image, ImageDraw, ImageFont

    bg, fg = (215, 213, 203), (20, 20, 20)
    img = Image.new("RGB", (size, size), bg)
    try:
        font = ImageFont.truetype("Arial Bold.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((size // 2, size // 2), letter, fill=fg, font=font, anchor="mm")
    return img


def test_preprocess_gives_dark_text_on_white():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from wordhunt.ocr import TARGET_HEIGHT, preprocess

    out = np.asarray(preprocess(_letter_image("W")))
    assert out.shape[0] == TARGET_HEIGHT + 60
    assert out[0, 0] == 255 and out[-1, -1] == 255  # white margin
    assert (out < 128).mean() > 0.05                # some dark ink


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_reads_letters():
    pytest.importorskip("pytesseract")
    from wordhunt.glyphs import GlyphMatcher
    from wordhunt.ocr import TileReader

    import pytesseract

    reader = TileReader.__new__(TileReader)  # skip the screen-capture setup
    reader.tess, reader.debug_dir = pytesseract, None
    reader.matcher, reader.last_glyphs = GlyphMatcher(), {}  # no templates: Tesseract only
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        reader.grab_many = lambda cells: {cell: _letter_image(letter) for cell in cells}
        assert reader.read((0, 0)).letter == letter


def test_covered_or_blank_tiles_are_unknown():
    pytest.importorskip("numpy")
    from PIL import Image

    from wordhunt.ocr import extract_glyph

    dark_window = Image.new("RGB", (60, 60), (30, 30, 30))
    blank_tile = Image.new("RGB", (60, 60), (215, 213, 203))
    blue_board = Image.new("RGB", (60, 60), (44, 90, 127))
    assert extract_glyph(dark_window) is None
    assert extract_glyph(blank_tile) is None
    assert extract_glyph(blue_board) is None
    assert extract_glyph(_letter_image("A")) is not None


def test_half_covered_tile_is_unknown():
    pytest.importorskip("numpy")
    from wordhunt.ocr import extract_glyph

    img = _letter_image("E")
    img.paste((30, 30, 30), (0, 0, 40, 60))  # a window covers the left two thirds
    assert extract_glyph(img) is None


def test_ignores_line_touching_edge():
    np = pytest.importorskip("numpy")
    from wordhunt.ocr import extract_glyph

    img = _letter_image("I")
    img.paste((20, 20, 20), (0, 56, 60, 59))  # a dark edge line wider than the letter
    glyph = extract_glyph(img)
    assert glyph is not None and glyph.shape[0] > glyph.shape[1]  # the tall I, not the line

"""Letter recognition against a real screenshot of the game (tests/fixtures/board_sample.png)."""
import os
import random

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from wordhunt.calibration import Calibration, build_regions, interpolate_centers
from wordhunt.glyphs import GlyphMatcher
from wordhunt.ocr import TileReader, extract_glyph
from wordhunt.solver import Board

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "fixtures", "board_sample.png")
TRUTH = "KIKOHOIDMNDOPXNG"
# Centers of the top-left and bottom-right tiles in the sample, in its pixels.
CENTERS = interpolate_centers((80, 86.5), (459, 465))


@pytest.fixture(scope="module")
def sample():
    return Image.open(SAMPLE).convert("RGB")


@pytest.fixture(scope="module")
def font_matcher():
    m = GlyphMatcher()
    if m.add_font() is None:
        pytest.skip("no suitable system font")
    return m


def tile(sample, r, c, box=0.7, dx=0.0, dy=0.0):
    g = build_regions(CENTERS, box)[r][c]
    left, top = g["left"] + dx, g["top"] + dy
    return sample.crop((round(left), round(top), round(left + g["width"]), round(top + g["height"])))


def test_dots_are_removed(sample):
    # The I tile (row 1, col 2) has two dots under it. Its glyph should be one
    # tall thin stroke the height of a letter, with no dots below.
    glyph = extract_glyph(tile(sample, 0, 1))
    h, w = glyph.shape
    assert 38 <= h <= 48 and w <= 10
    assert glyph.all()  # a solid bar: nothing else survived


def test_font_templates_read_the_sample_board(sample, font_matcher):
    reads = [font_matcher.classify(extract_glyph(tile(sample, r, c))) for r in range(4) for c in range(4)]
    assert "".join(letter for letter, _ in reads) == TRUTH
    assert min(conf for _, conf in reads) >= 70


def test_robust_to_imprecise_calibration(sample, font_matcher):
    rng = random.Random(1)
    for _ in range(5):
        got = ""
        for r in range(4):
            for c in range(4):
                dx, dy = rng.uniform(-10, 10), rng.uniform(-10, 10)
                got += font_matcher.classify(extract_glyph(tile(sample, r, c, dx=dx, dy=dy)))[0]
        assert got == TRUTH


def test_non_letters_get_low_confidence(font_matcher):
    blob = np.ones((40, 40), dtype=bool)
    noise = np.random.default_rng(0).random((40, 40)) > 0.5
    assert font_matcher.classify(blob)[1] < 30
    assert font_matcher.classify(noise)[1] < 30


def test_learn_saves_and_reloads(sample, tmp_path):
    m = GlyphMatcher()
    glyph = extract_glyph(tile(sample, 3, 0))  # P
    assert m.learn("P", glyph, str(tmp_path))
    assert not m.learn("P", glyph, str(tmp_path))  # near-duplicate skipped
    assert os.listdir(tmp_path) == ["P_00.png"]

    m2 = GlyphMatcher()
    assert m2.load_dir(str(tmp_path)) == 1
    assert m2.classify(glyph) == ("P", 100.0)


def test_reader_reads_and_learns_from_confirmed_board(sample, tmp_path):
    """TileReader end to end, with the screenshot standing in for the screen."""
    reader = TileReader.__new__(TileReader)  # skip mss / tesseract setup
    reader.tess, reader.debug_dir = None, None
    reader.matcher, reader.glyph_dir, reader.last_glyphs = GlyphMatcher(), str(tmp_path), {}
    reader.matcher.add_font()
    reader.grab_many = lambda cells: {cell: tile(sample, *cell) for cell in cells}

    reads = reader.read_all()
    assert "".join(reads[(r, c)].letter for r in range(4) for c in range(4)) == TRUTH
    added = reader.learn(Board.parse(TRUTH))
    assert added == len(os.listdir(tmp_path)) >= 10


def _game_font(px):
    from PIL import ImageFont

    path = "/System/Library/Fonts/SFNS.ttf"
    if not os.path.exists(path):
        pytest.skip("SF Pro not available")
    f = ImageFont.truetype(path, px)
    f.set_variation_by_axes([600 if a["name"] in (b"Weight", "Weight") else a["default"]
                             for a in f.get_variation_axes()])
    return f


def _mock_tile(text, px, dx=0, dy=0, dots=3):
    """A tile like the game's: dark text on the light tile color, dots below."""
    from PIL import ImageDraw

    img = Image.new("RGB", (86, 86), (215, 213, 203))
    d = ImageDraw.Draw(img)
    d.text((43 + dx, 35 + dy), text, fill=(20, 20, 20), font=_game_font(px), anchor="mm")
    for i in range(dots):
        d.ellipse((34 + i * 7, 69 + dy, 38 + i * 7, 73 + dy), fill=(20, 20, 20))
    return img


def _mock_reader(font_matcher):
    reader = TileReader.__new__(TileReader)  # skip mss / tesseract setup
    reader.tess, reader.debug_dir, reader.glyph_dir, reader.last_glyphs = None, None, None, {}
    reader.matcher = font_matcher
    return reader


OFFSETS = [(0, 0), (-6, 4), (6, -4), (4, 6)]


@pytest.mark.parametrize("text,px", [("Qu", 58), ("Qu", 46), ("QU", 40)])
def test_qu_tile_reads_as_qu(font_matcher, text, px):
    reader = _mock_reader(font_matcher)
    for dx, dy in OFFSETS:
        reader.grab_many = lambda cells: {cell: _mock_tile(text, px, dx, dy) for cell in cells}
        assert reader.read((0, 0)).letter == "QU", (dx, dy)


@pytest.mark.parametrize("text", ["Q", "O", "G", "C"])
def test_single_round_letters_are_not_qu(font_matcher, text):
    reader = _mock_reader(font_matcher)
    for dx, dy in OFFSETS:
        for dots in (1, 4):
            reader.grab_many = lambda cells: {cell: _mock_tile(text, 58, dx, dy, dots) for cell in cells}
            assert reader.read((0, 0)).letter == text, (dx, dy, dots)


def test_learns_qu_tile_as_q(font_matcher, tmp_path):
    reader = _mock_reader(GlyphMatcher())
    reader.matcher.add_font()
    reader.glyph_dir = str(tmp_path)
    reader.grab_many = lambda cells: {cell: _mock_tile("Qu", 58) for cell in cells}
    reader.read((0, 0))
    board = Board(["QU"] + ["A"] * 15)
    assert reader.learn(board) == 1 and os.listdir(tmp_path) == ["Q_00.png"]


@pytest.mark.parametrize("text,px", [("Qu", 58), ("Qu", 46), ("QU", 40)])
def test_qu_tile_main_shape_is_q(font_matcher, text, px):
    """The Q is the bigger of a Qu tile's two shapes, so it's the one matched."""
    for dx, dy in OFFSETS:
        assert font_matcher.classify(extract_glyph(_mock_tile(text, px, dx, dy)))[0] == "Q"


# ------------------------------------------------------------------ real Qu tile

QU_SAMPLE = os.path.join(HERE, "fixtures", "board_qu.png")
QU_TRUTH = ["E", "R", "T", "W", "H", "L", "A", "D", "N", "QU", "L", "E", "K", "A", "H", "O"]
QU_CENTERS = interpolate_centers((96, 88.5), (475, 467))


def test_real_qu_board(font_matcher):
    """The game's Qu tile: its Q has a short tail that matches O better than Q,
    so the reader goes by the second letter (the u) instead."""
    img = Image.open(QU_SAMPLE).convert("RGB")
    regions = build_regions(QU_CENTERS, 0.7)
    reader = _mock_reader(font_matcher)
    rng = random.Random(2)
    for jitter in (0, 10, 10, 10):  # exact centers, then imprecise calibration
        got = []
        for r in range(4):
            for c in range(4):
                g = regions[r][c]
                dx, dy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
                box = (round(g["left"] + dx), round(g["top"] + dy),
                       round(g["left"] + g["width"] + dx), round(g["top"] + g["height"] + dy))
                reader.grab_many = lambda cells, box=box: {cell: img.crop(box) for cell in cells}
                got.append(reader.read((r, c)).letter)
        assert got == QU_TRUTH


def test_real_board_with_y_w_c_s(font_matcher):
    img = Image.open(os.path.join(HERE, "fixtures", "board_y.png")).convert("RGB")
    regions = build_regions(interpolate_centers((78, 82.5), (457, 461)), 0.7)
    reader = _mock_reader(font_matcher)
    rng = random.Random(3)
    for jitter in (0, 10, 10, 10):
        got = ""
        for r in range(4):
            for c in range(4):
                g = regions[r][c]
                dx, dy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
                box = (round(g["left"] + dx), round(g["top"] + dy),
                       round(g["left"] + g["width"] + dx), round(g["top"] + g["height"] + dy))
                reader.grab_many = lambda cells, box=box: {cell: img.crop(box) for cell in cells}
                got += reader.read((r, c)).letter
        assert got == "MRYYOHAYTOWNXSXC"

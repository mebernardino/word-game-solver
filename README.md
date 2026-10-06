# wordhunt

Auto-plays a 4x4 word-search puzzle on your screen (Boggle / GamePigeon Word Hunt
style). It reads the tiles with OCR, finds words with a trie + DFS, and drags the
mouse along each word's path.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Tesseract** (backup OCR engine; letters are mainly read by template matching, see below):

- macOS: `brew install tesseract`
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, then either add it to
  PATH or pass `--tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"`

**Dictionary**: any plain-text list, one word per line. ENABLE (~173k words, public domain) works well:

```bash
curl -O https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt
```

The game uses its own word list, so some ENABLE words will be rejected. The bot spots
this (the tiles still show the same letters after a second look), doesn't count the points,
and moves on. Rejected words are saved in `rejected.txt` and skipped in every later game
(use `--reject-after 2` to require two separate games first). A rejection is only saved once a later word in the same game
is accepted, so a game where no drags register (e.g. the window lost focus) can't add words
to it. Rejected words don't animate, so once the bot has seen how soon an accepted word's new
letters appear, it calls a word rejected shortly after that point instead of waiting the full
`--post-word-delay` and taking a second look. After 5 rejections in a row it prints a warning but keeps playing. Edit `rejected.txt` freely: delete a line to
allow a word again, or add a word on its own line to skip it right away.

### macOS permissions

In **System Settings > Privacy & Security**, grant your terminal app (Terminal, iTerm, VS Code...):

- **Accessibility**: mouse control (pyautogui)
- **Screen Recording**: screenshots (mss). Without it, captures show only the wallpaper.
- **Input Monitoring**: global hotkeys (pynput)

Restart the terminal after granting them. On Mac keyboards you may need `fn+F8`, or pick
another key with `--calib-key`.

## Usage

```bash
# 1. Calibrate: hover over the CENTER of the top-left tile and press F8, then the bottom-right tile.
python -m wordhunt --calibrate
python -m wordhunt --calibrate --advanced     # record all 16 centers if the grid is skewed

# 2. Rehearse without clicking
python -m wordhunt --dict enable1.txt --dry-run --max-words 5

# 3. Play
python -m wordhunt --dict enable1.txt --time-limit 80
python -m wordhunt --dict enable1.txt --strategy rate --segment-time 0.05
python -m wordhunt --dict enable1.txt --manual   # type the letters yourself, no OCR
```

At start-up you get a 3-second countdown to click the game window. Make sure no other
window (including the terminal) covers any part of the board: a covered tile can't be read.
The detected grid is then printed in the terminal; unreadable tiles show as `?`. Press Enter to accept it, type all 16
letters, or type `row col letter` (1-based, e.g. `2 3 E`) to fix one tile.

**Q and Qu are different tiles.** A Q tile spells Q (QI, QAT) and a Qu tile spells QU
(QUIT uses three tiles and scores as four letters). Fix one with `3 2 Q` or `3 2 Qu`.
When typing whole rows, a row with 5 letters containing QU means that QU is one tile
(`QUATX` = Qu, A, T, X), while `QUAT` is four tiles (Q, U, A, T).

**Hotkeys:** `Esc` stops immediately (the mouse button is released). `F9` pauses after the
current word so you can correct tiles. Slamming the mouse into a screen corner also
stops it (pyautogui's fail-safe).

### Main flags

| flag | default | |
|---|---|---|
| `--strategy` | `score` | `score`, `rate` (points per second of dragging), `keep-good-letters` |
| `--segment-time` | 0.06 | seconds per tile-to-tile move |
| `--post-word-delay` | 0.51 | wait for the replacement animation before re-reading tiles |
| `--countdown` | 3 | seconds to focus the game window |
| `--max-words` / `--time-limit` | off | stop conditions |
| `--full-reread-every` | 1 | re-read the whole board every N words to correct misreads (1 = after every word) |
| `--min-conf` | 60 | OCR confidence below which a tile is re-shot once; a tile that still reads blank is skipped and re-read after the next word |
| `--rejected` / `--reject-after` | `rejected.txt` / 1 | where rejected words are saved / games before a word is skipped (0 = never) |
| `--min-len` / `--scores` | 3 / `3:100,4:400,5:800,6:1400,7:1800,8:2200` | |
| `--box` | 0.7 | size of the area read on each tile, as a fraction of the tile spacing |
| `--glyphs DIR` | `glyphs` | letter templates; boards you confirm are added here |
| `--no-learn` / `--font PATH` | | don't add templates / render templates from a different font |
| `--save-tiles DIR` | | dump raw and preprocessed tile images to debug OCR |

Run `python -m wordhunt --help` for everything.

### How letters are read

The game draws every letter in one font, so tiles are matched against templates
instead of relying on general OCR:

1. The letter is separated from the tile, keeping only the largest dark shape.
   That drops the point-value dots underneath.
2. It's compared at 32x32 against templates: A-Z rendered from SF Pro Semibold (the
   iOS font, which matched a real board best), plus the real tiles in `glyphs/`.
3. Tesseract is only consulted when the template match isn't confident.

`glyphs/` starts with 14 tiles from a real screenshot. Every board you confirm at the
start of a game (or correct with F9) is added to it, so letters it hasn't seen yet,
and any rendering differences, are learned as you play.

### OCR troubleshooting

Run once with `--save-tiles ocr_debug` and look at the `_glyph.png` files. Each should
show one black letter on white. If part of a neighboring tile leaks in, lower `--box`.
If letters are clipped, recalibrate, aiming at the tile centers. If a wrong board got
confirmed, delete the bad templates from `glyphs/` (files are named by letter).

## Layout

```
wordhunt/
  solver.py       Board, trie, DFS, candidate set (pure Python, no dependencies)
  strategy.py     word choice: score / rate / keep-good-letters
  calibration.py  recording tile centers, interpolation, calibration.json
  screen.py       Retina / high-DPI coordinate conversion
  dragger.py      mouse dragging with abort + fail-safe
  ocr.py          mss capture, letter extraction, template + Tesseract reading
  glyphs.py       template matching against the game's font
glyphs/           letter templates (seeded from a real board, grows as you play)
  hotkeys.py      global hotkeys (pynput)
  rejects.py      rejected.txt: words the game refused
  cli.py          argparse + main loop
tests/            pytest suite, including a simulated full game
```

```bash
pytest
```

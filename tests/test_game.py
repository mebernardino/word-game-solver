"""Run the real play loop against a simulated game (no screen, mouse or OCR)."""
import random
import threading

from wordhunt.cli import Game, build_parser
from wordhunt.ocr import TileRead
from wordhunt.solver import Board, Solver, Trie

WORDS = ["CAT", "CATS", "COD", "CODE", "CORE", "DOG", "DOGS", "DOTS", "TONE", "TONED", "DONE",
         "RED", "TEN", "NET", "ODE", "DOE", "ROE", "TOE", "EON", "NOD", "TOD", "GOD", "COT"]


class FakeScreen:
    def __init__(self, letters, seed=0):
        self.board = Board.parse(letters)
        self.rng = random.Random(seed)
        self.drags = []
        self.paths = []


class FakeReader:
    def __init__(self, screen, unclear_once=(), unreadable=(), misread_once=None):
        self.screen = screen
        self.unclear = set(unclear_once)
        self.unreadable = set(unreadable)
        self.misread = dict(misread_once or {})  # cell -> wrong letter, confidently, once

    def read_cells(self, cells):
        out = {}
        for cell in cells:
            if cell in self.misread:
                out[cell] = TileRead(self.misread.pop(cell), 95)
            elif cell in self.unreadable:
                out[cell] = TileRead("?", 0)
            elif cell in self.unclear:
                self.unclear.discard(cell)  # low confidence first time, fine on retry
                out[cell] = TileRead("?", 10)
            else:
                out[cell] = TileRead(self.screen.board[cell], 95)
        return out


    def learn(self, board):
        return 0


class FakeDragger:
    def __init__(self, screen):
        self.screen = screen

    def drag(self, path):
        spelled = "".join(self.screen.board.tile_text(r * 4 + c) for r, c in path)
        self.screen.drags.append(spelled)
        self.screen.paths.append(path)
        for cell in path:  # the game refills used tiles with random letters
            self.screen.board[cell] = self.screen.rng.choice("EEAOTNRDSCG")


def make_game(argv, screen, reader):
    args = build_parser().parse_args(["--countdown", "0", "--post-word-delay", "0", "--retry-delay", "0"] + argv)
    solver = Solver(Trie.from_words(WORDS))
    return Game(args, solver, reader, FakeDragger(screen), threading.Event(), threading.Event())


def test_plays_valid_words_tracks_board_and_never_repeats(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": "")  # accept the detected board
    screen = FakeScreen("CATS ODOG RENT SXQZ")
    game = make_game(["--max-words", "12", "--full-reread-every", "4", "--no-repeats"], screen, FakeReader(screen))
    game.run()

    assert 1 <= game.words <= 12
    # Every drag spelled exactly the word that was chosen and scored.
    assert screen.drags == [w for w in screen.drags if w in WORDS]
    assert len(screen.drags) == len(set(screen.drags))
    # Our model of the board matches the "screen" after all the refills.
    assert game.board.letters == screen.board.letters


def test_unclear_tile_is_retried(monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or "")
    screen = FakeScreen("CATS ODOG RENT SXQZ")
    game = make_game(["--max-words", "1"], screen, FakeReader(screen, unclear_once={(0, 0)}))
    game.run()
    # Only the board-confirmation prompt; the unclear tile resolved on retry.
    assert not any("unclear" in p for p in prompts)
    assert screen.drags == ["TONED"]


def test_unreadable_tile_never_prompts_mid_game(monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or "")
    screen = FakeScreen("CATS ODOG RENT SXQZ")
    reader = FakeReader(screen, unreadable={(1, 1)})  # the D is never readable
    game = make_game(["--max-words", "5"], screen, reader)
    game.run()
    assert len(prompts) == 1  # only the board confirmation before play starts
    assert game.words >= 1
    # Every word was real and avoided the unreadable tile.
    assert all(w in WORDS for w in screen.drags)
    assert all((1, 1) not in path for path in screen.paths)


def test_unknown_tile_is_reread_after_next_word(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda p="": "")
    screen = FakeScreen("CATS ODOG RENT SXQZ")
    reader = FakeReader(screen, unreadable={(3, 3)})
    game = make_game(["--max-words", "1"], screen, reader)
    game.board = game.read_full_board()
    assert game.board[(3, 3)] == "?"
    reader.unreadable.clear()  # it becomes readable
    game.update_after_word(((0, 0), (0, 1), (0, 2)))
    assert game.board[(3, 3)] == "Z"


def test_misread_is_corrected_after_next_word(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda p="": "")
    screen = FakeScreen("CATS ODOG RENT SXQZ")
    # The Z (bottom right) is confidently misread as K at the start.
    game = make_game(["--max-words", "1"], screen, FakeReader(screen, misread_once={(3, 3): "K"}))
    game.run()
    assert game.board[(3, 3)] == "Z"
    assert "Re-read corrected 1 tile(s): 44 K->Z" in capsys.readouterr().out

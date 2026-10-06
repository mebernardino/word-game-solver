import threading

from test_game import WORDS, FakeReader, FakeScreen
from wordhunt.cli import Game, build_parser
from wordhunt.ocr import TileRead
from wordhunt.rejects import RejectLog
from wordhunt.solver import Board, Solver, Trie

BOARD = "CATS ODOG RENT SXQZ"  # best word TONED, then others


# ------------------------------------------------------------------ the log file

def test_log_parsing_and_threshold(tmp_path):
    path = tmp_path / "rejected.txt"
    path.write_text("# comment\nQOPH 2\nza 1\nXYST\n\n  # indented comment\n")
    log = RejectLog(str(path), threshold=2)
    assert log.counts == {"QOPH": 2, "ZA": 1, "XYST": 2}  # a bare word is blocked outright
    assert log.blocked == {"QOPH", "XYST"}
    assert RejectLog(str(path), threshold=0).blocked == set()


def test_pending_only_saved_once_confirmed(tmp_path):
    path = tmp_path / "rejected.txt"
    log = RejectLog(str(path), threshold=2)
    log.suspect("ZA")
    log.discard_pending()
    assert not path.exists()

    log.suspect("ZA")
    assert log.confirm_pending() == []           # 1 game so far: not blocked yet
    log.suspect("ZA")
    assert log.confirm_pending() == ["ZA"]       # 2nd game: now blocked
    assert RejectLog(str(path)).counts == {"ZA": 2}
    assert "ZA 2" in path.read_text()


# ------------------------------------------------------------------ the game loop

class RejectingDragger:
    """Like the real game: refills tiles for accepted words, leaves rejected ones alone."""

    def __init__(self, screen, rejects=(), refuse_everything=False):
        self.screen = screen
        self.rejects = set(rejects)
        self.refuse_everything = refuse_everything
        self.played = []

    def drag(self, path):
        word = "".join(self.screen.board.tile_text(r * 4 + c) for r, c in path)
        self.played.append(word)
        self.screen.before = list(self.screen.board.letters)  # what a slow animation still shows
        if self.refuse_everything or word in self.rejects:
            return
        for cell in path:
            self.screen.board[cell] = self.screen.rng.choice("EEAOTNRDSCG")


class SlowAnimationReader(FakeReader):
    """The first read after a word still shows the old letters."""

    def __init__(self, screen, dragger):
        super().__init__(screen)
        self.dragger = dragger
        self._seen = 0

    def read_cells(self, cells):
        if len(self.dragger.played) != self._seen:  # first look after a word
            self._seen = len(self.dragger.played)
            return {(r, c): TileRead(self.screen.before[r * 4 + c], 95) for r, c in cells}
        return super().read_cells(cells)


def play(tmp_path, dragger_kw, argv=(), reader_cls=None, seed=0):
    screen = FakeScreen(BOARD, seed=seed)
    dragger = RejectingDragger(screen, **dragger_kw)
    reader = reader_cls(screen, dragger) if reader_cls else FakeReader(screen)
    args = build_parser().parse_args(
        ["--countdown", "0", "--post-word-delay", "0", "--retry-delay", "0", "--full-reread-every", "0"]
        + list(argv))
    log = RejectLog(str(tmp_path / "rejected.txt"), threshold=2)
    game = Game(args, Solver(Trie.from_words(WORDS)), reader, dragger,
                threading.Event(), threading.Event(), rejects=log)
    game.run()
    return game, dragger, log


def test_rejected_word_not_scored_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda p="": "")
    game, dragger, log = play(tmp_path, {"rejects": {"TONED"}}, ["--max-words", "3"])
    assert dragger.played[0] == "TONED"
    assert game.rejected == 1 and game.words == 3
    assert game.total == sum(400 if len(w) == 4 else 100 if len(w) == 3 else 800
                             for w in dragger.played[1:])  # TONED's 800 not counted
    assert log.counts == {"TONED": 1}                       # saved after the next accepted word


def test_blocked_after_two_games(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda p="": "")
    for _ in range(2):
        play(tmp_path, {"rejects": {"TONED"}}, ["--max-words", "2"])
    _, dragger, log = play(tmp_path, {"rejects": {"TONED"}}, ["--max-words", "2"])
    assert "TONED" in log.blocked
    assert "TONED" not in dragger.played  # third game never tries it


def test_slow_animation_is_not_a_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda p="": "")
    game, _, log = play(tmp_path, {}, ["--max-words", "2"], reader_cls=SlowAnimationReader)
    assert game.rejected == 0 and game.words == 2
    assert log.counts == {}


def test_nothing_landing_keeps_going_and_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda p="": "")
    game, dragger, log = play(tmp_path, {"refuse_everything": True})
    assert len(dragger.played) > Game.REJECT_WARNING  # doesn't stop early
    assert game.words == 0 and game.total == 0
    assert log.counts == {} and not (tmp_path / "rejected.txt").exists()


def test_default_blocks_after_one_rejection(tmp_path):
    path = tmp_path / "rejected.txt"
    path.write_text("ZA 1\nQOPH 3\n")
    assert RejectLog(str(path)).blocked == {"ZA", "QOPH"}
    log = RejectLog(str(tmp_path / "new.txt"))
    log.suspect("XU")
    assert log.confirm_pending() == ["XU"]  # blocked as soon as it's confirmed


def test_rejections_skip_the_wait_once_timing_is_learned(tmp_path, monkeypatch):
    """After an accepted word shows when new letters appear, a rejected word
    (tiles never change) is detected shortly after that, not after the full
    post-word delay plus a second look."""
    import time

    monkeypatch.setattr("builtins.input", lambda p="": "")
    delay = 0.4
    screen = FakeScreen(BOARD)
    dragger = RejectingDragger(screen, rejects=set(WORDS) - {"TONED"})
    args = build_parser().parse_args(
        ["--countdown", "0", "--post-word-delay", str(delay), "--retry-delay", "0", "--max-words", "1"])
    game = Game(args, Solver(Trie.from_words(WORDS)), FakeReader(screen), dragger,
                threading.Event(), threading.Event())

    game.board = Board.parse(BOARD)
    game.cands.refresh(game.board)
    path = game.cands.paths["TONED"]
    dragger.drag(path)  # accepted: tiles change right away
    game.attempts += 1
    changed, accepted = game.update_after_word(path, time.monotonic())
    assert accepted
    assert game.letters_appear is not None and game.letters_appear < 0.1
    game.cands.mark_played("TONED")
    game.cands.refresh(game.board, changed=changed)

    pick = next(c for c in game.cands.available() if c.word != "TONED")
    dragger.drag(pick.path)  # rejected: nothing changes
    game.attempts += 1
    start = time.monotonic()
    _, accepted = game.update_after_word(pick.path, start)
    took = time.monotonic() - start
    assert not accepted
    assert took < delay  # was delay * 2 (full wait + second look)

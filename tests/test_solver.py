import pytest

from wordhunt.solver import (
    Board, Candidate, CandidateSet, Solver, Trie, parse_scores, score_word, split_tiles,
)
from wordhunt.strategy import Timing, choose

#   C A T S
#   O D O G
#   R E N T
#   S X Q Z
BOARD = "CATS ODOG RENT SXQZ"
PRESENT = {"CAT", "CATS", "COD", "CODE", "CORE", "DOG", "DOGS", "DOTS", "TONE", "TONED", "DONE", "RED"}
ABSENT = {
    "TAT",   # would need to reuse the T
    "CODS",  # no S next to the D
    "NOTE",  # T not adjacent to E
    "HEN", "HENS", "QUIT",
    "AT",    # below min length
}
WORDS = PRESENT | ABSENT


def make_solver(words=WORDS, **kw):
    return Solver(Trie.from_words(words, min_len=kw.pop("min_len", 3)), **kw)


def is_valid_path(board, word, path):
    """Path is in-bounds, adjacent step to step, no reuse, and spells the word."""
    if len(set(path)) != len(path):
        return False
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if max(abs(r1 - r2), abs(c1 - c2)) != 1:
            return False
    spelled = "".join(board.tile_text(r * 4 + c) for r, c in path)
    return spelled == word


# ------------------------------------------------------------------ finding words

def test_finds_known_words_and_only_those():
    board = Board.parse(BOARD)
    found = make_solver().solve(board)
    assert set(found) == PRESENT
    for word, path in found.items():
        assert is_valid_path(board, word, path), word


def test_specific_path():
    found = make_solver().solve(Board.parse(BOARD))
    assert found["CAT"] == ((0, 0), (0, 1), (0, 2))
    assert found["DONE"] == ((1, 1), (1, 2), (2, 2), (2, 1))


def test_min_length_configurable():
    found = make_solver(min_len=5).solve(Board.parse(BOARD))
    assert set(found) == {"TONED"}


def test_qu_tile_contributes_two_letters():
    words = {"QUIT", "QUITE", "QIT"}
    board = Board.parse("QuITE XXXX XXXX XXXX")
    assert board.letters[:4] == ["QU", "I", "T", "E"]
    found = make_solver(words).solve(board)
    assert set(found) == {"QUIT", "QUITE"}
    assert found["QUITE"] == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert score_word("QUITE") == 800  # scored by letters, not tiles


def test_plain_q_tile_is_just_q():
    words = {"QUIT", "QIT", "QAT"}
    board = Board.parse("QITE XXXX XXXX XXXX")
    assert set(make_solver(words).solve(board)) == {"QIT"}


def test_q_and_qu_tiles_on_one_board():
    #  Q  A  T  X
    #  Qu I  T  X
    words = {"QAT", "QUIT", "QUA", "QI", "QUAT", "TAQUI"}
    board = Board.parse("QATX QuITX XXXX XXXX")
    assert board.letters[:8] == ["Q", "A", "T", "X", "QU", "I", "T", "X"]
    found = make_solver(words).solve(board)
    assert set(found) == {"QAT", "QUIT", "QUA", "QUAT", "TAQUI"}  # QI is too short
    assert found["TAQUI"] == ((0, 2), (0, 1), (1, 0), (1, 1))
    assert found["QAT"][0] == (0, 0) and found["QUIT"][0] == (1, 0)


def test_parsing_q_and_qu():
    # 16 letters: every letter is its own tile, even a Q then a U.
    assert Board.parse("QUAT XXXX XXXX XXXX").letters[:4] == ["Q", "U", "A", "T"]
    # 5 letters in a row of 4: the QU must be one tile.
    assert Board.parse("QUATX XXXX XXXX XXXX").letters[:4] == ["QU", "A", "T", "X"]
    # Without spaces, typed "Qu" is preferred as the merged tile.
    assert Board.parse("QUQuATXXXXXXXXXXX").letters[:4] == ["Q", "U", "QU", "A"]
    assert Board.parse("Qu I T E  XXXX XXXX XXXX").letters[:4] == ["QU", "I", "T", "E"]
    with pytest.raises(ValueError):
        Board.parse("ABC")


def test_split_tiles():
    assert split_tiles("QuAT", 3) == ["QU", "A", "T"]
    assert split_tiles("QUAT", 4) == ["Q", "U", "A", "T"]
    with pytest.raises(ValueError):
        split_tiles("QAT", 2)


def test_scoring():
    assert [score_word("x" * n) for n in range(2, 11)] == [0, 100, 400, 800, 1400, 1800, 2200, 2200, 2200]
    table = parse_scores("3:1,4:2")
    assert score_word("ABCDEFG", table) == 2


# ------------------------------------------------------------------ replacement

def test_replacing_tiles_updates_candidates():
    board = Board.parse(BOARD)
    cands = CandidateSet(make_solver())
    cands.refresh(board)
    done_path = cands.paths["DONE"]

    # Play CAT; the game swaps those three tiles for H, E, N.
    used = cands.paths["CAT"]
    for cell, letter in zip(used, "HEN"):
        board[cell] = letter
    cands.refresh(board, changed=used)

    # H E N S
    # O D O G
    # R E N T
    # S X Q Z
    assert set(cands.paths) == {"HEN", "HENS", "DOG", "DOGS", "TONE", "TONED", "DONE", "RED"}
    for word, path in cands.paths.items():
        assert is_valid_path(board, word, path), word
    # DONE's old path avoided the new tiles, so it's kept, even though the
    # fresh DFS reaches D-O-N(0,2)-E(0,1) first.
    assert cands.paths["DONE"] == done_path
    # TONE's old path started on the replaced T, but it still exists via T(2,3).
    assert cands.paths["TONE"][0] == (2, 3)
    assert cands.last_solve_ms >= 0


# ------------------------------------------------------------------ never repeat

def test_never_replays_a_word():
    board = Board.parse(BOARD)
    cands = CandidateSet(make_solver())
    cands.refresh(board)

    played = []
    while True:
        pick = choose(cands.available())
        if pick is None:
            break
        played.append(pick.word)
        cands.mark_played(pick.word)
        cands.refresh(board, changed=[])  # board unchanged (e.g. word rejected / dry run)
    assert sorted(played) == sorted(PRESENT)
    assert len(played) == len(set(played))


def test_played_word_stays_excluded_when_it_reappears():
    board = Board.parse(BOARD)
    cands = CandidateSet(make_solver())
    cands.refresh(board)
    cands.mark_played("TONED")
    used = cands.paths["CAT"]
    for cell, letter in zip(used, "HEN"):
        board[cell] = letter
    cands.refresh(board, changed=used)
    assert "TONED" in cands.paths  # still on the board...
    assert "TONED" not in {c.word for c in cands.available()}  # ...but never offered


# ------------------------------------------------------------------ strategies

def test_score_strategy_picks_highest():
    cands = CandidateSet(make_solver())
    cands.refresh(Board.parse(BOARD))
    assert choose(cands.available(), "score").word == "TONED"


def path_of(n):
    return tuple((0, c) for c in range(min(n, 4))) + tuple((1, c) for c in range(max(0, n - 4)))


def test_rate_strategy_prefers_points_per_second():
    long_word = Candidate("LONGWORD", path_of(8), 1000)
    short_word = Candidate("QUA", path_of(3), 900)
    timing = Timing(segment_time=0.06, overhead=0.3)
    # 1000 / (7*0.06 + 0.3) = 1389/s   vs   900 / (2*0.06 + 0.3) = 2143/s
    assert choose([long_word, short_word], "score").word == "LONGWORD"
    assert choose([long_word, short_word], "rate", timing=timing).word == "QUA"


def test_keep_good_letters_strategy():
    #  E A R S
    #  Z Q X T
    #  J K V B
    #  B B B B
    board = Board.parse("EARS ZQXT JKVB BBBB")
    common = Candidate("EARS", ((0, 0), (0, 1), (0, 2), (0, 3)), 1000)
    hard = Candidate("ZQXT", ((1, 0), (1, 1), (1, 2), (1, 3)), 950)        # within 10%
    hardest = Candidate("JKVB", ((2, 0), (2, 1), (2, 2), (2, 3)), 800)     # outside 10%
    cands = [common, hard, hardest]
    assert choose(cands, "score").word == "EARS"
    assert choose(cands, "keep-good-letters", board=board).word == "ZQXT"


def test_ties_break_to_shorter_path_then_alpha():
    a = Candidate("QUIT", path_of(3), 400)
    b = Candidate("BAIT", path_of(4), 400)
    c = Candidate("ABIT", path_of(4), 400)
    assert choose([b, c, a], "score").word == "QUIT"
    assert choose([b, c], "score").word == "ABIT"


def test_no_candidates():
    assert choose([], "score") is None


def test_unknown_strategy():
    with pytest.raises(ValueError):
        choose([Candidate("CAT", path_of(3), 100)], "nope")

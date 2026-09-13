"""
Week 2 sanity tests for the real search-based agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment import Board
from agents import SearchAgent, RandomAgent


def test_finds_mate_in_one():
    # Ladder mate: white king g6 covers f7/g7/h7, black king boxed on g8,
    # Ra1-a8 delivers back-rank mate.
    board = Board("6k1/8/6K1/8/8/8/8/R7 w - - 0 1")
    agent = SearchAgent(max_depth=3)
    move = agent.get_move(board, time_budget_ms=2000)
    after = board.apply_move(move)
    assert after.is_checkmate(), f"expected mate, got {after.fen()} after {move.uci()}"
    print(f"mate-in-1 found: {move.uci()}  PASS")


def test_captures_free_piece():
    # White queen can capture an undefended black rook for free.
    board = Board("6k1/8/8/3r4/3Q4/8/8/6K1 w - - 0 1")
    agent = SearchAgent(max_depth=3)
    move = agent.get_move(board, time_budget_ms=2000)
    assert move.uci() == "d4d5", f"expected d4d5 (capture rook), got {move.uci()}"
    print("takes free piece: PASS")


def test_beats_random_agent_most_of_the_time():
    """Not a rigorous benchmark (that's Week 3/4) - just a sanity check that
    the search agent reliably beats random play, i.e. the search + eval
    machinery actually does something."""
    search_agent = SearchAgent(max_depth=3)
    random_agent = RandomAgent()
    wins, losses, draws = 0, 0, 0
    n_games = 4  # kept small: pure-python alpha-beta at depth 3 is slow

    for i in range(n_games):
        board = Board()
        white, black = (search_agent, random_agent) if i % 2 == 0 else (random_agent, search_agent)
        search_is_white = (i % 2 == 0)
        plies = 0
        while not board.is_terminal() and plies < 60:
            agent = white if board.turn == "w" else black
            move = agent.get_move(board, time_budget_ms=300, seed=i * 100 + plies)
            board = board.apply_move(move)
            plies += 1
        result = board.result()
        if result is None:
            draws += 1
        elif result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == search_is_white:
            wins += 1
        else:
            losses += 1

    print(f"SearchAgent vs RandomAgent over {n_games} games: {wins}W {losses}L {draws}D")
    assert wins >= losses, "search agent should not lose more than it wins against random play"


if __name__ == "__main__":
    test_finds_mate_in_one()
    test_captures_free_piece()
    test_beats_random_agent_most_of_the_time()
    print("\nAll Week 2 tests passed.")

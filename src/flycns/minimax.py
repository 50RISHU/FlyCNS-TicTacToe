"""Exact minimax solver for tic-tac-toe.

Tic-tac-toe is a solved game: perfect play never loses, and every full game
with optimal play from both sides ends in a draw. This module provides the
fallback policy that makes the circuit-backed agent effectively unbeatable,
independent of the raw activity produced by the biological network.
"""

from functools import lru_cache

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def _winner(board):
    """Return the winning mark, 'draw' for a full board, or None."""
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(board):
        return "draw"
    return None


@lru_cache(maxsize=None)
def _minimax(board_key, turn, me):
    """Recursively score a board position from the perspective of `me`."""
    board = list(board_key)
    result = _winner(board)

    if result == me:
        return 1
    if result == "draw":
        return 0
    if result is not None:
        return -1

    opponent = "O" if turn == "X" else "X"
    scores = []

    for i in range(9):
        if board[i] is None:
            board[i] = turn
            scores.append(_minimax(tuple(board), opponent, me))
            board[i] = None

    return max(scores) if turn == me else min(scores)


def optimal_moves(board, mark):
    """Return the optimal legal moves and their minimax score for the given mark.

    The score indicates whether perfect play from this position leads to a win,
    draw, or loss for `mark`.
    """
    board = list(board)
    opponent = "O" if mark == "X" else "X"

    scored = []
    for i in range(9):
        if board[i] is None:
            board[i] = mark
            score = _minimax(tuple(board), opponent, mark)
            board[i] = None
            scored.append((i, score))

    if not scored:
        raise ValueError("No empty cells left to move into.")

    best_score = max(score for _, score in scored)
    best_moves = [i for i, score in scored if score == best_score]
    return best_moves, best_score
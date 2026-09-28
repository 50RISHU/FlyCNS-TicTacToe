"""Simple tic-tac-toe gameplay loop using the connectome-backed agent.

This module provides a human-playable interface and a lightweight self-play mode
for checking that the trained circuit behaves sensibly across different board
states.
"""

from .agent import FlyBrainAgent

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


def check_winner(board):
    """Return the winning mark, or 'draw' if the board is full, else None."""
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(board):
        return "draw"
    return None


def print_board(board):
    """Render the board to the terminal with numbered cells for empty positions."""
    symbols = [cell if cell else str(i + 1) for i, cell in enumerate(board)]
    rows = [symbols[i : i + 3] for i in range(0, 9, 3)]
    print()
    for row in rows:
        print(" ".join(row))
    print()


def human_move(board):
    """Prompt the user for a legal move and return the selected cell index."""
    while True:
        raw = input("Your move (1-9): ").strip()
        if not raw.isdigit():
            print("Enter a number 1-9.")
            continue
        cell = int(raw) - 1
        if not (0 <= cell <= 8):
            print("Enter a number 1-9.")
            continue
        if board[cell]:
            print("That cell is taken.")
            continue
        return cell


def play_vs_flybrain(human_mark: str = "X"):
    """Play a local human-vs-agent game using the circuit-backed strategy."""
    agent_mark = "O" if human_mark == "X" else "X"
    agent = FlyBrainAgent()

    board = [None] * 9
    turn = "X"

    print("You are playing against a circuit built from the Drosophila")
    print("male-CNS connectome (cb_intrinsic core -> descending outputs).")
    print_board(board)

    while True:
        if turn == human_mark:
            cell = human_move(board)
        else:
            cell, scores = agent.choose_move(board, agent_mark)
            print(f"FlyBrain plays cell {cell + 1} (activation: {scores[cell]:.3f})")

        board[cell] = turn
        print_board(board)

        result = check_winner(board)
        if result == "draw":
            print("Draw.")
            return
        if result:
            print(f"{result} wins.")
            return

        turn = "O" if turn == "X" else "X"


def flybrain_vs_flybrain(verbose: bool = True):
    """Run a two-agent game to sanity-check that the circuit decision logic is stable."""
    agent_x = FlyBrainAgent()
    agent_o = FlyBrainAgent()
    agents = {"X": agent_x, "O": agent_o}

    board = [None] * 9
    turn = "X"

    if verbose:
        print_board(board)

    while True:
        cell, scores = agents[turn].choose_move(board, turn)
        board[cell] = turn

        if verbose:
            print(f"{turn} plays cell {cell + 1} (activation: {scores[cell]:.3f})")
            print_board(board)

        result = check_winner(board)
        if result:
            if verbose:
                print("Draw." if result == "draw" else f"{result} wins.")
            return result

        turn = "O" if turn == "X" else "X"


if __name__ == "__main__":
    play_vs_flybrain()
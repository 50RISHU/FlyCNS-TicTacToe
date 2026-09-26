"""FlyBrainAgent: turns a tic-tac-toe board into a move by driving the
core connectome circuit and reading off which output group ends up most
active among the still-empty cells.
"""
from .circuit import (
    N_CELLS,
    load_core_graph,
    load_core_neurons,
    load_selected_output_neurons,
    build_output_groups,
    build_input_groups,
)
from .simulate import CoreSimulator
from .minimax import optimal_moves


class FlyBrainAgent:
    def __init__(
        self,
        graph_path: str = "data/fly_core.graphml",
        core_neurons_path: str = "data/fly_core_neurons.csv",
        output_neurons_path: str = "data/selected_output_neurons.csv",
        own_drive: float = 1.0,
        opponent_drive: float = -1.0,
        decay: float = 0.7,
        steps: int = 15,
        unbeatable: bool = True,
    ):
        self.graph = load_core_graph(graph_path)
        core_neurons = load_core_neurons(core_neurons_path)
        output_neurons = load_selected_output_neurons(output_neurons_path)

        self.input_groups = build_input_groups(core_neurons, N_CELLS)
        self.output_groups = build_output_groups(output_neurons, N_CELLS)

        self.own_drive = own_drive
        self.opponent_drive = opponent_drive
        self.simulator = CoreSimulator(self.graph, decay=decay, steps=steps)

        # When True, the circuit's activation only breaks ties among
        # moves that are already game-theoretically optimal (see
        # minimax.py) -- it can influence WHICH good move gets played,
        # but it can never cause a loss. Set False to see what the
        # connectome does on its own, with no safety net.
        self.unbeatable = unbeatable

    def choose_move(self, board, mark: str):
        """board: list of length 9, each entry 'X', 'O', or None/''.
        mark: 'X' or 'O' -- which mark this agent is playing.

        Returns (cell_index, scores) where `scores` is {cell: activation}
        for every empty cell (the circuit's raw opinion, for inspection),
        regardless of whether `unbeatable` restricted the final choice.
        """
        opponent = "O" if mark == "X" else "X"

        injection = {}
        for i, cell in enumerate(board):
            if cell == mark:
                value = self.own_drive
            elif cell == opponent:
                value = self.opponent_drive
            else:
                continue
            for body_id in self.input_groups[i]:
                injection[body_id] = value

        activation = self.simulator.run(injection)

        scores = {}
        for i, group in enumerate(self.output_groups):
            if board[i]:
                continue
            values = [activation[bid] for bid in group if bid in activation]
            scores[i] = sum(values) / len(values) if values else float("-inf")

        if not scores:
            raise ValueError("No empty cells left to move into.")

        if self.unbeatable:
            best_moves, _ = optimal_moves(board, mark)
            candidates = {i: scores[i] for i in best_moves}
        else:
            candidates = scores

        best_cell = max(candidates, key=candidates.get)
        return best_cell, scores
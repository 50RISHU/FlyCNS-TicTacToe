"""Agent that converts a tic-tac-toe board state into a legal move.

The agent treats each board cell as a drive site in the saved fly core circuit,
then reads the activity of the corresponding output neurons to choose the most
promising move. A minimax safety layer can be enabled to ensure the final move
remains game-theoretically optimal whenever a safe choice exists.
"""

from .circuit import (
    N_CELLS,
    build_input_groups,
    build_output_groups,
    load_core_graph,
    load_core_neurons,
    load_selected_output_neurons,
)
from .minimax import optimal_moves
from .simulate import CoreSimulator


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
        """Initialize the circuit-backed decision agent."""
        self.graph = load_core_graph(graph_path)
        core_neurons = load_core_neurons(core_neurons_path)
        output_neurons = load_selected_output_neurons(output_neurons_path)

        self.input_groups = build_input_groups(core_neurons, N_CELLS)
        self.output_groups = build_output_groups(output_neurons, N_CELLS)

        self.own_drive = own_drive
        self.opponent_drive = opponent_drive
        self.simulator = CoreSimulator(self.graph, decay=decay, steps=steps)

        # When enabled, the circuit only breaks ties among moves that are already
        # game-theoretically optimal. This keeps the biological signal from
        # producing a losing move while still allowing it to prefer among good
        # options. Setting this to False exposes the raw circuit policy.
        self.unbeatable = unbeatable

    def choose_move(self, board, mark: str):
        """Choose a legal move for the given mark.

        Args:
            board: A 9-slot board with entries 'X', 'O', or None.
            mark: The mark being played by this agent, either 'X' or 'O'.

        Returns:
            A tuple of (cell_index, scores), where scores maps each empty cell to
            the circuit's raw activation estimate for that move.
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
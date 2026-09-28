"""Load the saved core circuit and define the board-to-neuron mapping.

The raw connectome does not contain a native representation of a tic-tac-toe
board. To make the circuit playable, this module defines an engineered input and
output convention:

- Output mapping: selected descending-neuron groups correspond to the nine board
  cells.
- Input mapping: the CB-intrinsic neurons in the core are partitioned into nine
  disjoint groups, one per board cell, so board state can be injected into the
  graph.

The fixed ordering is row-major, with index 0 representing the top-left cell and
index 8 representing the bottom-right cell.
"""

import networkx as nx
import numpy as np
import pandas as pd

N_CELLS = 9


def load_core_graph(path: str = "data/fly_core.graphml") -> nx.DiGraph:
    """Load the saved signed core graph from disk."""
    return nx.read_graphml(path, node_type=int)


def load_core_neurons(path: str = "data/fly_core_neurons.csv") -> pd.DataFrame:
    """Load the neuron annotation table for the core graph."""
    return pd.read_csv(path)


def load_selected_output_neurons(path: str = "data/selected_output_neurons.csv") -> pd.DataFrame:
    """Load the output neuron table chosen for the board mapping."""
    return pd.read_csv(path)


def build_output_groups(output_neurons: pd.DataFrame, n_groups: int = N_CELLS):
    """Return one list of neuron body IDs per output group, ordered by strength.

    Each output group corresponds to a board cell and contains the body IDs of the
    descending neurons assigned to that cell.
    """
    grouped = (
        output_neurons
        .groupby("type")
        .agg(total_weight=("cb_input_weight", "sum"), body_ids=("bodyId", list))
        .sort_values("total_weight", ascending=False)
    )

    if len(grouped) < n_groups:
        raise ValueError(
            f"Only {len(grouped)} output neuron types are available, but "
            f"{n_groups} are required. Re-run `python -m flycns.main` with a "
            "larger OUTPUT_TOP_N / N_OUTPUT_GROUPS to generate enough cell-level "
            "output groups."
        )

    return [list(grouped.iloc[i]["body_ids"]) for i in range(n_groups)]


def build_input_groups(core_neurons: pd.DataFrame, n_groups: int = N_CELLS):
    """Split CB-intrinsic neurons into equal-sized input groups for each board cell."""
    cb_ids = sorted(
        core_neurons.loc[core_neurons["superclass"] == "cb_intrinsic", "bodyId"]
    )

    if len(cb_ids) < n_groups:
        raise ValueError(
            f"Only {len(cb_ids)} cb_intrinsic neurons remain in the core graph, but "
            f"{n_groups} are needed to map one input site per board cell. Re-run "
            "`python -m flycns.main` with a larger UPSTREAM_TOP_N."
        )

    return [list(group) for group in np.array_split(cb_ids, n_groups)]
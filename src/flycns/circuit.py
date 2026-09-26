"""Load the saved core circuit and set up the 3x3 board <-> neuron-group
mapping.

The connectome itself has no "board state" input. CB-intrinsic neurons
in this subgraph don't carry real visual/sensory synapses -- those were
cut off further upstream when the pipeline (main.py) selected the
CB-intrinsic layer in the first place. So to actually play a game we
need an *input* convention, on top of the *output* convention the
pipeline already built:

  - OUTPUT: `select_output_groups(..., n_groups=9)` in the pipeline
    already picked the 9 descending-neuron types most strongly driven
    by the CB core; one type = one board cell.
  - INPUT: this module partitions the same core's CB-intrinsic neurons
    into 9 disjoint groups (by sorted bodyId, evenly split) and treats
    each group as the injection site for one board cell. This is an
    engineering choice to get board state INTO the graph, not a claim
    about real fly sensory wiring.

Both partitions use the same fixed cell order: index 0 = top-left ...
index 8 = bottom-right, row-major -- so a game engine can address a
board as a flat list of 9 and use the same indices here.
"""
import numpy as np
import networkx as nx
import pandas as pd

N_CELLS = 9


def load_core_graph(path: str = "data/fly_core.graphml") -> nx.DiGraph:
    return nx.read_graphml(path, node_type=int)


def load_core_neurons(path: str = "data/fly_core_neurons.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_selected_output_neurons(path: str = "data/selected_output_neurons.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def build_output_groups(output_neurons: pd.DataFrame, n_groups: int = N_CELLS):
    """Return a list of `n_groups` lists of bodyIds, one per output
    neuron *type*, ordered by descending total cb_input_weight (already
    computed by the pipeline) -- group 0 is the type most strongly
    driven by the CB core.
    """
    grouped = (
        output_neurons
        .groupby("type")
        .agg(total_weight=("cb_input_weight", "sum"), body_ids=("bodyId", list))
        .sort_values("total_weight", ascending=False)
    )

    if len(grouped) < n_groups:
        raise ValueError(
            f"Only {len(grouped)} output neuron types available, need "
            f"{n_groups}. Re-run `python -m flycns.main` with a larger "
            "OUTPUT_TOP_N / N_OUTPUT_GROUPS so there are enough distinct "
            "types to fill the board."
        )

    return [list(grouped.iloc[i]["body_ids"]) for i in range(n_groups)]


def build_input_groups(core_neurons: pd.DataFrame, n_groups: int = N_CELLS):
    """Partition the core's cb_intrinsic neurons into `n_groups`
    equal-ish, disjoint groups by sorted bodyId. Group i is the
    injection site used for cell i.
    """
    cb_ids = sorted(
        core_neurons.loc[core_neurons["superclass"] == "cb_intrinsic", "bodyId"]
    )

    if len(cb_ids) < n_groups:
        raise ValueError(
            f"Only {len(cb_ids)} cb_intrinsic neurons survived into the "
            f"core graph, need at least {n_groups} to give each cell its "
            "own input site. Re-run `python -m flycns.main` with a "
            "larger UPSTREAM_TOP_N."
        )

    return [list(group) for group in np.array_split(cb_ids, n_groups)]
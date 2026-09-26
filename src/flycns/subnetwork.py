"""Selection and filtering utilities for building the CB-intrinsic ->
descending-output sub-circuit.

Note on `get_connections` return order: it's (neuron_info, edges) --
see the docstring on `get_connections` in neuprint.py for exactly why.
The two functions below just forward that tuple as-is; don't
"re-fix" the order here without re-checking the actual return order
of fetch_adjacencies() in whatever neuprint-python version is
installed, since that's what caused this to break twice.
"""
import networkx as nx
import pandas as pd

from .neuprint import create_client, get_connections


def get_upstream_network(body_ids):
    """Return (neuron_info, edges) for everything synapsing ONTO body_ids."""
    client = create_client()
    neuron_info, edges = get_connections(client, target=body_ids)
    return neuron_info, edges


def get_descending_network(body_ids):
    """Return (neuron_info, edges) for everything body_ids synapse ONTO."""
    client = create_client()
    neuron_info, edges = get_connections(client, source=body_ids)
    return neuron_info, edges


def rank_by_weight(edges: pd.DataFrame, group_col: str, top_n: int | None = None) -> pd.Series:
    """Rank body IDs in `group_col` ("bodyId_pre" or "bodyId_post") by
    total synaptic weight, descending. This is the one place the
    "top-N by summed weight" logic lives -- it used to be duplicated
    (with slightly different signatures) across three functions.
    """
    strength = (
        edges
        .groupby(group_col)["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    if top_n is not None:
        strength = strength.head(top_n)

    return strength


def select_ids_by_superclass(neuron_annotations: pd.DataFrame, superclass: str) -> set:
    """Return the set of bodyIds in `neuron_annotations` whose
    `superclass` column matches `superclass`. Returns an empty set
    (rather than raising) if the column isn't present.
    """
    if "superclass" not in neuron_annotations.columns:
        return set()

    return set(
        neuron_annotations.loc[
            neuron_annotations["superclass"] == superclass,
            "bodyId",
        ]
    )


def filter_edges_by_ids(edges: pd.DataFrame, pre_ids=None, post_ids=None) -> pd.DataFrame:
    """Filter `edges` to rows whose bodyId_pre/bodyId_post is in the
    given id sets. Either side can be left as None to skip that filter.
    Replaces the old single-purpose `filter_cb_intrinsic_network`.
    """
    mask = pd.Series(True, index=edges.index)

    if pre_ids is not None:
        mask &= edges["bodyId_pre"].isin(pre_ids)

    if post_ids is not None:
        mask &= edges["bodyId_post"].isin(post_ids)

    return edges[mask].copy()


def group_output_candidates(output_neurons: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        output_neurons
        .groupby("type")
        .agg(
            count=("bodyId", "count"),
            instances=("instance", list),
            body_ids=("bodyId", list),
        )
        .sort_values("count", ascending=False)
    )

    return grouped


def select_output_groups(output_neurons: pd.DataFrame, n_groups: int = 9):
    """Pick the `n_groups` output *types* with the highest total
    CB-intrinsic input (output_neurons must already have a
    'cb_input_weight' column), and return both the group summary and
    the matching neurons. `n_groups=9` maps one output group per
    tic-tac-toe cell.
    """
    grouped = (
        output_neurons
        .groupby("type")
        .agg(
            total_weight=("cb_input_weight", "sum"),
            neuron_count=("bodyId", "count"),
        )
        .sort_values("total_weight", ascending=False)
    )

    selected = grouped.head(n_groups)
    selected_types = selected.index.tolist()

    selected_neurons = output_neurons[
        output_neurons["type"].isin(selected_types)
    ].copy()

    return selected, selected_neurons


def build_output_matrix(cb_edges: pd.DataFrame, cb_neurons: pd.DataFrame, output_neurons: pd.DataFrame) -> pd.DataFrame:
    cb_ids = set(cb_neurons["bodyId"])
    output_ids = set(output_neurons["bodyId"])

    edges = filter_edges_by_ids(cb_edges, pre_ids=cb_ids, post_ids=output_ids)

    matrix = edges.pivot_table(
        index="bodyId_pre",
        columns="bodyId_post",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )

    matrix = matrix.reindex(
        index=sorted(cb_ids),
        columns=sorted(output_ids),
        fill_value=0,
    )

    return matrix


def build_core_subgraph(graph: nx.DiGraph, keep_ids) -> nx.DiGraph:
    """Restrict `graph` to `keep_ids`, then return only its largest
    weakly-connected component.

    This is what connects the CB-intrinsic -> selected-output-group
    selection to the final "core" circuit: previously the core graph
    was rebuilt from an unrelated, unrestricted top-N-by-weight pass
    over ALL upstream neurons, so the carefully curated CB/output
    selection never actually fed into the saved core network.
    """
    sub = graph.subgraph(set(graph.nodes()) & set(keep_ids)).copy()

    if sub.number_of_nodes() == 0:
        return sub

    components = sorted(nx.weakly_connected_components(sub), key=len, reverse=True)

    return sub.subgraph(components[0]).copy()
"""Utilities for selecting and filtering the CB-intrinsic -> output subnetwork."""

import networkx as nx
import pandas as pd

from .neuprint import create_client, get_connections


def get_upstream_network(body_ids):
    """Return (neuron_info, edges) for all neurons synapsing onto the supplied IDs."""
    client = create_client()
    neuron_info, edges = get_connections(client, target=body_ids)
    return neuron_info, edges


def get_descending_network(body_ids):
    """Return (neuron_info, edges) for all neurons targeted by the supplied IDs."""
    client = create_client()
    neuron_info, edges = get_connections(client, source=body_ids)
    return neuron_info, edges


def rank_by_weight(edges: pd.DataFrame, group_col: str, top_n: int | None = None) -> pd.Series:
    """Rank the values in `group_col` by total synaptic weight in descending order."""
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
    """Return the body IDs whose superclass matches the requested value."""
    if "superclass" not in neuron_annotations.columns:
        return set()

    return set(
        neuron_annotations.loc[
            neuron_annotations["superclass"] == superclass,
            "bodyId",
        ]
    )


def filter_edges_by_ids(edges: pd.DataFrame, pre_ids=None, post_ids=None) -> pd.DataFrame:
    """Return only the edges whose pre- or post-synaptic IDs match the provided sets."""
    mask = pd.Series(True, index=edges.index)

    if pre_ids is not None:
        mask &= edges["bodyId_pre"].isin(pre_ids)

    if post_ids is not None:
        mask &= edges["bodyId_post"].isin(post_ids)

    return edges[mask].copy()


def group_output_candidates(output_neurons: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate output neurons by neuron type and the number of instances."""
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
    """Choose the strongest output types and return both the summary and matching neurons."""
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
    """Build a dense matrix of synaptic weights from central-brain neurons to outputs."""
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
    """Restrict the graph to the selected neurons and keep only its largest component."""
    sub = graph.subgraph(set(graph.nodes()) & set(keep_ids)).copy()

    if sub.number_of_nodes() == 0:
        return sub

    components = sorted(nx.weakly_connected_components(sub), key=len, reverse=True)
    return sub.subgraph(components[0]).copy()
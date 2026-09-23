from .neuprint import create_client, get_connections
import pandas as pd

def get_descending_network(body_ids):
    client = create_client()

    neuron_info, edges = get_connections(client, source=body_ids)

    return neuron_info, edges

def get_upstream_network(body_ids):
    client = create_client()

    neuron_info, edges = get_connections(client, target=body_ids)

    return neuron_info, edges

def filter_upstream_network(edges, top_n=200):
    strength = (
        edges
        .groupby("bodyId_pre")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    selected_ids = strength.head(top_n).index

    filtered_edges = edges[
        edges["bodyId_pre"].isin(selected_ids)
    ].copy()

    return filtered_edges, strength

def get_selected_upstream_neurons(
    edges,
    top_n=200,
):
    strength = (
        edges
        .groupby("bodyId_pre")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    selected_ids = strength.head(top_n).index.tolist()

    return selected_ids, strength
    

def filter_cb_intrinsic_network(
    edges,
    neuron_annotations,
):
    cb_ids = set(
        neuron_annotations.loc[
            neuron_annotations["superclass"]
            == "cb_intrinsic",
            "bodyId",
        ]
    )

    filtered_edges = edges[
        edges["bodyId_pre"].isin(cb_ids)
    ].copy()

    return filtered_edges

def rank_descending_outputs(cb_edges, top_n = 50):
    strength = (
        cb_edges
        .groupby("bodyId_post")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    return strength.head(top_n)

def group_output_candidates(output_neurons):
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

def score_output_groups(cb_edges, output_neurons):
    
    output_ids = set(output_neurons["bodyId"])

    scores = cb_edges[cb_edges["bodyId_post"].isin(output_ids)].groupby("bodyId_post")["weight"].sum().sort_values(ascending=False)


    return scores


def select_output_groups(
    output_neurons,
    n_groups=9,
):
    grouped = (
        output_neurons
        .groupby("type")
        .agg(
            total_weight=("cb_input_weight", "sum"),
            neuron_count=("bodyId", "count"),
        )
        .sort_values(
            "total_weight",
            ascending=False,
        )
    )

    selected = grouped.head(n_groups)

    selected_types = selected.index.tolist()

    selected_neurons = output_neurons[
        output_neurons["type"].isin(selected_types)
    ].copy()

    return selected, selected_neurons



def build_output_matrix(
    cb_edges,
    cb_neurons,
    output_neurons,
):
    cb_ids = set(cb_neurons["bodyId"])
    output_ids = set(output_neurons["bodyId"])

    edges = cb_edges[
        cb_edges["bodyId_pre"].isin(cb_ids)
        & cb_edges["bodyId_post"].isin(output_ids)
    ]

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






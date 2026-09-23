import networkx as nx

from .graph import build_graph
from .explore import get_descending_neurons, get_neurons_by_ids
from .analyze import analyze_core

from .subnetwork import (
    get_upstream_network,
    filter_upstream_network,
    get_selected_upstream_neurons,
    filter_cb_intrinsic_network,
    rank_descending_outputs,
    group_output_candidates,
    score_output_groups,
    select_output_groups,
    build_output_matrix,
)


def main():
    # --------------------------------------------------
    # 1. Get descending neurons
    # --------------------------------------------------

    neurons, _ = get_descending_neurons()

    print("Descending neurons:", len(neurons))

    body_ids = neurons["bodyId"].tolist()

    # --------------------------------------------------
    # 2. Get upstream network
    # --------------------------------------------------

    neuron_info, edges = get_upstream_network(
        body_ids
    )

    print("\n=== ORIGINAL NETWORK ===")

    print("Connections:", len(edges))

    print(
        "Upstream neurons:",
        edges["bodyId_pre"].nunique(),
    )

    # --------------------------------------------------
    # 3. Select strongest upstream neurons
    # --------------------------------------------------

    selected_upstream_ids, strength = (
        get_selected_upstream_neurons(
            edges,
            top_n=200,
        )
    )

    selected_neurons, _ = get_neurons_by_ids(
        selected_upstream_ids
    )

    cb_edges = filter_cb_intrinsic_network(
        edges,
        selected_neurons,
    )

    print("\n=== CB_INTRINSIC NETWORK ===")

    print(
        "CB intrinsic neurons:",
        selected_neurons[
            selected_neurons["superclass"]
            == "cb_intrinsic"
        ]["bodyId"].nunique(),
    )

    print(
        "Descending neurons reached:",
        cb_edges["bodyId_post"].nunique(),
    )

    print(
        "Connections:",
        len(cb_edges),
    )

    cb_graph = build_graph(cb_edges)

    output_strength = rank_descending_outputs(
        cb_edges,
        top_n=50,
    )

    output_ids = output_strength.index.tolist()

    output_neurons, _ = get_neurons_by_ids(
        output_ids
    )

    output_scores = score_output_groups(
        cb_edges,
        output_neurons,
    )

    output_neurons = output_neurons.copy()

    output_neurons["cb_input_weight"] = (
        output_neurons["bodyId"]
        .map(output_scores)
        .fillna(0)
    )

    output_neurons = output_neurons.sort_values(
        "cb_input_weight",
        ascending=False,
    )

    print("\n=== OUTPUT NEURONS BY CB INPUT ===")

    print(
        output_neurons[
            [
                "bodyId",
                "type",
                "instance",
                "cb_input_weight",
            ]
        ].to_string(index=False)
    )

    selected_groups, selected_outputs = (
        select_output_groups(
            output_neurons,
            n_groups=9,
        )
    )

    print("\n=== SELECTED OUTPUT GROUPS ===")

    print(
        selected_groups.to_string()
    )

    print("\n=== SELECTED OUTPUT NEURONS ===")

    print(
        selected_outputs[
            [
                "bodyId",
                "type",
                "instance",
                "cb_input_weight",
            ]
        ]
        .sort_values("type")
        .to_string(index=False)
    )

    selected_outputs.to_csv(
        "data/selected_output_neurons.csv",
        index=False,
    )

    selected_groups.to_csv(
        "data/selected_output_groups.csv"
    )

    output_groups = group_output_candidates(
        output_neurons
    )

    output_groups.to_csv(
        "data/output_groups.csv"
    )

    print("\n=== OUTPUT GROUPS ===")
    print(output_groups.to_string())

    print("\n=== OUTPUT NEURONS ===")

    print(
        output_neurons[
            ["bodyId", "type", "instance"]
        ].to_string(index=False)
    )


    cb_neurons = selected_neurons[
        selected_neurons["superclass"] == "cb_intrinsic"
    ].copy()

    output_matrix = build_output_matrix(
        cb_edges,
        cb_neurons,
        selected_outputs,
    )

    print("\n=== CB → OUTPUT MATRIX ===")

    print(
        "CB neurons:",
        output_matrix.shape[0],
    )

    print(
        "Output neurons:",
        output_matrix.shape[1],
    )

    print(
        "Non-zero connections:",
        (output_matrix > 0).sum().sum(),
    )

    print("\nMatrix:")
    print(output_matrix)

    output_matrix.to_csv(
        "data/cb_output_matrix.csv"
    )


    output_neurons.to_csv(
        "data/candidate_output_neurons.csv",
        index=False,
    )

    print("\n=== TOP DESCENDING OUTPUTS ===")
    print(output_strength)

    print("\n=== CB GRAPH ===")

    print(
        "Nodes:",
        cb_graph.number_of_nodes(),
    )

    print(
        "Edges:",
        cb_graph.number_of_edges(),
    )

    nx.write_graphml(
        cb_graph,
        "data/cb_intrinsic_to_descending.graphml",
    )

    cb_edges.to_csv(
        "data/cb_intrinsic_to_descending.csv",
        index=False,
    )

    print("\n=== UPSTREAM SUPERCLASSES ===")

    if "superclass" in selected_neurons.columns:
        print(
            selected_neurons["superclass"]
            .value_counts(dropna=False)
        )

    print("\n=== UPSTREAM CLASSES ===")

    if "class" in selected_neurons.columns:
        print(
            selected_neurons["class"]
            .value_counts(dropna=False)
        )

    selected_neurons.to_csv(
        "data/selected_upstream_annotations.csv",
        index=False,
    )

    print("\n=== SELECTED UPSTREAM NEURONS ===")

    print(
        "Selected upstream neurons:",
        len(selected_upstream_ids),
    )

    # --------------------------------------------------
    # 4. Inspect selected upstream neurons
    # --------------------------------------------------

    upstream_neurons = neuron_info[
        neuron_info["bodyId"].isin(
            selected_upstream_ids
        )
    ].copy()

    print("\nUpstream neuron columns:")
    print(upstream_neurons.columns.tolist())

    print("\nFirst 20 upstream neurons:")

    print(
        upstream_neurons[
            ["bodyId", "type", "instance"]
        ]
        .head(20)
        .to_string(index=False)
    )

    # Save upstream neurons

    upstream_neurons.to_csv(
        "data/selected_upstream_neurons.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. Filter network
    # --------------------------------------------------

    filtered_edges, strength = (
        filter_upstream_network(
            edges,
            top_n=200,
        )
    )

    print("\n=== FILTERED NETWORK ===")

    print(
        "Selected upstream neurons:",
        filtered_edges["bodyId_pre"].nunique(),
    )

    print(
        "Descending neurons reached:",
        filtered_edges["bodyId_post"].nunique(),
    )

    print(
        "Connections:",
        len(filtered_edges),
    )

    print("\nTop 20 upstream neurons:")

    print(
        strength.head(20)
    )

    # Save filtered network

    filtered_edges.to_csv(
        "data/upstream_filtered.csv",
        index=False,
    )

    # --------------------------------------------------
    # 6. Build graph
    # --------------------------------------------------

    graph = build_graph(
        filtered_edges
    )

    print("\n=== GRAPH INFO ===")

    print(
        "Nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Edges:",
        graph.number_of_edges(),
    )

    # --------------------------------------------------
    # 7. Find connected components
    # --------------------------------------------------

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    components.sort(
        key=len,
        reverse=True,
    )

    print("\n=== LARGEST COMPONENTS ===")

    for i, component in enumerate(
        components[:10],
        start=1,
    ):
        print(
            f"{i}: {len(component)} neurons"
        )

    # --------------------------------------------------
    # 8. Select largest component
    # --------------------------------------------------

    largest_component = components[0]

    core_graph = graph.subgraph(
        largest_component
    ).copy()

    print("\n=== CORE GRAPH INFO ===")

    print(
        "Core neurons:",
        core_graph.number_of_nodes(),
    )

    print(
        "Core connections:",
        core_graph.number_of_edges(),
    )

    # --------------------------------------------------
    # 9. Save core graph
    # --------------------------------------------------

    nx.write_graphml(
        core_graph,
        "data/fly_core.graphml",
    )

    # --------------------------------------------------
    # 10. Get core neuron information
    # --------------------------------------------------

    core_ids = set(
        core_graph.nodes()
    )

    core_neurons = neuron_info[
        neuron_info["bodyId"].isin(
            core_ids
        )
    ].copy()

    core_neurons.to_csv(
        "data/fly_core_neurons.csv",
        index=False,
    )

    # --------------------------------------------------
    # 11. Analyze core
    # --------------------------------------------------

    analyze_core(
        core_neurons
    )

    print("\n=== CORE TYPES ===")

    print(
        core_neurons[
            [
                "bodyId",
                "type",
                "instance",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nFiles saved:")
    print("data/selected_upstream_neurons.csv")
    print("data/upstream_filtered.csv")
    print("data/fly_core.graphml")
    print("data/fly_core_neurons.csv")


if __name__ == "__main__":
    main()
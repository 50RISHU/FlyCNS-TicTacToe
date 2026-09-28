"""Build the core fly-brain circuit and export the files needed by the game agent."""

import networkx as nx
import pandas as pd

from .analyze import analyze_core, report_cb_intrinsic_fraction
from .explore import get_descending_neurons, get_neurons_by_ids
from .graph import build_graph
from .signs import neuron_signs
from .subnetwork import (
    build_core_subgraph,
    build_output_matrix,
    filter_edges_by_ids,
    get_upstream_network,
    group_output_candidates,
    rank_by_weight,
    select_ids_by_superclass,
    select_output_groups,
)

UPSTREAM_TOP_N = 200
OUTPUT_TOP_N = 50
N_OUTPUT_GROUPS = 9  # One output group per tic-tac-toe cell


def main():
    """Construct the CB-intrinsic subgraph and save the network used by the game."""
    # 1. Retrieve the descending-neuron population that serves as the output layer.
    neurons, _ = get_descending_neurons()
    print("Descending neurons:", len(neurons))
    body_ids = neurons["bodyId"].tolist()

    # 2. Fetch the full upstream connectivity for those descending neurons.
    neuron_info, edges = get_upstream_network(body_ids)
    print("\n=== ORIGINAL NETWORK ===")
    print("Connections:", len(edges))
    print("Upstream neurons:", edges["bodyId_pre"].nunique())

    # 3. Keep the strongest upstream inputs and restrict them to the central-brain layer.
    upstream_strength = rank_by_weight(edges, "bodyId_pre", top_n=UPSTREAM_TOP_N)
    selected_upstream_ids = upstream_strength.index.tolist()

    selected_neurons, _ = get_neurons_by_ids(selected_upstream_ids)

    print("\n=== SELECTED UPSTREAM NEURONS ===")
    print("Selected upstream neurons:", len(selected_upstream_ids))

    if "superclass" in selected_neurons.columns:
        print("\n=== UPSTREAM SUPERCLASSES ===")
        print(selected_neurons["superclass"].value_counts(dropna=False))

    report_cb_intrinsic_fraction(selected_neurons)
    selected_neurons.to_csv("data/selected_upstream_annotations.csv", index=False)

    cb_ids = select_ids_by_superclass(selected_neurons, "cb_intrinsic")
    cb_edges = filter_edges_by_ids(edges, pre_ids=cb_ids)

    print("\n=== CB_INTRINSIC NETWORK ===")
    print("CB intrinsic neurons:", len(cb_ids))
    print("Descending neurons reached:", cb_edges["bodyId_post"].nunique())
    print("Connections:", len(cb_edges))

    # 4. Rank candidate output neurons by CB-intrinsic input and fetch their metadata.
    output_strength = rank_by_weight(cb_edges, "bodyId_post", top_n=OUTPUT_TOP_N)
    output_ids = output_strength.index.tolist()

    output_neurons, _ = get_neurons_by_ids(output_ids)
    output_neurons = output_neurons.copy()
    output_neurons["cb_input_weight"] = (
        output_neurons["bodyId"].map(output_strength).fillna(0)
    )
    output_neurons = output_neurons.sort_values("cb_input_weight", ascending=False)

    print("\n=== OUTPUT NEURONS BY CB INPUT ===")
    print(
        output_neurons[["bodyId", "type", "instance", "cb_input_weight"]]
        .to_string(index=False)
    )

    output_groups = group_output_candidates(output_neurons)
    output_groups.to_csv("data/output_groups.csv")

    print("\n=== OUTPUT GROUPS ===")
    print(output_groups.to_string())

    output_neurons.to_csv("data/candidate_output_neurons.csv", index=False)

    # 5. Choose the output groups that map to the 3x3 board.
    selected_groups, selected_outputs = select_output_groups(
        output_neurons, n_groups=N_OUTPUT_GROUPS
    )

    print("\n=== SELECTED OUTPUT GROUPS ===")
    print(selected_groups.to_string())

    print("\n=== SELECTED OUTPUT NEURONS ===")
    print(
        selected_outputs[["bodyId", "type", "instance", "cb_input_weight"]]
        .sort_values("type")
        .to_string(index=False)
    )

    selected_outputs.to_csv("data/selected_output_neurons.csv", index=False)
    selected_groups.to_csv("data/selected_output_groups.csv")

    # 6. Build the CB-to-output matrix used to inspect the selected wiring pattern.
    cb_neurons = selected_neurons[selected_neurons["bodyId"].isin(cb_ids)].copy()
    output_matrix = build_output_matrix(cb_edges, cb_neurons, selected_outputs)

    print("\n=== CB -> OUTPUT MATRIX ===")
    print("CB neurons:", output_matrix.shape[0])
    print("Output neurons:", output_matrix.shape[1])
    print("Non-zero connections:", (output_matrix > 0).sum().sum())
    print("\nMatrix:")
    print(output_matrix)

    output_matrix.to_csv("data/cb_output_matrix.csv")
    cb_edges.to_csv("data/cb_intrinsic_to_descending.csv", index=False)

    # 7. Build the connected core circuit from the curated CB and output neuron sets.
    # The original pipeline risked discarding this selection by rebuilding from an
    # unrelated unrestricted top-N pass. This step keeps the final circuit aligned
    # with the board-specific selection process.
    full_graph = build_graph(cb_edges)
    keep_ids = cb_ids | set(selected_outputs["bodyId"])
    core_graph = build_core_subgraph(full_graph, keep_ids)

    print("\n=== CORE GRAPH INFO ===")
    print("Core neurons:", core_graph.number_of_nodes())
    print("Core connections:", core_graph.number_of_edges())

    core_ids = set(core_graph.nodes())
    core_neurons = pd.concat(
        [
            selected_neurons[selected_neurons["bodyId"].isin(core_ids)],
            output_neurons[output_neurons["bodyId"].isin(core_ids)],
        ]
    ).drop_duplicates(subset="bodyId")

    # 8. Add sign information and save the signed core graph used by the simulator.
    signs = neuron_signs(core_neurons)

    if signs:
        core_edges = cb_edges[
            cb_edges["bodyId_pre"].isin(core_ids)
            & cb_edges["bodyId_post"].isin(core_ids)
        ]
        signed_core_graph = build_graph(core_edges, signs=signs)
    else:
        print(
            "\nNo 'predictedNt' column found on the core neurons -- saving the "
            "core graph without excitatory/inhibitory sign information. Until that "
            "annotation is available, the downstream simulation will treat all edges "
            "as excitatory."
        )
        signed_core_graph = core_graph

    nx.write_graphml(signed_core_graph, "data/fly_core.graphml")
    core_neurons.to_csv("data/fly_core_neurons.csv", index=False)

    # 9. Run a final summary of the saved core.
    analyze_core(core_neurons)

    print("\n=== CORE TYPES ===")
    print(
        core_neurons[["bodyId", "type", "instance"]]
        .head(20)
        .to_string(index=False)
    )

    print("\nFiles saved:")
    for path in [
        "data/selected_upstream_annotations.csv",
        "data/output_groups.csv",
        "data/candidate_output_neurons.csv",
        "data/selected_output_neurons.csv",
        "data/selected_output_groups.csv",
        "data/cb_output_matrix.csv",
        "data/cb_intrinsic_to_descending.csv",
        "data/fly_core.graphml",
        "data/fly_core_neurons.csv",
    ]:
        print(path)


if __name__ == "__main__":
    main()
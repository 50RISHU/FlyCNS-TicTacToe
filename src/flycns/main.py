import networkx as nx
import pandas as pd

from .graph import build_graph
from .explore import get_descending_neurons, get_neurons_by_ids
from .analyze import analyze_core, report_cb_intrinsic_fraction
from .signs import neuron_signs

from .subnetwork import (
    get_upstream_network,
    rank_by_weight,
    select_ids_by_superclass,
    filter_edges_by_ids,
    group_output_candidates,
    select_output_groups,
    build_output_matrix,
    build_core_subgraph,
)

UPSTREAM_TOP_N = 200
OUTPUT_TOP_N = 50
N_OUTPUT_GROUPS = 9  # one candidate output group per tic-tac-toe cell


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
    neuron_info, edges = get_upstream_network(body_ids)

    print("\n=== ORIGINAL NETWORK ===")
    print("Connections:", len(edges))
    print("Upstream neurons:", edges["bodyId_pre"].nunique())

    # --------------------------------------------------
    # 3. Select strongest upstream neurons, restrict to cb_intrinsic
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 4. Rank + fetch candidate output (descending) neurons
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 5. Select the output groups that map onto the board
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 6. CB -> output weight matrix
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 7. Build the connected "core" circuit
    #
    # Previously this was rebuilt from an unrelated, unrestricted
    # top-N-by-weight pass over ALL upstream neurons, so the CB/output
    # selection above never actually fed into the saved core network.
    # Now the core graph comes directly from cb_edges, restricted to
    # the CB-intrinsic + selected-output neurons and their largest
    # connected component.
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 8. Attach neurotransmitter sign (excitatory/inhibitory) and
    #    save the signed core graph.
    # --------------------------------------------------
    signs = neuron_signs(core_neurons)

    if signs:
        core_edges = cb_edges[
            cb_edges["bodyId_pre"].isin(core_ids)
            & cb_edges["bodyId_post"].isin(core_ids)
        ]
        signed_core_graph = build_graph(core_edges, signs=signs)
    else:
        print(
            "\nNo 'predictedNt' column found on the core neurons -- "
            "saving the core graph without excitatory/inhibitory sign "
            "info. Every edge will look excitatory to a downstream "
            "simulation until this is available."
        )
        signed_core_graph = core_graph

    nx.write_graphml(signed_core_graph, "data/fly_core.graphml")
    core_neurons.to_csv("data/fly_core_neurons.csv", index=False)

    # --------------------------------------------------
    # 9. Analyze core
    # --------------------------------------------------
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
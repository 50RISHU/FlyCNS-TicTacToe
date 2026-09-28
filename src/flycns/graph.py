"""Graph-building utilities for the connectome-derived circuit."""

import networkx as nx
import pandas as pd


def build_graph(edges: pd.DataFrame, signs: dict | None = None) -> nx.DiGraph:
    """Construct a directed graph from an edge table.

    The input edge table is expected to contain bodyId_pre, bodyId_post, and
    weight columns. If a sign map is provided, each edge is annotated with the
    presynaptic neuron's sign and a signed weight used for inhibitory/excitatory
    dynamics during simulation.
    """
    graph = nx.DiGraph()

    for edge in edges.itertuples(index=False):
        pre = int(edge.bodyId_pre)
        post = int(edge.bodyId_post)
        weight = float(edge.weight)

        attrs = {"weight": weight}

        if signs is not None:
            sign = signs.get(pre, 0)
            attrs["sign"] = sign
            attrs["signed_weight"] = weight * sign

        graph.add_edge(pre, post, **attrs)

    return graph
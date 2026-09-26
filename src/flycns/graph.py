import networkx as nx
import pandas as pd


def build_graph(edges: pd.DataFrame, signs: dict | None = None) -> nx.DiGraph:
    """Build a directed graph from an edge table with
    bodyId_pre / bodyId_post / weight columns.

    If `signs` is given (a {bodyId: +1/-1/0} map, e.g. from
    `flycns.signs.neuron_signs`), each edge also gets a `sign` and a
    `signed_weight` attribute, where signed_weight = weight * sign of
    the presynaptic neuron. Without this, every edge is implicitly
    treated as excitatory, which is fine for graph analysis but not
    for a simulation that has to pick between competing moves.
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
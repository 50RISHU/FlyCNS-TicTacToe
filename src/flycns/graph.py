import networkx as nx
import pandas as pd


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()

    for edge in edges.itertuples(index=False):
        graph.add_edge(
            int(edge.bodyId_pre),
            int(edge.bodyId_post),
            weight=float(edge.weight),
        )

    return graph
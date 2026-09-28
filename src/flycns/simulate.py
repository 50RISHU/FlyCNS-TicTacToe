"""A lightweight leaky-integrator model over the signed core graph.

This is intentionally simpler than a biological spiking model: it provides a
stable, numerically robust way to transform injected activity at one set of
neurons into a dynamic activation pattern across the network while respecting the
connectome-derived excitatory and inhibitory signs.
"""

import networkx as nx
import numpy as np


class CoreSimulator:
    def __init__(self, graph: nx.DiGraph, decay: float = 0.7, steps: int = 15):
        """Initialize the simulator with a graph and fixed dynamical parameters."""
        self.graph = graph
        self.decay = decay
        self.steps = steps

        self.node_ids = list(graph.nodes())
        self.index = {node: i for i, node in enumerate(self.node_ids)}
        n = len(self.node_ids)

        # A[i, j] represents the signed connection from j to i. We normalize by the
        # largest absolute edge weight so the behavior does not depend on raw
        # synapse-count scale.
        weights = [
            abs(data.get("signed_weight", data.get("weight", 0.0)))
            for _, _, data in graph.edges(data=True)
        ]
        max_weight = max(weights) if weights else 1.0
        max_weight = max_weight or 1.0

        A = np.zeros((n, n), dtype=np.float64)
        for u, v, data in graph.edges(data=True):
            w = data.get("signed_weight", data.get("weight", 0.0))
            A[self.index[v], self.index[u]] += w / max_weight

        self.A = A

    def run(self, injection: dict) -> dict:
        """Run the simulation for a sustained external input.

        Args:
            injection: A mapping of bodyId -> value describing the constant drive
                applied to selected neurons.

        Returns:
            A mapping of bodyId -> final activation for every node in the core graph.
        """
        n = len(self.node_ids)
        drive = np.zeros(n)

        for node, value in injection.items():
            if node in self.index:
                drive[self.index[node]] = value

        a = np.zeros(n)
        for _ in range(self.steps):
            a = self.decay * a + np.tanh(drive + self.A @ a)

        return {node: a[i] for node, i in self.index.items()}
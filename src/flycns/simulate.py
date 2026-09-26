"""A small leaky-integrator simulation over the signed core graph.

This is deliberately NOT a biophysical spiking model -- it's the
simplest numerically-stable way to turn "inject activation at some
neurons" into "read out activation at some other neurons" while still
running through the *real* wiring and *real* excitatory/inhibitory
signs pulled from the connectome (see signs.py). Swap this out for
something like Brian2 later if you want actual spiking dynamics -- the
graph and the I/O convention in circuit.py don't depend on which
simulator reads them.
"""
import numpy as np
import networkx as nx


class CoreSimulator:
    def __init__(self, graph: nx.DiGraph, decay: float = 0.7, steps: int = 15):
        self.graph = graph
        self.decay = decay
        self.steps = steps

        self.node_ids = list(graph.nodes())
        self.index = {node: i for i, node in enumerate(self.node_ids)}
        n = len(self.node_ids)

        # A[i, j] = signed weight of the edge j -> i, scaled by the
        # largest edge weight in the graph so the simulation's behavior
        # doesn't depend on the raw synapse-count scale.
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
        """injection: {bodyId: value}, an external drive held constant
        at every step (a sustained stimulus, not a single pulse).
        Returns {bodyId: final_activation} for every node in the core
        graph.
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
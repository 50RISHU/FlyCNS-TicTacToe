"""Thin wrappers around the NeuPrint client API used by the project."""

from neuprint import Client, fetch_adjacencies, fetch_neurons

from .config import NEUPRINT_DATASET, NEUPRINT_SERVER, NEUPRINT_TOKEN


def create_client() -> Client:
    """Create a configured NeuPrint client for the project dataset."""
    return Client(
        server=NEUPRINT_SERVER,
        dataset=NEUPRINT_DATASET,
        token=NEUPRINT_TOKEN,
    )


def get_neurons(client: Client, criteria):
    """Fetch neurons matching the supplied criteria and return the raw result."""
    neurons, synapses = fetch_neurons(criteria, client=client)
    return neurons, synapses


def get_connections(client: Client, source=None, target=None):
    """Return the NeuPrint adjacency data as (neuron_info, edges).

    The underlying NeuPrint API returns (neurons_df, roi_conn_df). The
    ordering is easy to misread because examples often name the local
    variables as though the edges were returned first. This wrapper keeps
    the API contract explicit and stable for downstream code.
    """
    neuron_info, edges = fetch_adjacencies(source, target, client=client)
    return neuron_info, edges
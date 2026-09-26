from neuprint import Client, fetch_neurons, fetch_adjacencies

from .config import NEUPRINT_SERVER, NEUPRINT_DATASET, NEUPRINT_TOKEN

def create_client() -> Client:
    return Client(server=NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=NEUPRINT_TOKEN)

def get_neurons(client: Client, criteria):
    neurons, synapses = fetch_neurons(criteria, client=client)

    return neurons, synapses

def get_connections(client: Client, source=None, target=None):
    """Return (neuron_info, edges).

    fetch_adjacencies() returns (neurons_df, roi_conn_df) -- neurons
    FIRST, connections SECOND (confirmed directly from the installed
    neuprint-python's docstring: "Returns: Two DataFrames,
    (neurons_df, roi_conn_df)..."). It's easy to get this backwards
    because some example code online names its local variables
    `outgoing_edges, neuron_info = fetch_adjacencies(...)`, which reads
    like "edges first" but is just that example's own (misleading)
    variable names, not the actual return order.
    """
    neuron_info, edges = fetch_adjacencies(source, target, client=client)

    return neuron_info, edges
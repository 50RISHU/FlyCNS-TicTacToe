from neuprint import Client, fetch_neurons, fetch_adjacencies

from .config import NEUPRINT_SERVER, NEUPRINT_DATASET, NEUPRINT_TOKEN

def create_client() -> Client:
    return Client(server=NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=NEUPRINT_TOKEN)

def get_neurons(client: Client, criteria):
    neurons, synapses = fetch_neurons(criteria, client=client)

    return neurons, synapses

def get_connections(client: Client, source=None, target=None):
    edges, neuron_info = fetch_adjacencies(source, target, client=client)

    return edges, neuron_info


"""Helpers for querying the fly connectome by neuron class and identifier."""

from neuprint import NeuronCriteria

from .neuprint import create_client, get_neurons


def get_descending_neurons():
    """Return all descending neurons from the configured NeuPrint dataset."""
    client = create_client()
    criteria = NeuronCriteria(superclass="descending_neuron", client=client)
    neurons, synapses = get_neurons(client, criteria)
    return neurons, synapses


def get_neurons_by_ids(body_ids):
    """Fetch neuron metadata for the specified body IDs."""
    client = create_client()
    criteria = NeuronCriteria(bodyId=body_ids, client=client)
    neurons, synapses = get_neurons(client, criteria)
    return neurons, synapses
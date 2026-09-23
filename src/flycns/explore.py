from neuprint import NeuronCriteria
from .neuprint import create_client, get_neurons


def get_descending_neurons():
    client = create_client()

    criteria = NeuronCriteria(superclass="descending_neuron", client=client)

    neurons, synapses = get_neurons(client, criteria)

    return neurons, synapses

def get_neurons_by_ids(body_ids):
    client = create_client()

    criteria = NeuronCriteria(
        bodyId=body_ids,
        client=client,
    )

    neurons, synapses = get_neurons(
        client,
        criteria,
    )

    return neurons, synapses
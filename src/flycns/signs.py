"""Map predicted neurotransmitters to excitatory or inhibitory signs.

NeuPrint annotations provide a predicted transmitter for each neuron in the
``predictedNt`` column. That information is useful for simulations because the
same synapse count can represent either an excitatory drive or a suppressive one
depending on the presynaptic neuron type.
"""

EXCITATORY = {"acetylcholine"}
INHIBITORY = {"gaba", "glutamate"}
# Dopamine, serotonin, octopamine, tyramine, histamine, "unclear", and None are
# treated as modulatory or unknown and therefore assigned sign 0.


def sign_for_transmitter(transmitter) -> int:
    """Map a single predicted neurotransmitter to +1, -1, or 0."""
    if not isinstance(transmitter, str):
        return 0

    key = transmitter.strip().lower()

    if key in EXCITATORY:
        return 1

    if key in INHIBITORY:
        return -1

    return 0


def neuron_signs(neuron_annotations) -> dict:
    """Build a {bodyId: sign} mapping from a neuron-annotation dataframe.

    If the dataset does not contain a ``predictedNt`` column, the function returns
    an empty dict instead of raising an exception.
    """
    if "predictedNt" not in neuron_annotations.columns:
        return {}

    return {
        int(row.bodyId): sign_for_transmitter(row.predictedNt)
        for row in neuron_annotations.itertuples(index=False)
    }
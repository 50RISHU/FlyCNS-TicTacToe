"""Neurotransmitter -> sign mapping.

male-cns (like MANC and the optic-lobe connectome) annotates each neuron
with a predicted neurotransmitter in the ``predictedNt`` column (Berg et
al. 2025). Nothing else in this pipeline uses that column, which means
every edge produced by ``build_graph`` looks "excitatory" by default
(``weight`` is just a synapse count). That's fine for pure graph
analysis, but useless for a simulation that has to decide between
mutually exclusive moves, where inhibition matters as much as
excitation.

The excitatory/inhibitory split below follows the convention used in
recent fly connectome papers (e.g. Shiu et al. 2023, Lin et al. 2024):
acetylcholine is treated as excitatory, GABA and glutamate as
inhibitory, and the aminergic/peptidergic transmitters are left as
neutral/modulatory (sign 0) since their effect depends on which
receptor the postsynaptic neuron expresses -- information neuprint
does not provide.
"""

EXCITATORY = {"acetylcholine"}
INHIBITORY = {"gaba", "glutamate"}
# dopamine, serotonin, octopamine, tyramine, histamine, "unclear", and
# None are all left as modulatory / unsigned (sign 0) -- see module
# docstring above.


def sign_for_transmitter(transmitter) -> int:
    """Map a single predictedNt value to +1 (excitatory), -1
    (inhibitory), or 0 (modulatory / unknown / missing)."""
    if not isinstance(transmitter, str):
        return 0

    key = transmitter.strip().lower()

    if key in EXCITATORY:
        return 1

    if key in INHIBITORY:
        return -1

    return 0


def neuron_signs(neuron_annotations) -> dict:
    """Build a ``{bodyId: sign}`` map from a neuron annotation dataframe.

    Returns an empty dict (rather than raising) if the dataset doesn't
    expose a ``predictedNt`` column, so callers can treat "no NT data"
    and "all-unsigned" the same way instead of crashing.
    """
    if "predictedNt" not in neuron_annotations.columns:
        return {}

    return {
        int(row.bodyId): sign_for_transmitter(row.predictedNt)
        for row in neuron_annotations.itertuples(index=False)
    }
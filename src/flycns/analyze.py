"""Utilities for summarizing the selected connectome core and its annotations."""

import pandas as pd


def analyze_core(neurons: pd.DataFrame):
    """Print a compact summary of the neuron annotations within the core graph."""
    print("\n === Core Neurons Analysis ===")
    print(f"Total core neurons: {len(neurons)}")

    for column in [
        "type",
        "class",
        "subclass",
        "superclass",
        "hemilineage",
        "predictedNt",
    ]:
        if column not in neurons.columns:
            continue

        print(f"\n === {column.upper()} ===")
        print(neurons[column].value_counts(dropna=False).head(20))


def report_cb_intrinsic_fraction(selected_neurons: pd.DataFrame) -> float:
    """Report how much of the selected upstream set belongs to the central brain.

    The strongest upstream inputs to descending neurons are often not
    cb_intrinsic neurons. This summary makes that imbalance visible early so the
    downstream filtering step does not silently create an empty or undersized
    central-brain core.
    """
    if "superclass" not in selected_neurons.columns or len(selected_neurons) == 0:
        print("\n(no superclass info available to check cb_intrinsic fraction)")
        return 0.0

    counts = selected_neurons["superclass"].value_counts(dropna=False)
    cb_count = int(counts.get("cb_intrinsic", 0))
    fraction = cb_count / len(selected_neurons)

    print(
        f"\ncb_intrinsic neurons among selected upstream set: "
        f"{cb_count}/{len(selected_neurons)} ({fraction:.0%})"
    )
    print(counts.head(10))

    if fraction < 0.2:
        print(
            "Warning: cb_intrinsic neurons are a small fraction of the "
            "selected upstream set. The strongest inputs to these "
            "descending neurons are likely dominated by non-central-brain "
            "cell types (sensory, visual-projection, or ascending). "
            "Consider selecting cb_intrinsic neurons directly by superclass "
            "instead of ranking by total weight first and filtering later."
        )

    return fraction
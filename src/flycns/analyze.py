import pandas as pd


def analyze_core(neurons: pd.DataFrame):
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
    """Print (and return) what fraction of the top-strength upstream
    neurons are actually cb_intrinsic.

    Descending neurons often get a lot of their strongest input
    directly from visual-projection or ascending neurons rather than
    central-brain interneurons, so this number can end up much smaller
    than expected. Printing it here makes that visible instead of
    silently ending up with a tiny (or empty) cb_intrinsic set several
    steps later.
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
            "selected upstream set -- the strongest inputs to these "
            "descending neurons are mostly non-central-brain types "
            "(sensory / visual-projection / ascending). Consider "
            "selecting cb_intrinsic neurons directly (by superclass) "
            "instead of taking top-N by weight first and filtering "
            "afterwards."
        )

    return fraction
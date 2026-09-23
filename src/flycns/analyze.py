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
    ]:
        if column not in neurons.columns:
            continue

        print(f"\n === {column.upper()} ===")
        print(neurons[column].value_counts(dropna=False).head(20))

import pandas as pd
from pathlib import Path


# =========================================================
# Aggregation of experimental runs
# =========================================================
def aggregate_results(folder_path):
    """
    Aggregate multiple CSV runs into a single summary file.

    Expected CSV columns:
        - temperature
        - magnitude
        - fpr
        - accuracy

    Output:
        aggregated_results.csv with mean/std statistics
    """

    folder = Path(folder_path)

    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Processing folder: {folder}")

    # Collect all CSV files in directory
    csv_files = list(folder.glob("*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in: {folder}")

    # -----------------------------------------------------
    # Load and concatenate all runs
    # -----------------------------------------------------
    dfs = []
    for file in csv_files:
        dfs.append(pd.read_csv(file))

    combined_df = pd.concat(dfs, ignore_index=True)

    # Keep only required fields
    combined_df = combined_df[
        ["temperature", "magnitude", "fpr", "accuracy"]
    ]

    # -----------------------------------------------------
    # Aggregate statistics across runs
    # -----------------------------------------------------
    aggregated = (
        combined_df
        .groupby(["temperature", "magnitude"])
        .agg(
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
        .sort_values(["temperature", "magnitude"])
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    output_file = folder / "aggregated_results.csv"
    aggregated.to_csv(output_file, index=False)

    print(f"[INFO] Aggregated results saved to: {output_file}")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":

    folder_path = "results/temp/ViT/swin_tinyimagenet/cmd"

    aggregate_results(folder_path)
import os
import pandas as pd
import sys

METRICS = ["fpr", "auc", "aurc", "accuracy"]


# =========================================================
# Aggregation of experimental runs
# =========================================================
def aggregate_best_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        best_files = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith("_best_fpr.csv")
        ]

        if not best_files:
            continue

        if len(best_files) != 3:
            raise ValueError(
                f"[ERROR] {root} contains {len(best_files)} best files (expected 3)"
            )

        dfs = []
        for f in best_files:
            df = pd.read_csv(f)

            if df.empty:
                raise ValueError(f"[ERROR] Empty file: {f}")

            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)

        missing = [m for m in METRICS if m not in df_all.columns]
        if missing:
            raise ValueError(f"[ERROR] Missing columns {missing} in {root}")

        # compute mean and std, convert to %
        mean_vals = (df_all[METRICS].mean() * 100).round(3)
        std_vals = (df_all[METRICS].std() * 100).round(3)

        # build vertical result
        rows = []
        for metric in METRICS:
            rows.append({
                "metric": f"{metric}_mean",
                "value": mean_vals[metric]
            })
            rows.append({
                "metric": f"{metric}_std",
                "value": std_vals[metric]
            })

        result = pd.DataFrame(rows)

        # save
        output_path = os.path.join(root, "res.csv")
        result.to_csv(output_path, index=False)

        print(f"[OK] Aggregated {root} → {output_path}")


# =========================================================
# Main
# =========================================================
def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate_results.py <root_directory>")
        sys.exit(1)

    root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory")
        sys.exit(1)

    aggregate_best_files(root_dir)


if __name__ == "__main__":
    main()
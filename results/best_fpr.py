import os
import sys
import pandas as pd


def process_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return

    # Normalize column names (remove spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    if "fpr" not in df.columns:
        print(f"[SKIP] {csv_path} (no 'fpr' column found)")
        return

    if df.empty:
        print(f"[SKIP] {csv_path} (empty file)")
        return

    try:
        best_row = df.loc[df["fpr"].idxmin()].to_frame().T
    except Exception as e:
        print(f"[ERROR] Could not compute min FPR in {csv_path}: {e}")
        return

    base, ext = os.path.splitext(csv_path)
    output_path = f"{base}_best_fpr{ext}"

    try:
        best_row.to_csv(output_path, index=False)
        print(f"[OK] {csv_path} → {output_path}")
    except Exception as e:
        print(f"[ERROR] Could not write {output_path}: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python best_fpr_per_csv.py <directory>")
        sys.exit(1)

    root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory")
        sys.exit(1)

    # Recursive scan
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv") and not file.endswith("_best_fpr.csv"):
                csv_path = os.path.join(root, file)
                process_csv(csv_path)


if __name__ == "__main__":
    main()
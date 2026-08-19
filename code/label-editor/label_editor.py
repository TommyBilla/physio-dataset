"""
Interactive error-label editor for exercise feature parquet files.

No CLI args — run it, answer prompts. Converts target parquet to a
throwaway CSV, lets you add binary error-label columns (broadcast same
value to every row, same pattern as `baseline`/`ideal_floor_y`), writes
back to parquet (overwrite), deletes the CSV.
"""

import pandas as pd
from pathlib import Path


def main():
    raw_path = input("Path to .parquet file: ").strip().strip('"')
    parquet_path = Path(raw_path)

    if not parquet_path.exists():
        print(f"[error] file not found: {parquet_path}")
        return
    if parquet_path.suffix.lower() != ".parquet":
        print(f"[error] not a .parquet file: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"[ok] loaded {parquet_path.name}  ({df.shape[0]} rows, {df.shape[1]} cols)")

    csv_path = parquet_path.with_suffix(".tmp_edit.csv")
    df.to_csv(csv_path, index=False)
    print(f"[ok] staged csv: {csv_path.name}")

    print("\nAdd error-label columns. Each is broadcast (same value) to every row.")
    print("Leave column name blank to finish.\n")

    added = {}
    while True:
        col_name = input("Error column name (blank to stop): ").strip()
        if not col_name:
            break

        value_str = input(f"Value for '{col_name}' (0 or 1): ").strip()
        if value_str not in ("0", "1"):
            print("[warn] value must be 0 or 1, skipping this column")
            continue

        value = int(value_str)
        df[col_name] = value
        added[col_name] = value
        print(f"[ok] set {col_name}={value} for all {df.shape[0]} rows\n")

    if not added:
        print("[info] no columns added, nothing to write")
        csv_path.unlink(missing_ok=True)
        return

    df.to_parquet(parquet_path, index=False)
    print(f"\n[ok] overwrote {parquet_path.name}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"[ok] added: {added}")

    csv_path.unlink(missing_ok=True)
    print(f"[ok] disposed of temp csv: {csv_path.name}")


if __name__ == "__main__":
    main()

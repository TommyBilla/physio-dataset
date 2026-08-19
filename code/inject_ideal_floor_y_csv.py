"""
Recursive ideal_floor_y injector for CSV files.

No CLI args — run it, answer prompts. Walks every subfolder from a given
root, sets ideal_floor_y = value (broadcast, same for every row) in every
.csv file found, overwrites in place.
"""

import pandas as pd
from pathlib import Path


def main():
    root = input("Root folder path: ").strip().strip('"')
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        print(f"[error] not a valid folder: {root_path}")
        return

    value_str = input("Value to inject into 'ideal_floor_y': ").strip()
    try:
        value = float(value_str)
    except ValueError:
        print(f"[error] not a valid number: {value_str}")
        return

    files = list(root_path.rglob("*.csv"))
    print(f"[ok] found {len(files)} csv file(s) under {root_path}")

    updated, failed = 0, 0
    for f in files:
        try:
            df = pd.read_csv(f)
            df["ideal_floor_y"] = value
            df.to_csv(f, index=False)
            print(f"[ok] {f}")
            updated += 1
        except Exception as e:
            print(f"[error] {f}: {e}")
            failed += 1

    print(f"\n[done] updated={updated}  failed={failed}  value={value}")


if __name__ == "__main__":
    main()

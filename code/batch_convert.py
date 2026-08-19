"""
Recursive CSV <-> Parquet converter.

No CLI args — run it, answer prompts. Walks every subfolder from a given
root, converts every matching file, writes alongside the original (same
name, new extension). Originals untouched.
"""

import pandas as pd
from pathlib import Path


def main():
    root = input("Root folder path: ").strip().strip('"')
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        print(f"[error] not a valid folder: {root_path}")
        return

    direction = input("Convert 'csv' to parquet, or 'parquet' to csv? [csv/parquet]: ").strip().lower()
    if direction not in ("csv", "parquet"):
        print("[error] must be 'csv' or 'parquet'")
        return

    src_ext = f".{direction}"
    dst_ext = ".parquet" if direction == "csv" else ".csv"

    files = list(root_path.rglob(f"*{src_ext}"))
    print(f"[ok] found {len(files)} {src_ext} file(s) under {root_path}")

    converted, failed = 0, 0
    for f in files:
        out_path = f.with_suffix(dst_ext)
        try:
            if direction == "csv":
                df = pd.read_csv(f)
                df.to_parquet(out_path, index=False)
            else:
                df = pd.read_parquet(f)
                df.to_csv(out_path, index=False)
            print(f"[ok] {f} -> {out_path}")
            converted += 1
        except Exception as e:
            print(f"[error] {f}: {e}")
            failed += 1

    print(f"\n[done] converted={converted}  failed={failed}")


if __name__ == "__main__":
    main()

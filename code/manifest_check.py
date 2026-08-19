#!/usr/bin/env python3
"""
manifest_check.py

Per-parquet integrity/metadata probe for the physio pose pipeline.

Outputs, per file:
    clip_id            file stem
    parquet_path       path as given
    frame_count        max(frame) + 1  (frame is 0-indexed)
    n_rows_parquet     len(df)
    duration_ms        max(timestamp_ms) - min(timestamp_ms)
    mean_visibility    mean of mean_landmark_visibility over all rows
    frame_gap          frame_count - n_rows_parquet  (0 = clean, >0 = dropped frames)

Usage:
    python manifest_check.py file.parquet
    python manifest_check.py dir/ --recursive
    python manifest_check.py dir/ -r -o manifest_check.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

FRAME_COL = "frame"
TIME_COL = "timestamp_ms"
VIS_COL = "mean_landmark_visibility"


def probe(path: Path) -> dict:
    """Read one parquet, return its metadata row. Never raises — errors land in the row."""
    row = {
        "clip_id": path.stem,
        "parquet_path": str(path),
        "frame_count": None,
        "n_rows_parquet": None,
        "duration_ms": None,
        "mean_visibility": None,
        "frame_gap": None,
        "error": "",
    }

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        row["error"] = f"read_failed: {exc}"
        return row

    row["n_rows_parquet"] = len(df)

    if len(df) == 0:
        row["error"] = "empty_file"
        return row

    missing = [c for c in (FRAME_COL, TIME_COL, VIS_COL) if c not in df.columns]
    if missing:
        row["error"] = f"missing_cols: {','.join(missing)}"

    if FRAME_COL in df.columns:
        # frame is 0-indexed, so count = max + 1
        row["frame_count"] = int(df[FRAME_COL].max()) + 1
        row["frame_gap"] = row["frame_count"] - row["n_rows_parquet"]

    if TIME_COL in df.columns:
        row["duration_ms"] = float(df[TIME_COL].max() - df[TIME_COL].min())

    if VIS_COL in df.columns:
        row["mean_visibility"] = float(df[VIS_COL].mean())

    return row


def collect(target: Path, recursive: bool) -> list:
    if target.is_file():
        return [target]
    pattern = "**/*.parquet" if recursive else "*.parquet"
    return sorted(target.glob(pattern))


def main():
    ap = argparse.ArgumentParser(description="Probe parquet files for manifest metadata.")
    ap.add_argument("target", type=Path, help="parquet file or directory")
    ap.add_argument("-r", "--recursive", action="store_true", help="recurse into subdirs")
    ap.add_argument("-o", "--out", type=Path, default=None, help="write results to CSV")
    args = ap.parse_args()

    if not args.target.exists():
        sys.exit(f"not found: {args.target}")

    files = collect(args.target, args.recursive)
    if not files:
        sys.exit(f"no parquet files under: {args.target}")

    out = pd.DataFrame([probe(f) for f in files])

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.to_string(index=False))

    # flag anything worth a human look
    bad = out[(out["error"] != "") | (out["frame_gap"].fillna(0) != 0)]
    if len(bad):
        print(f"\n!! {len(bad)} file(s) need attention (error or frame_gap != 0):")
        print(bad[["clip_id", "frame_count", "n_rows_parquet", "frame_gap", "error"]].to_string(index=False))
    else:
        print(f"\nOK — {len(out)} file(s), no gaps, no errors.")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

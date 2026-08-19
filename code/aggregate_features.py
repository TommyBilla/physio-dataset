#!/usr/bin/env python3
"""
aggregate_features.py

Per-frame parquet  ->  one row per clip.

Why per-clip: labels are broadcast (constant across a clip's frames), the actor
is the same in every video, and adjacent frames are near-duplicates. 150 frames
carry ~1 clip's worth of information, not 150. Aggregating makes the row count
honest and stops the model memorising frame-level noise.

For every numeric feature column, emits: min, max, mean, std, range, first, last.

Leak columns are dropped on purpose (see LEAK_COLS): clip duration separates
good from bad in this dataset, so anything encoding length hands the model a
shortcut that has nothing to do with biomechanics.

Usage:
    python aggregate_features.py manifest.csv -o out/
    python aggregate_features.py manifest.csv -o out/ --path-prefix /new/root
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# never become features
LEAK_COLS = {
    "frame",           # index; encodes position/length
    "timestamp_ms",    # ditto
    "person_id",       # constant
    "phase_normalized",  # derived from clip length
}

# manifest columns that are metadata, not labels
META_COLS = {
    "clip_id", "exercise", "source_recording_id", "subject_id",
    "parquet_path", "video_path", "frame_count", "n_rows_parquet",
    "fps", "duration_in_ms", "duration", "mean_visibility", "folder",
    "label_source", "n_labels_active", "excluded", "excluded_reason",
}

AGGS = ["min", "max", "mean", "std", "range", "first", "last"]

MIN_POSITIVES = 3  # labels below this are unevaluable; dropped with a warning


def clip_features(path: Path) -> pd.Series:
    """Collapse one clip's frames into a single feature row."""
    df = pd.read_parquet(path)

    num = df.select_dtypes(include=[np.number])
    num = num.drop(columns=[c for c in LEAK_COLS if c in num.columns], errors="ignore")

    out = {}
    for col in num.columns:
        s = num[col]
        out[f"{col}__min"] = s.min()
        out[f"{col}__max"] = s.max()
        out[f"{col}__mean"] = s.mean()
        out[f"{col}__std"] = s.std()
        out[f"{col}__range"] = s.max() - s.min()
        out[f"{col}__first"] = s.iloc[0]
        out[f"{col}__last"] = s.iloc[-1]
    return pd.Series(out)


def label_cols(man: pd.DataFrame) -> list:
    """Everything in the manifest that isn't metadata is a label."""
    return [c for c in man.columns if c not in META_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("train_data"))
    ap.add_argument("--path-prefix", default=None,
                    help="replace the manifest's parquet_path root with this")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest).dropna(how="all")
    man = man[pd.to_numeric(man["excluded"], errors="coerce").fillna(0) == 0]

    labels = label_cols(man)
    print(f"{len(man)} clips after exclusions | {len(labels)} label columns\n")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = []

    for ex, sub in man.groupby("exercise"):
        rows, kept_ids = [], []
        for _, r in sub.iterrows():
            p = Path(r["parquet_path"])
            if args.path_prefix:
                p = Path(args.path_prefix) / p.name
            if not p.exists():
                print(f"  !! missing, skipped: {p}")
                continue
            rows.append(clip_features(p))
            kept_ids.append(r["clip_id"])

        if not rows:
            print(f"{ex}: no readable parquets, skipped")
            continue

        X = pd.DataFrame(rows, index=kept_ids)
        # a feature that never varies across clips teaches nothing
        X = X.loc[:, X.std(numeric_only=True).fillna(0) > 0]

        meta = sub.set_index("clip_id").loc[kept_ids]
        y = meta[labels].apply(pd.to_numeric, errors="coerce")

        # keep only labels that apply to this exercise and have enough signal
        pos = y.sum()
        usable = [c for c in labels if pos[c] >= MIN_POSITIVES]
        for c in labels:
            if 0 < pos[c] < MIN_POSITIVES:
                print(f"  {ex}: dropped '{c}' — only {int(pos[c])} positive(s), "
                      f"cannot be trained or evaluated")

        out = pd.concat(
            [meta[["source_recording_id", "subject_id"]], X, y[usable]],
            axis=1,
        )
        out.index.name = "clip_id"

        dest = args.outdir / f"train_{ex}.parquet"
        out.to_parquet(dest)

        summary.append({
            "exercise": ex,
            "clips": len(out),
            "groups": out["source_recording_id"].nunique(),
            "features": X.shape[1],
            "labels": len(usable),
            "min_positives": int(y[usable].sum().min()) if usable else 0,
        })
        print(f"{ex}: {len(out)} clips x {X.shape[1]} features, "
              f"{len(usable)} labels -> {dest}")

    print()
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

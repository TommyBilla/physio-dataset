"""
Mini Squat exercise-specific feature extractor.

Input : common_feature.parquet
Output: mini_squat_feature.parquet

Scope: near-whole-body (errors involve knee alignment, knees-past-toes,
back posture/slouch, ankle/foot position, heel lifting). Drops only
face detail (nose) and wrist landmarks/velocities - keeps everything
else including elbows, shoulders, trunk, pelvis, hips, knees, ankles,
heels, foot index. Raw features only, no error/threshold logic.

Adds one derived col:
- ankle_width_3d: euclidean distance between left/right ankle_rel coords.
  Supports "ankles too close together" error (compare against
  shoulder_width_3d, already in common schema, at train/label time).
"""

import sys
import pandas as pd
import numpy as np

DROP_PREFIXES = (
    "nose_",
    "left_wrist_", "right_wrist_",
)


def select_columns(columns):
    return [c for c in columns if not c.startswith(DROP_PREFIXES)]


def extract(input_path: str, output_path: str):
    df = pd.read_parquet(input_path)

    selected = select_columns(df.columns.tolist())
    out = df[selected].copy()

    ankle_cols = [
        "left_ankle_rel_x", "left_ankle_rel_y", "left_ankle_rel_z",
        "right_ankle_rel_x", "right_ankle_rel_y", "right_ankle_rel_z",
    ]
    if all(c in out.columns for c in ankle_cols):
        dx = out["left_ankle_rel_x"] - out["right_ankle_rel_x"]
        dy = out["left_ankle_rel_y"] - out["right_ankle_rel_y"]
        dz = out["left_ankle_rel_z"] - out["right_ankle_rel_z"]
        out["ankle_width_3d"] = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        missing = [c for c in ankle_cols if c not in out.columns]
        print(f"[warn] cannot compute ankle_width_3d, missing: {missing}")

    out.to_parquet(output_path, index=False)

    dropped = [c for c in df.columns if c not in selected]
    print(f"Input : {input_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Output: {output_path}  ({out.shape[0]} rows, {out.shape[1]} cols)")
    print(f"Dropped ({len(dropped)}): {dropped}")
    print(f"Added: ankle_width_3d")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mini_squat_extractor.py <common_feature.parquet> <mini_squat_feature.parquet>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
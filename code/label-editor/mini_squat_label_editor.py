"""
Mini-squat label editor.

No CLI args — run it, answer prompts.

Does TWO things to the parquet:
1. AUTO: computes `squat_too_shallow` per-row, but only evaluates the
   window around the bottom of the squat (max pelvis_center_y), not the
   whole rep. Outside that window (standing phases) the flag is forced 0,
   so standing frames never get mislabeled as a depth error.
   Window: 1000ms before -> 1000ms after the bottom frame's timestamp_ms.
   Trigger condition inside window: knee_angle > (180 - KNEE_THRESHOLD_DEG)
   AND hip_angle > (180 - HIP_THRESHOLD_DEG)  -> squat_too_shallow = 1
   (uses mean of left/right knee angle and mean of left/right hip angle;
   adjust to max()/min() if you want single-leg sensitivity instead)
2. MANUAL (optional): same broadcast-to-whole-file pattern as the general
   label_editor.py, for any other error type you want to stamp on this
   file. Skip entirely by leaving the column name blank at the prompt.

Converts to a throwaway CSV during editing (per your process), writes
back to parquet (overwrite), deletes the CSV.
"""

import pandas as pd
from pathlib import Path

KNEE_THRESHOLD_DEG = 30  # required flexion from straight (180deg)
HIP_THRESHOLD_DEG = 30
WINDOW_MS = 1500  # +/- around bottom-of-squat frame


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

    required = [
        "pelvis_center_y", "timestamp_ms",
        "left_knee_angle_2d", "right_knee_angle_2d",
        "left_hip_angle_2d", "right_hip_angle_2d",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[error] missing required cols for auto depth-check: {missing}")
        return

    # --- 1. AUTO depth-check, windowed around bottom of squat ---
    bottom_idx = df["pelvis_center_y"].idxmax()
    bottom_ts = df.loc[bottom_idx, "timestamp_ms"]

    knee_angle = (df["left_knee_angle_2d"] + df["right_knee_angle_2d"]) / 2
    hip_angle = (df["left_hip_angle_2d"] + df["right_hip_angle_2d"]) / 2

    knee_shallow = knee_angle > (180 - KNEE_THRESHOLD_DEG)
    hip_shallow = hip_angle > (180 - HIP_THRESHOLD_DEG)

    df["squat_too_shallow"] = 0
    df.loc[bottom_idx, "squat_too_shallow"] = int(knee_shallow[bottom_idx] and hip_shallow[bottom_idx])

    flagged = int(df["squat_too_shallow"].sum())
    print(f"[ok] bottom frame idx={bottom_idx} ts={bottom_ts}ms")
    print(f"[ok] squat_too_shallow=1 on {flagged} row(s)")


    # --- path_deviation: perpendicular distance from top->bottom straight line ---
    top_idx = df["pelvis_center_y"].idxmin()
    bot_idx = df["pelvis_center_y"].idxmax()

    top_x, top_y = df.loc[top_idx, "pelvis_center_x"], df.loc[top_idx, "pelvis_center_y"]
    bot_x, bot_y = df.loc[bot_idx, "pelvis_center_x"], df.loc[bot_idx, "pelvis_center_y"]

    dx = bot_x - top_x
    dy = bot_y - top_y
    line_len = (dx**2 + dy**2) ** 0.5

    if line_len == 0:
        df["path_deviation"] = 0.0
    else:
        # perpendicular distance of each row's pelvis point from the top->bottom line
        df["path_deviation"] = (
            (dy * df["pelvis_center_x"] - dx * df["pelvis_center_y"] + bot_x*top_y - bot_y*top_x).abs()
            / line_len
        )

    PATH_DEVIATION_THRESHOLD = 0.05  # adjust to your unit scale
    df["path_deviation_error"] = (df["path_deviation"] > PATH_DEVIATION_THRESHOLD).astype(int)

    print(f"[ok] path_deviation computed, max={df['path_deviation'].max():.4f}, "
          f"path_deviation_error=1 on {int(df['path_deviation_error'].sum())} row(s)")

    # --- 2. MANUAL optional broadcast labels ---
    csv_path = parquet_path.with_suffix(".tmp_edit.csv")
    df.to_csv(csv_path, index=False)
    print(f"[ok] staged csv: {csv_path.name}")

    print("\nOptional: add other error-label columns (broadcast whole file).")
    print("Leave column name blank to skip/finish.\n")

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

    df.to_parquet(parquet_path, index=False)
    print(f"[ok] overwrote {parquet_path.name}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    if added:
        print(f"[ok] manual labels added: {added}")

    csv_path.unlink(missing_ok=True)
    print(f"[ok] disposed of temp csv: {csv_path.name}")


if __name__ == "__main__":
    main()

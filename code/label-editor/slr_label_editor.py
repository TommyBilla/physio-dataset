"""
Straight Leg Raise (SLR) label editor.

No CLI args — run it, answer prompts.

Only checks the ACTIVE leg (bigger hip_angle_2d range = actively raising).
Inactive leg (resting bent) is skipped entirely for these checks.

AUTO checks, single-point (not per-row — avoids false positives during
ascent/descent, only flags at the true extreme):
- knee_bend_excessive: flag = 1 if knee angle at its most-bent frame is
  < 175 deg. Before checking, occluded frames (detected via thigh_length_3d
  deviating from file median — occlusion/leg-overlap corrupts landmark
  tracking even when the resulting angle looks plausible) are interpolated
  from neighboring reliable frames, so garbage readings don't get picked
  as the "worst" frame.
- hip_angle_exceeded: flag = 1 if raise angle (180 - hip_angle_2d) at the
  peak-raise frame exceeds 46 deg.

MANUAL (optional): same broadcast-to-whole-file pattern as other label
editors, for any other error type on this file. Skip by leaving column
name blank.

Converts to a throwaway CSV during editing, writes back to parquet
(overwrite), deletes the CSV.
"""

import pandas as pd
from pathlib import Path

KNEE_MIN_DEG = 175        # error if knee angle at worst-point < this
HIP_MAX_DEG = 46          # error if raise angle (180-hip_angle) at peak > this
LENGTH_TOLERANCE = 0.03   # thigh_length_3d deviation from median = occlusion signal
_SIDES = ["left", "right"]


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

    # --- detect active leg (bigger hip_angle_2d range = actively raising) ---
    if "left_hip_angle_2d" in df.columns and "right_hip_angle_2d" in df.columns:
        left_range = df["left_hip_angle_2d"].max() - df["left_hip_angle_2d"].min()
        right_range = df["right_hip_angle_2d"].max() - df["right_hip_angle_2d"].min()
        active_side = "left" if left_range > right_range else "right"
        print(f"[ok] active leg detected: {active_side} (range L={left_range:.1f} R={right_range:.1f})")
        check_sides = [active_side]
    else:
        print("[warn] cannot detect active leg, missing hip_angle_2d cols — checking both")
        check_sides = _SIDES

    # --- AUTO checks, active leg only ---
    for side in check_sides:
        knee_col = f"{side}_knee_angle_2d"
        hip_col = f"{side}_hip_angle_2d"
        length_col = f"{side}_thigh_length_3d"

        # knee flexion check, with occlusion interpolation
        if knee_col in df.columns and length_col in df.columns:
            median_len = df[length_col].median()
            reliable = (df[length_col] - median_len).abs() < LENGTH_TOLERANCE
            n_unreliable = int((~reliable).sum())

            if n_unreliable > 0:
                print(f"[occlusion] {side}: {n_unreliable} row(s) unreliable, "
                      f"interpolating from neighbors")
                df.loc[~reliable, knee_col] = None
                df[knee_col] = df[knee_col].interpolate(limit_direction="both")
            else:
                print(f"[ok] {side}: no occlusion detected")

            flag_col = f"{side}_knee_bend_excessive"
            df[flag_col] = (df[knee_col] < KNEE_MIN_DEG).astype(int)
            print(f"[ok] {flag_col}=1 on {int(df[flag_col].sum())} row(s)")
        else:
            print(f"[warn] {knee_col}/{length_col} missing, skipping knee check for {side}")

        # hip raise-angle check, peak-frame only
        if hip_col in df.columns:
            peak_idx = df[hip_col].idxmin()  # min hip_angle_2d = max raise
            peak_raise = 180 - df.loc[peak_idx, hip_col]
            flag_col = f"{side}_hip_angle_exceeded"
            df[flag_col] = 0
            df.loc[peak_idx, flag_col] = int(peak_raise > HIP_MAX_DEG)
            print(f"[ok] {flag_col}: peak_raise={peak_raise:.1f}deg, flag={df.loc[peak_idx, flag_col]}")
        else:
            print(f"[warn] {hip_col} missing, skipping hip check for {side}")

    # --- MANUAL optional broadcast labels ---
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

"""
Seated Knee Extension exercise-specific feature extractor.

Input : common_feature.parquet
Output: seated_knee_extension_feature.parquet

Scope: whole body, both legs (no leg-side filtering - compensation on
either leg matters). Full pass-through of all common columns, plus one
added column:

  - baseline : pelvis_center_y averaged over the ENTIRE parquet file,
               a single fixed value repeated on every row. Acts as the
               seat-plane reference (no floor landmark exists in the
               data - heel dangles, never contacts ground - so hip lift
               off the seat is measured against this whole-file average
               instead of a per-rep or frame-window baseline).

Other signals of interest for this exercise (already present as raw
common features, no derivation needed):
  - knee angle ROM (extension/flexion)      -> *_knee_angle_2d/3d
  - trunk lean / slouch                     -> trunk_left/right_angle_2d
  - hip lifting off seat                    -> *_hip_rel_y, pelvis_center_y (vs baseline)

Raw features only, no error/threshold logic - for downstream annotation
(e.g. CVAT) + model training.
"""

import sys
import pandas as pd


def extract(input_path: str, output_path: str):
    df = pd.read_parquet(input_path)

    if "pelvis_center_y" not in df.columns:
        print("[warn] pelvis_center_y not found - cannot compute baseline")
        baseline = None
    else:
        baseline = df["pelvis_center_y"].mean()

    df["baseline"] = baseline

    if "nose_rel_x" in df.columns and "shoulder_center_x" in df.columns:
        head_x = df["nose_rel_x"] + df["shoulder_center_x"]  # absolute head x-position
        df["rocking_score"] = head_x.std()  # whole-file std-dev, single value repeated every row
    else:
        df["rocking_score"] = None
        print("[warn] cannot compute rocking_score - missing nose_rel_x/shoulder_center_x")

    df.to_parquet(output_path, index=False)

    print(f"Input : {input_path}  ({df.shape[0]} rows, {df.shape[1] - 2} cols)")
    print(f"Output: {output_path}  ({df.shape[0]} rows, {df.shape[1]} cols)  [pass-through + baseline]")
    print(f"baseline (pelvis_center_y, whole-file mean) = {baseline}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python seated_knee_extension_extractor.py <common_feature.parquet> <seated_knee_extension_feature.parquet>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])

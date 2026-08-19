"""
Terminal Knee Extension (TKE) — exercise-specific feature extractor.

Input:  common feature parquet (154-col whole-body schema from common extractor)
Output: trimmed parquet for TKE, raw features only (no error/threshold logic
        baked in — error labeling happens via manual CVAT annotation).

Design notes (confirmed via interview):
- Exercise performed lying down. No wrist/nose/elbow/shoulder-arm relevance,
  but trunk is still tracked (back-movement error deferred to v2, so no
  derived trunk baseline/error col — raw trunk cols kept as passthrough).
- Both legs tracked (compensation on either leg matters).
- Floor/bed reference is NOT computed in this extractor. Patient lies on a
  bed with a fixed floor-plane per actor, so the "ideal" y-reference is
  derived externally by the user from known-good files and manually
  injected into bad files. This script only guarantees an empty
  `ideal_floor_y` column is present in the output (both good and bad runs)
  for that manual injection step. Do NOT auto-populate it.
- Column selection is via an explicit confirmed list (not keyword guessing),
  matching the project convention used in heel_slide_extractor.py.
"""

import pandas as pd
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Explicit confirmed column selection
# ---------------------------------------------------------------------------

IDENTITY_TIME_COLS = [
    "frame", "timestamp_ms", "person_id", "phase_normalized",
]

POSE_QUALITY_COLS = [
    "mean_landmark_visibility", "min_landmark_visibility",
    "low_visibility_landmark_count",
    "mean_landmark_presence", "min_landmark_presence",
]

REFERENCE_SCALE_COLS = [
    "pelvis_center_x", "pelvis_center_y", "pelvis_center_z",
    "shoulder_center_x", "shoulder_center_y", "shoulder_center_z",
    "torso_length_3d", "hip_width_3d", "body_scale",
]

# Lower-body rel coords only (hip/knee/ankle/heel/foot_index), both legs.
_LOWER_LANDMARKS = ["hip", "knee", "ankle", "heel", "foot_index"]
_SIDES = ["left", "right"]
_AXES = ["x", "y", "z"]

REL_COORD_COLS = [
    f"{side}_{landmark}_rel_{axis}"
    for landmark in _LOWER_LANDMARKS
    for side in _SIDES
    for axis in _AXES
]

# Lower-body + trunk angles. No elbow/shoulder(arm) angles.
ANGLE_COLS = (
    [f"{side}_knee_angle_2d" for side in _SIDES]
    + [f"{side}_knee_angle_3d" for side in _SIDES]
    + [f"{side}_hip_angle_2d" for side in _SIDES]
    + [f"{side}_hip_angle_3d" for side in _SIDES]
    + [f"{side}_ankle_angle_2d" for side in _SIDES]
    + [f"{side}_ankle_angle_3d" for side in _SIDES]
    + ["trunk_left_angle_2d", "trunk_right_angle_2d"]
)

# Lower-body segment lengths only (thigh, shank). No upper_arm/forearm/
# shoulder_width. Real col name includes "_length_" (not just "_3d").
SEGMENT_LENGTH_COLS = [
    f"{side}_thigh_length_3d" for side in _SIDES
] + [
    f"{side}_shank_length_3d" for side in _SIDES
]

# Lower-body + pelvis velocities. No elbow/wrist velocities.
_VEL_LANDMARKS = ["knee", "ankle", "heel"]
VELOCITY_COLS = (
    [f"pelvis_center_{axis}_velocity" for axis in _AXES]
    + ["pelvis_center_speed"]
    + [
        f"{side}_{landmark}_{axis}_velocity"
        for landmark in _VEL_LANDMARKS
        for side in _SIDES
        for axis in _AXES
    ]
    + [f"{side}_{landmark}_speed" for landmark in _VEL_LANDMARKS for side in _SIDES]
)

# Angle velocities: knee, hip, ankle (2d only). No elbow/shoulder.
# Real col name order is {side}_{joint}_angle_2d_velocity.
ANGLE_VELOCITY_COLS = [
    f"{side}_{joint}_angle_2d_velocity"
    for joint in ["knee", "hip", "ankle"]
    for side in _SIDES
]

# Raw trunk passthrough is already covered by shoulder_center_* (in
# REFERENCE_SCALE_COLS) and trunk_left/right_angle_2d (in ANGLE_COLS).
# No derived trunk baseline / trunk error column — deferred to v2.

KEEP_COLS = (
    IDENTITY_TIME_COLS
    + POSE_QUALITY_COLS
    + REFERENCE_SCALE_COLS
    + REL_COORD_COLS
    + ANGLE_COLS
    + SEGMENT_LENGTH_COLS
    + VELOCITY_COLS
    + ANGLE_VELOCITY_COLS
)

# Manual-injection placeholder: user fills this in for bad-posture files
# (avg pelvis_center_y across known-good files for the same actor). Left
# empty here for both good and bad output — never auto-populated.
INJECTED_COL = "ideal_floor_y"


def extract(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)

    # Column presence varies by exercise/video — only keep what's actually
    # in this file, don't assume every column always present.
    available = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"[warn] {Path(input_path).name}: {len(missing)} expected cols "
              f"absent, skipping: {missing}", file=sys.stderr)

    out = df[available].copy()
    out[INJECTED_COL] = pd.NA  # empty col — manual injection target
    
    if "pelvis_center_y" in out.columns:
        out["avg_pelvis_center_y"] = out["pelvis_center_y"].mean()
    else:
        out["avg_pelvis_center_y"] = pd.NA
        print(f"[warn] {Path(input_path).name}: pelvis_center_y missing, "
              f"cannot compute avg_pelvis_center_y", file=sys.stderr)

    out.to_parquet(output_path, index=False)
    print(f"[ok] {Path(input_path).name}: {df.shape[1]} -> {out.shape[1]} cols "
          f"(incl. empty '{INJECTED_COL}') -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python terminal_knee_extension_extractor.py <input.parquet> <output.parquet>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])

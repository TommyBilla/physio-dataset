"""
Straight Leg Raise (SLR) — exercise-specific feature extractor.

Input:  common feature parquet (154-col whole-body schema from common extractor)
Output: trimmed parquet for SLR, raw features only (no error/threshold logic
        baked in — error labeling happens via manual CVAT annotation).

Design notes (confirmed via interview):
- Exercise performed lying down, leg raised straight (knee should stay near
  extension, tolerance ~170-180 degrees rather than exact 180).
- Both legs tracked (compensation on either leg matters).
- 4 error signals of interest (labeled downstream in CVAT, not computed
  here): hip raise ROM, knee bending too much (<~170 deg), head raise,
  pelvis/hip hike.
- Upper body (shoulder, elbow — coords/angles 2d+3d/segment lengths/
  velocities) kept in full for head/upper-body compensation tracking.
  Wrist explicitly dropped (not relevant to SLR compensation patterns).
- No head-angle column exists anywhere in the common schema (2d or 3d) —
  this is a gap in the common extractor, not a MediaPipe limitation.
  Raw `nose_rel_x/y/z` used as-is; deriving a new head-angle formula is
  out of scope for this extractor (flagged as a future common-extractor
  addition if head-tilt error becomes important later).
- Floor/bed reference is NOT computed in this extractor. Same
  actor/bed setup as terminal_knee_extension_extractor.py: `pelvis_center_y`
  averaged over the whole file (own-file baseline for good takes), plus an
  empty `ideal_floor_y` column left for manual injection into bad-posture
  files (avg pelvis_center_y from known-good files for the same actor).
  Do NOT auto-populate `ideal_floor_y`.
- Column selection is via an explicit confirmed list (not keyword
  guessing), matching the project convention used throughout.
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

_SIDES = ["left", "right"]
_AXES = ["x", "y", "z"]

# Head: raw position only, no angle col exists in common schema.
HEAD_COL = ["nose_rel_x", "nose_rel_y", "nose_rel_z"]

# Upper body kept in full (shoulder, elbow). Wrist explicitly dropped.
_UPPER_LANDMARKS = ["shoulder", "elbow"]
# Lower body kept in full.
_LOWER_LANDMARKS = ["hip", "knee", "ankle", "heel", "foot_index"]

REL_COORD_COLS = HEAD_COL + [
    f"{side}_{landmark}_rel_{axis}"
    for landmark in (_UPPER_LANDMARKS + _LOWER_LANDMARKS)
    for side in _SIDES
    for axis in _AXES
]

# Angles: upper body (shoulder, elbow) + lower body (knee, hip, ankle),
# all 2d+3d, plus trunk (2d only — no 3d trunk in schema).
_ANGLE_JOINTS = ["shoulder", "elbow", "knee", "hip", "ankle"]
ANGLE_COLS = (
    [f"{side}_{joint}_angle_2d" for joint in _ANGLE_JOINTS for side in _SIDES]
    + [f"{side}_{joint}_angle_3d" for joint in _ANGLE_JOINTS for side in _SIDES]
    + ["trunk_left_angle_2d", "trunk_right_angle_2d"]
)

# Segment lengths: upper_arm (shoulder-elbow) kept, forearm dropped (wrist
# dropped). Lower body kept. shoulder_width_3d is NOT side-prefixed (single
# whole-body col), unlike the per-limb lengths.
SEGMENT_LENGTH_COLS = (
    [f"{side}_upper_arm_length_3d" for side in _SIDES]
    + [f"{side}_thigh_length_3d" for side in _SIDES]
    + [f"{side}_shank_length_3d" for side in _SIDES]
    + ["shoulder_width_3d"]
)

# Velocities: elbow + lower body + pelvis. Wrist velocities dropped.
_VEL_LANDMARKS = ["elbow", "knee", "ankle", "heel"]
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

# Angle velocities: elbow, shoulder, knee, hip, ankle (2d only, real col
# name order is {side}_{joint}_angle_2d_velocity). No wrist angle velocity
# in schema anyway.
ANGLE_VELOCITY_COLS = [
    f"{side}_{joint}_angle_2d_velocity"
    for joint in ["elbow", "shoulder", "knee", "hip", "ankle"]
    for side in _SIDES
]

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

# Own-file floor baseline (good takes) — whole-file avg of pelvis_center_y.
BASELINE_COL = "pelvis_y_avg"


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

    if "pelvis_center_y" in out.columns:
        out[BASELINE_COL] = out["pelvis_center_y"].mean()
    else:
        print(f"[warn] {Path(input_path).name}: pelvis_center_y missing, "
              f"cannot compute {BASELINE_COL}", file=sys.stderr)

    out[INJECTED_COL] = pd.NA  # empty col — manual injection target

    out.to_parquet(output_path, index=False)
    print(f"[ok] {Path(input_path).name}: {df.shape[1]} -> {out.shape[1]} cols "
          f"(incl. '{BASELINE_COL}' + empty '{INJECTED_COL}') -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python straight_leg_raise_extractor.py <input.parquet> <output.parquet>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])

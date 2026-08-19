"""
Heel Slide exercise-specific feature extractor.

Input : common_feature.parquet
Output: heel_slide_feature.parquet

Column list below verified against actual common-extractor output
(154-col schema). Keeps both legs (L+R). No error/threshold logic -
raw features only, for downstream annotation (e.g. CVAT) + model training.
"""

import sys
import pandas as pd

COLUMNS = [
    # identity / time
    "frame", "timestamp_ms", "person_id", "phase_normalized",

    # pose quality
    "mean_landmark_visibility", "min_landmark_visibility",
    "low_visibility_landmark_count",
    "mean_landmark_presence", "min_landmark_presence",

    # reference / scale
    "pelvis_center_x", "pelvis_center_y", "pelvis_center_z",
    "body_scale", "hip_width_3d",

    # relative coords: hip, knee, ankle, heel, foot_index (both legs)
    "left_hip_rel_x", "left_hip_rel_y", "left_hip_rel_z",
    "right_hip_rel_x", "right_hip_rel_y", "right_hip_rel_z",
    "left_knee_rel_x", "left_knee_rel_y", "left_knee_rel_z",
    "right_knee_rel_x", "right_knee_rel_y", "right_knee_rel_z",
    "left_ankle_rel_x", "left_ankle_rel_y", "left_ankle_rel_z",
    "right_ankle_rel_x", "right_ankle_rel_y", "right_ankle_rel_z",
    "left_heel_rel_x", "left_heel_rel_y", "left_heel_rel_z",
    "right_heel_rel_x", "right_heel_rel_y", "right_heel_rel_z",
    "left_foot_index_rel_x", "left_foot_index_rel_y", "left_foot_index_rel_z",
    "right_foot_index_rel_x", "right_foot_index_rel_y", "right_foot_index_rel_z",

    # angles: knee, hip, ankle, trunk (2d + 3d where available)
    "left_knee_angle_2d", "right_knee_angle_2d",
    "left_knee_angle_3d", "right_knee_angle_3d",
    "left_hip_angle_2d", "right_hip_angle_2d",
    "left_hip_angle_3d", "right_hip_angle_3d",
    "left_ankle_angle_2d", "right_ankle_angle_2d",
    "left_ankle_angle_3d", "right_ankle_angle_3d",
    "trunk_left_angle_2d", "trunk_right_angle_2d",

    # segment lengths (scale refs for thresholds later)
    "left_thigh_length_3d", "right_thigh_length_3d",
    "left_shank_length_3d", "right_shank_length_3d",

    # velocities: pelvis, knee, ankle, heel (position) + knee/hip/ankle angle
    "pelvis_center_x_velocity", "pelvis_center_y_velocity",
    "pelvis_center_z_velocity", "pelvis_center_speed",
    "left_knee_x_velocity", "left_knee_y_velocity",
    "left_knee_z_velocity", "left_knee_speed",
    "right_knee_x_velocity", "right_knee_y_velocity",
    "right_knee_z_velocity", "right_knee_speed",
    "left_ankle_x_velocity", "left_ankle_y_velocity",
    "left_ankle_z_velocity", "left_ankle_speed",
    "right_ankle_x_velocity", "right_ankle_y_velocity",
    "right_ankle_z_velocity", "right_ankle_speed",
    "left_heel_x_velocity", "left_heel_y_velocity",
    "left_heel_z_velocity", "left_heel_speed",
    "right_heel_x_velocity", "right_heel_y_velocity",
    "right_heel_z_velocity", "right_heel_speed",
    "left_knee_angle_2d_velocity", "right_knee_angle_2d_velocity",
    "left_hip_angle_2d_velocity", "right_hip_angle_2d_velocity",
    "left_ankle_angle_2d_velocity", "right_ankle_angle_2d_velocity",
]


def extract(input_path: str, output_path: str):
    df = pd.read_parquet(input_path)

    present = [c for c in COLUMNS if c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        print(f"[warn] {len(missing)} expected columns missing from input, skipped:")
        for c in missing:
            print(f"  - {c}")

    out = df[present].copy()
    out.to_parquet(output_path, index=False)

    print(f"\nInput : {input_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Output: {output_path}  ({out.shape[0]} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python heel_slide_extractor.py <common_feature.parquet> <heel_slide_feature.parquet>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
#!/usr/bin/env python3
"""
Common biomechanical feature extractor.

Input:
    Raw MediaPipe landmark CSV or Parquet.

Output:
    Feature-only Parquet with:
      - frame/timestamp/person identity
      - pose quality
      - normalized body geometry
      - common joint angles
      - common segment lengths/distances
      - landmark motion
      - temporal phase

Raw landmarks remain in the original Parquet and are NOT copied into the
feature table.

This module does NOT classify errors and does NOT use legacy heuristic
error flags as features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def p3(df: pd.DataFrame, lm_id: int) -> np.ndarray:
    return df[
        [f"lm{lm_id}_x", f"lm{lm_id}_y", f"lm{lm_id}_z"]
    ].astype(float).to_numpy()


def p2(df: pd.DataFrame, lm_id: int) -> np.ndarray:
    return df[
        [f"lm{lm_id}_x", f"lm{lm_id}_y"]
    ].astype(float).to_numpy()


def add_angle_2d(
    out: dict[str, np.ndarray],
    df: pd.DataFrame,
    a_id: int,
    vertex_id: int,
    c_id: int,
    name: str,
) -> None:
    a = p2(df, a_id)
    b = p2(df, vertex_id)
    c = p2(df, c_id)

    ba = a - b
    bc = c - b

    ba_n = np.linalg.norm(ba, axis=1)
    bc_n = np.linalg.norm(bc, axis=1)
    denom = ba_n * bc_n
    valid = denom > 1e-9

    cosine = np.full(len(df), np.nan)
    cosine[valid] = np.sum(
        ba[valid] * bc[valid],
        axis=1,
    ) / denom[valid]

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.full(len(df), np.nan)
    angle[valid] = np.degrees(np.arccos(cosine[valid]))

    out[name] = angle


def add_angle_3d(
    out: dict[str, np.ndarray],
    df: pd.DataFrame,
    a_id: int,
    vertex_id: int,
    c_id: int,
    name: str,
) -> None:
    a = p3(df, a_id)
    b = p3(df, vertex_id)
    c = p3(df, c_id)

    ba = a - b
    bc = c - b

    ba_n = np.linalg.norm(ba, axis=1)
    bc_n = np.linalg.norm(bc, axis=1)
    denom = ba_n * bc_n
    valid = denom > 1e-9

    cosine = np.full(len(df), np.nan)
    cosine[valid] = np.sum(
        ba[valid] * bc[valid],
        axis=1,
    ) / denom[valid]

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.full(len(df), np.nan)
    angle[valid] = np.degrees(np.arccos(cosine[valid]))

    out[name] = angle


def add_distance_3d(
    out: dict[str, np.ndarray],
    df: pd.DataFrame,
    a_id: int,
    b_id: int,
    name: str,
) -> None:
    d = p3(df, a_id) - p3(df, b_id)
    out[name] = np.linalg.norm(d, axis=1)


def derivative(
    values: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    if len(values) < 2:
        return np.full(len(values), np.nan)

    dt = np.gradient(time_s)
    dt = np.where(dt > 1e-6, dt, np.nan)
    return np.gradient(values) / dt


def add_derivative_features(
    out: dict[str, np.ndarray],
    time_s: np.ndarray,
    source_name: str,
    values: np.ndarray,
    acceleration: bool = True,
) -> None:
    vel = derivative(values, time_s)
    out[f"{source_name}_velocity"] = vel

    if acceleration:
        out[f"{source_name}_acceleration"] = derivative(
            vel,
            time_s,
        )


def extract_common_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build only the shared feature layer.

    No legacy error flags are copied into the training features.
    """

    df = raw.copy()

    df = df.sort_values(
        ["timestamp_ms", "frame"]
    ).reset_index(drop=True)

    out: dict[str, np.ndarray | pd.Series] = {}

    # ------------------------------------------------------------------
    # Identity / temporal information
    # ------------------------------------------------------------------

    out["frame"] = df["frame"].to_numpy()
    out["timestamp_ms"] = df["timestamp_ms"].to_numpy()
    out["person_id"] = df["person_id"].to_numpy()

    time_s = (
        df["timestamp_ms"].astype(float).to_numpy()
        / 1000.0
    )

    if len(df) <= 1:
        phase = np.zeros(len(df))
    else:
        t0 = float(time_s[0])
        t1 = float(time_s[-1])
        if t1 <= t0:
            phase = np.linspace(0.0, 1.0, len(df))
        else:
            phase = (time_s - t0) / (t1 - t0)

    out["phase_normalized"] = phase

    # ------------------------------------------------------------------
    # Pose quality
    # ------------------------------------------------------------------

    visibility_cols = [
        f"lm{i}_visibility"
        for i in range(33)
        if f"lm{i}_visibility" in df.columns
    ]

    presence_cols = [
        f"lm{i}_presence"
        for i in range(33)
        if f"lm{i}_presence" in df.columns
    ]

    if visibility_cols:
        vis = df[visibility_cols].astype(float)
        out["mean_landmark_visibility"] = vis.mean(axis=1).to_numpy()
        out["min_landmark_visibility"] = vis.min(axis=1).to_numpy()
        out["low_visibility_landmark_count"] = (
            (vis < 0.5).sum(axis=1).to_numpy()
        )

    if presence_cols:
        pres = df[presence_cols].astype(float)
        out["mean_landmark_presence"] = pres.mean(axis=1).to_numpy()
        out["min_landmark_presence"] = pres.min(axis=1).to_numpy()

    # ------------------------------------------------------------------
    # Body references
    # ------------------------------------------------------------------

    shoulder_center = (
        p3(df, 11) + p3(df, 12)
    ) / 2.0

    pelvis_center = (
        p3(df, 23) + p3(df, 24)
    ) / 2.0

    out["pelvis_center_x"] = pelvis_center[:, 0]
    out["pelvis_center_y"] = pelvis_center[:, 1]
    out["pelvis_center_z"] = pelvis_center[:, 2]

    out["shoulder_center_x"] = shoulder_center[:, 0]
    out["shoulder_center_y"] = shoulder_center[:, 1]
    out["shoulder_center_z"] = shoulder_center[:, 2]

    torso_vec = shoulder_center - pelvis_center
    torso_length = np.linalg.norm(torso_vec, axis=1)

    hip_vec = p3(df, 23) - p3(df, 24)
    hip_width = np.linalg.norm(hip_vec, axis=1)

    body_scale = np.sqrt(
        np.square(torso_length)
        + np.square(hip_width)
    )

    body_scale = np.where(
        body_scale > 1e-6,
        body_scale,
        np.nan,
    )

    out["torso_length_3d"] = torso_length
    out["hip_width_3d"] = hip_width
    out["body_scale"] = body_scale

    # ------------------------------------------------------------------
    # Body-relative lower-limb coordinates
    # ------------------------------------------------------------------

    body_landmarks = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    for name, lm_id in body_landmarks.items():
        rel = p3(df, lm_id) - pelvis_center
        rel = rel / body_scale[:, None]

        out[f"{name}_rel_x"] = rel[:, 0]
        out[f"{name}_rel_y"] = rel[:, 1]
        out[f"{name}_rel_z"] = rel[:, 2]

    # ------------------------------------------------------------------
    # Common joint angles
    # ------------------------------------------------------------------

    angle_2d_specs = {
        "left_elbow_angle_2d": (11, 13, 15),
        "right_elbow_angle_2d": (12, 14, 16),
        "left_shoulder_angle_2d": (13, 11, 23),
        "right_shoulder_angle_2d": (14, 12, 24),
        "left_knee_angle_2d": (23, 25, 27),
        "right_knee_angle_2d": (24, 26, 28),
        "left_hip_angle_2d": (11, 23, 25),
        "right_hip_angle_2d": (12, 24, 26),
        "left_ankle_angle_2d": (25, 27, 29),
        "right_ankle_angle_2d": (26, 28, 30),
        "trunk_left_angle_2d": (11, 23, 24),
        "trunk_right_angle_2d": (12, 24, 23),
    }

    for name, (a, b, c) in angle_2d_specs.items():
        add_angle_2d(
            out,
            df,
            a,
            b,
            c,
            name,
        )

    angle_3d_specs = {
        "left_elbow_angle_3d": (11, 13, 15),
        "right_elbow_angle_3d": (12, 14, 16),
        "left_shoulder_angle_3d": (13, 11, 23),
        "right_shoulder_angle_3d": (14, 12, 24),
        "left_knee_angle_3d": (23, 25, 27),
        "right_knee_angle_3d": (24, 26, 28),
        "left_hip_angle_3d": (11, 23, 25),
        "right_hip_angle_3d": (12, 24, 26),
        "left_ankle_angle_3d": (25, 27, 29),
        "right_ankle_angle_3d": (26, 28, 30),
    }

    for name, (a, b, c) in angle_3d_specs.items():
        add_angle_3d(
            out,
            df,
            a,
            b,
            c,
            name,
        )

    # ------------------------------------------------------------------
    # Common segment lengths / distances
    # ------------------------------------------------------------------

    distance_specs = {
        "left_upper_arm_length_3d": (11, 13),
        "right_upper_arm_length_3d": (12, 14),
        "left_forearm_length_3d": (13, 15),
        "right_forearm_length_3d": (14, 16),
        "left_thigh_length_3d": (23, 25),
        "right_thigh_length_3d": (24, 26),
        "left_shank_length_3d": (25, 27),
        "right_shank_length_3d": (26, 28),
        "shoulder_width_3d": (11, 12),
    }

    for name, (a, b) in distance_specs.items():
        add_distance_3d(
            out,
            df,
            a,
            b,
            name,
        )

    # ------------------------------------------------------------------
    # Motion of important landmarks
    # ------------------------------------------------------------------

    motion_points = {
        "left_elbow": p3(df, 13),
        "right_elbow": p3(df, 14),
        "left_wrist": p3(df, 15),
        "right_wrist": p3(df, 16),
        "pelvis_center": pelvis_center,
        "left_knee": p3(df, 25),
        "right_knee": p3(df, 26),
        "left_ankle": p3(df, 27),
        "right_ankle": p3(df, 28),
        "left_heel": p3(df, 29),
        "right_heel": p3(df, 30),
    }

    for name, points in motion_points.items():
        for i, axis in enumerate(("x", "y", "z")):
            add_derivative_features(
                out,
                time_s,
                f"{name}_{axis}",
                points[:, i],
                acceleration=False,
            )

        vx = out[f"{name}_x_velocity"]
        vy = out[f"{name}_y_velocity"]
        vz = out[f"{name}_z_velocity"]

        out[f"{name}_speed"] = np.sqrt(
            np.square(vx)
            + np.square(vy)
            + np.square(vz)
        )

    # ------------------------------------------------------------------
    # Angle velocities
    # ------------------------------------------------------------------

    for name in (
        "left_elbow_angle_2d",
        "right_elbow_angle_2d",
        "left_shoulder_angle_2d",
        "right_shoulder_angle_2d",
        "left_knee_angle_2d",
        "right_knee_angle_2d",
        "left_hip_angle_2d",
        "right_hip_angle_2d",
        "left_ankle_angle_2d",
        "right_ankle_angle_2d",
    ):
        add_derivative_features(
            out,
            time_s,
            name,
            np.asarray(out[name], dtype=float),
            acceleration=False,
        )

    result = pd.DataFrame(out)

    # Keep feature columns contiguous and avoid fragmentation.
    result = result.copy()

    return result


def extract_common_file(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    raw = read_table(input_path)
    features = extract_common_features(raw)
    write_parquet(features, output_path)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract common biomechanical features."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    extract_common_file(
        args.input,
        args.output,
    )

    print(
        f"[OK] Common features written to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()

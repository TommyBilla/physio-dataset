"""
wireframe.py — renders 2D skeleton wireframe video from exercise feature parquet.
Usage: python wireframe.py <input.parquet> [output.mp4]

FIX: *_rel_x/_rel_y are OFFSETS from a center, not absolute coords.
Lower body (hip/knee/ankle/heel/foot_index) relative to pelvis_center_x/y.
Upper body (nose/shoulder/elbow/wrist) relative to shoulder_center_x/y.
Absolute = center + rel_offset. Previous version skipped this step —
that's what broke the render.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

CANVAS = 640
PAD = 40

EDGES = [
    ("nose", "shoulder_center"),
    ("shoulder_center", "left_shoulder"), ("shoulder_center", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
]

UPPER_LANDMARKS = {"nose", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_wrist", "right_wrist"}


def absolute_xy(landmark, df):
    if landmark in ("pelvis_center", "shoulder_center"):
        x, y = f"{landmark}_x", f"{landmark}_y"
        if x in df.columns and y in df.columns:
            return df[x], df[y]
        return None
    rx, ry = f"{landmark}_rel_x", f"{landmark}_rel_y"
    if rx not in df.columns or ry not in df.columns:
        return None
    center = "shoulder_center" if landmark in UPPER_LANDMARKS else "pelvis_center"
    cx, cy = f"{center}_x", f"{center}_y"
    if cx not in df.columns or cy not in df.columns:
        return None
    return df[rx] + df[cx], df[ry] + df[cy]


def main():
    if len(sys.argv) < 2:
        print("Usage: python wireframe.py <input.parquet> [output.mp4]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix(".mp4")

    df = pd.read_parquet(in_path)
    print(f"[ok] loaded {in_path.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

    needed = {lm for pair in EDGES for lm in pair}
    resolved = {}
    for lm in needed:
        xy = absolute_xy(lm, df)
        if xy is not None:
            resolved[lm] = xy

    valid_edges = [(a, b) for a, b in EDGES if a in resolved and b in resolved]
    if not valid_edges:
        print("[error] no usable landmark pairs found in this file")
        return
    print(f"[ok] {len(valid_edges)}/{len(EDGES)} skeleton edges available "
          f"(landmarks resolved: {sorted(resolved.keys())})")

    all_x = pd.concat([resolved[lm][0] for lm in resolved], ignore_index=True).dropna()
    all_y = pd.concat([resolved[lm][1] for lm in resolved], ignore_index=True).dropna()
    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = (CANVAS - 2 * PAD) / span

    def to_px(vx, vy):
        px = PAD + (vx - min_x) * scale
        py = CANVAS - (PAD + (vy - min_y) * scale)
        return int(px), int(py)

    fps = 30.0
    if "timestamp_ms" in df.columns and df.shape[0] > 1:
        deltas = df["timestamp_ms"].diff().dropna()
        deltas = deltas[deltas > 0]
        if len(deltas) > 0:
            fps = 1000.0 / deltas.median()
    print(f"[ok] fps={fps:.2f}")

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CANVAS, CANVAS)
    )

    n = df.shape[0]
    for i in range(n):
        frame = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
        for a, b in valid_edges:
            va, vb = resolved[a][0].iloc[i], resolved[a][1].iloc[i]
            vc, vd = resolved[b][0].iloc[i], resolved[b][1].iloc[i]
            if pd.isna(va) or pd.isna(vb) or pd.isna(vc) or pd.isna(vd):
                continue
            p1 = to_px(va, vb)
            p2 = to_px(vc, vd)
            cv2.line(frame, p1, p2, (0, 0, 0), 2)
            cv2.circle(frame, p1, 4, (0, 0, 200), -1)
            cv2.circle(frame, p2, 4, (0, 0, 200), -1)
        writer.write(frame)

    writer.release()
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
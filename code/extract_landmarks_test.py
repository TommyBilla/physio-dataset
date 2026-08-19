#!/usr/bin/env python3

import argparse
import pandas as pd
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
}


# MediaPipe Pose landmark connections.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),

    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),

    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),

    (11, 23), (12, 24), (23, 24),

    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (27, 31),

    (24, 26), (26, 28),
    (28, 30), (30, 32),
    (28, 32),
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def draw_skeleton(frame, landmarks, width, height):
    """
    Draw the MediaPipe pose wireframe.

    IMPORTANT:
    This function is purely visual.
    It does not alter the extracted data.
    """

    if landmarks is None:
        cv2.putText(
            frame,
            "NO POSE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return frame

    points = {}

    for landmark_id, landmark in enumerate(landmarks):

        # Only skip the point visually if its coordinates are invalid.
        # We do NOT permanently disable landmarks based on visibility.
        if landmark.x is None or landmark.y is None:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points[landmark_id] = (x, y)

    # Draw bones.
    for a, b in POSE_CONNECTIONS:

        if a not in points or b not in points:
            continue

        cv2.line(
            frame,
            points[a],
            points[b],
            (0, 255, 0),
            2,
        )

    # Draw joints.
    for x, y in points.values():

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )

    return frame


def create_fieldnames():
    """
    Create the complete raw-landmark CSV schema.

    No derived biomechanical features are included here.

    This is intentional.
    """

    fieldnames = [
        "frame",
        "timestamp_ms",
        "person_id",
    ]

    for landmark_id in range(33):

        fieldnames.extend(
            [
                f"lm{landmark_id}_x",
                f"lm{landmark_id}_y",
                f"lm{landmark_id}_z",
                f"lm{landmark_id}_visibility",
                f"lm{landmark_id}_presence",
            ]
        )

    return fieldnames


def create_metadata(
    video_path,
    fps,
    width,
    height,
    frame_count,
    duration_seconds,
):
    """
    Create video-level metadata.

    This is saved separately from the frame-level CSV.
    """

    return {
        "source_video": str(video_path),
        "source_filename": video_path.name,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "landmark_model": "MediaPipe Pose Landmarker Heavy",
        "landmark_count": 33,
        "num_poses": 1,
        "data_type": "raw_landmarks",
    }


# ---------------------------------------------------------------------------
# Main video processing
# ---------------------------------------------------------------------------

def process_video(
    video_path,
    output_parquet,
    output_video,
    output_metadata,
    landmarker,
):
    """
    Process exactly ONE video.

    Responsibilities of this function:

        video
          ↓
        MediaPipe
          ↓
        raw landmarks
          ↓
        CSV
          +
        wireframe verification video

    It deliberately does NOT:
        - estimate floor
        - calculate angles
        - classify errors
        - detect heel lifting
        - calculate exercise-specific features
    """

    print(f"[INFO] Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        print(
            "[WARNING] Invalid FPS reported by video. "
            "Falling back to 30 FPS."
        )
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    reported_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration_seconds = (
        reported_frame_count / fps
        if reported_frame_count > 0
        else None
    )

    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Resolution: {width}x{height}")
    print(
        f"[INFO] Reported frame count: "
        f"{reported_frame_count}"
    )

    # ---------------------------------------------------------------
    # Prepare output directories
    # ---------------------------------------------------------------

    output_parquet.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Video writer
    # ---------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    video_writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height),
    )

    if not video_writer.isOpened():
        print(
            f"[ERROR] Could not create output video: "
            f"{output_video}"
        )
        cap.release()
        return False

    # ---------------------------------------------------------------
    # CSV preparation
    # ---------------------------------------------------------------

    fieldnames = create_fieldnames()

    rows = []

    frame_number = 0
    previous_timestamp_ms = -1

    # ---------------------------------------------------------------
    # Frame processing
    # ---------------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------------------------
        # Convert OpenCV BGR → RGB for MediaPipe
        # -----------------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # -----------------------------------------------------------
        # Timestamp
        #
        # MediaPipe VIDEO mode requires monotonically increasing
        # timestamps.
        # -----------------------------------------------------------

        timestamp_ms = int(
            (frame_number * 1000) / fps
        )

        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1

        previous_timestamp_ms = timestamp_ms

        # -----------------------------------------------------------
        # MediaPipe inference
        # -----------------------------------------------------------

        try:

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

        except Exception as error:

            print(
                f"[ERROR] MediaPipe failed at "
                f"frame {frame_number}: {error}"
            )

            # Still write the frame to the diagnostic video.
            cv2.putText(
                frame,
                "MEDIAPIPE ERROR",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            video_writer.write(frame)

            # Preserve the frame in the dataset.
            rows.append(
                {
                    "frame": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "person_id": -1,
                }
            )

            frame_number += 1
            continue

        # -----------------------------------------------------------
        # Pose detected
        # -----------------------------------------------------------

        if result.pose_landmarks:

            # num_poses = 1, so normally this is exactly one person.
            landmarks = result.pose_landmarks[0]

            row = {
                "frame": frame_number,
                "timestamp_ms": timestamp_ms,
                "person_id": 0,
            }

            # -------------------------------------------------------
            # Store ALL 33 landmarks.
            #
            # Nothing is permanently disabled.
            # Nothing is filtered out.
            # Visibility is stored as metadata.
            # -------------------------------------------------------

            for landmark_id, landmark in enumerate(
                landmarks
            ):

                row[
                    f"lm{landmark_id}_x"
                ] = landmark.x

                row[
                    f"lm{landmark_id}_y"
                ] = landmark.y

                row[
                    f"lm{landmark_id}_z"
                ] = landmark.z

                row[
                    f"lm{landmark_id}_visibility"
                ] = landmark.visibility

                row[
                    f"lm{landmark_id}_presence"
                ] = landmark.presence

            rows.append(row)

            # -------------------------------------------------------
            # Diagnostic overlay.
            #
            # This is ONLY for human verification.
            # -------------------------------------------------------

            frame = draw_skeleton(
                frame,
                landmarks,
                width,
                height,
            )

        else:

            # -------------------------------------------------------
            # No pose detected.
            #
            # We preserve the frame in the dataset instead of
            # silently deleting it.
            # -------------------------------------------------------

            rows.append(
                {
                    "frame": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "person_id": -1,
                }
            )

            frame = draw_skeleton(
                frame,
                None,
                width,
                height,
            )

        # -----------------------------------------------------------
        # Diagnostic information on overlay
        # -----------------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Time: {timestamp_ms / 1000:.3f}s",
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        video_writer.write(frame)

        frame_number += 1

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

    cap.release()
    video_writer.release()

    actual_frame_count = frame_number

    actual_duration = (
        actual_frame_count / fps
        if fps > 0
        else None
    )

    # ---------------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------------
    df = pd.DataFrame(rows)

    df.to_parquet(
        output_parquet,
        index=False,
    )

    # ---------------------------------------------------------------
    # Write metadata JSON
    # ---------------------------------------------------------------

    metadata = create_metadata(
        video_path=video_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=actual_frame_count,
        duration_seconds=actual_duration,
    )

    metadata["reported_frame_count"] = (
        reported_frame_count
    )

    metadata["frames_with_pose"] = sum(
        1
        for row in rows
        if row.get("person_id", -1) != -1
    )

    metadata["frames_without_pose"] = sum(
        1
        for row in rows
        if row.get("person_id", -1) == -1
    )

    with open(
        output_metadata,
        "w",
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    pose_frames = metadata["frames_with_pose"]

    pose_percentage = (
        (pose_frames / actual_frame_count) * 100
        if actual_frame_count > 0
        else 0
    )

    print()
    print("[OK] Processing completed")
    print(f"     Video:      {video_path}")
    print(f"     Frames:     {actual_frame_count}")
    print(f"     Pose frames:{pose_frames}")
    print(
        f"     Pose rate:  {pose_percentage:.2f}%"
    )
    print(f"     Parquet:        {output_parquet}")
    print(f"     Overlay:    {output_video}")
    print(f"     Metadata:   {output_metadata}")

    if pose_percentage < 80:
        print()
        print(
            "[WARNING] Less than 80% of frames contain "
            "a detected pose."
        )
        print(
            "         Inspect the overlay before using "
            "this clip for training."
        )

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase-2 raw MediaPipe landmark extractor. "
            "Processes ONE repetition video and produces "
            "raw landmark CSV + diagnostic wireframe video."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input video file.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help=(
            "Output directory. "
            "The script creates landmarks/, overlay/, "
            "and metadata/ inside it."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=(
            "Path to pose_landmarker_heavy.task."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    if not args.input.is_file():

        print(
            f"[ERROR] Input video does not exist: "
            f"{args.input}"
        )

        sys.exit(1)

    if args.input.suffix.lower() not in VIDEO_EXTENSIONS:

        print(
            f"[ERROR] Unsupported video format: "
            f"{args.input.suffix}"
        )

        sys.exit(1)

    if not args.model.is_file():

        print(
            f"[ERROR] MediaPipe model does not exist: "
            f"{args.model}"
        )

        sys.exit(1)

    # ---------------------------------------------------------------
    # Create MediaPipe Pose Landmarker
    # ---------------------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = (
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(
                    args.model
                )
            ),
            running_mode=(
                VisionRunningMode.VIDEO
            ),
            num_poses=1,

            # These are MediaPipe's detection/tracking
            # thresholds.
            #
            # IMPORTANT:
            # These do NOT delete landmarks from the
            # resulting dataset.
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    # ---------------------------------------------------------------
    # One fresh landmarker for ONE video.
    #
    # This is important because the entire project processes
    # each repetition independently.
    # ---------------------------------------------------------------

    with (
        mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker
    ):

        output_parquet = (
            args.output
            / "landmarks"
            / f"{args.input.stem}.parquet"
        )

        output_video = (
            args.output
            / "overlay"
            / f"{args.input.stem}_wireframe.mp4"
        )

        output_metadata = (
            args.output
            / "metadata"
            / f"{args.input.stem}.json"
        )

        success = process_video(
            video_path=args.input,
            output_parquet=output_parquet,
            output_video=output_video,
            output_metadata=output_metadata,
            landmarker=landmarker,
        )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


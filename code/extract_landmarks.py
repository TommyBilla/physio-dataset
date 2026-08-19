
#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
}


# ---------------------------------------------------------------------------
# FOOT / FLOOR CONFIGURATION
# ---------------------------------------------------------------------------

FOOT_LANDMARKS = {
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
    "left_ankle": 27,
    "right_ankle": 28,
}

FLOOR_REFERENCE_LANDMARKS = [
    "left_heel",
    "right_heel",
]

FLOOR_CONFIDENCE_MIN_SAMPLES = 5
FLOOR_CONFIDENCE_MIN_VISIBILITY = 0.5

HEEL_LIFT_THRESHOLD_NORM = 0.03

# A single frame above the threshold is not sufficient to classify
# a genuine heel lift.
HEEL_LIFT_MIN_CONSECUTIVE_FRAMES = 3

FLOOR_PLATEAU_MIN_RUN_FRAMES = 4
FLOOR_PLATEAU_TOLERANCE_NORM = 0.015
FLOOR_PLATEAU_BOTTOM_FRACTION = 0.20


# ---------------------------------------------------------------------------
# MEDIAPIPE LANDMARK CONFIGURATION
# ---------------------------------------------------------------------------

NUM_LANDMARKS = 33

# IMPORTANT:
# This is a quality threshold, NOT a permanent landmark-disable threshold.
#
# A landmark can be invisible for 20 frames and become reliable again later.
# Therefore we NEVER permanently disable landmarks.
LANDMARK_VISIBILITY_WARNING_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# FLOOR ESTIMATION
# ---------------------------------------------------------------------------

def find_longest_stable_run(values, tolerance):
    """
    Find the longest consecutive run whose values remain within
    `tolerance` of the run's first value.

    Returns:
        (start_idx, end_idx, run_mean)

    or:

        (None, None, None)
    """

    if not values:
        return None, None, None

    best_start = 0
    best_end = 0
    best_len = 1

    start = 0

    for i in range(1, len(values) + 1):

        if (
            i < len(values)
            and abs(values[i] - values[start]) <= tolerance
        ):
            continue

        run_len = i - start

        if run_len > best_len:
            best_len = run_len
            best_start = start
            best_end = i - 1

        start = i

    run_values = values[
        best_start : best_end + 1
    ]

    run_mean = sum(run_values) / len(run_values)

    return (
        best_start,
        best_end,
        run_mean,
    )


def estimate_floor_y(rows, landmark_name):
    """
    Estimate the surface/floor reference for one heel.

    IMPORTANT:
    This does NOT use a single lowest-y frame.

    Process:

    1. Collect only frames where the landmark exists.
    2. Require sufficient visibility.
    3. Take the bottom portion of the observed y-values.
    4. Search those candidates in temporal order.
    5. Find a stable consecutive plateau.
    6. Use the mean of that plateau as the floor reference.

    Returns:

        floor_y
        confidence
        number_of_samples
    """

    landmark_id = FOOT_LANDMARKS[landmark_name]

    y_key = f"lm{landmark_id}_y"
    visibility_key = f"lm{landmark_id}_visibility"

    timed_candidates = []

    for row in rows:

        if row.get("person_id", -1) == -1:
            continue

        y_value = row.get(y_key)
        visibility = row.get(visibility_key)

        if y_value is None:
            continue

        if visibility is None:
            continue

        if visibility < FLOOR_CONFIDENCE_MIN_VISIBILITY:
            continue

        timed_candidates.append(
            float(y_value)
        )

    num_samples = len(timed_candidates)

    if num_samples == 0:
        return None, "low", 0

    sorted_ys = sorted(timed_candidates)

    bottom_count = max(
        1,
        int(
            len(sorted_ys)
            * FLOOR_PLATEAU_BOTTOM_FRACTION
        ),
    )

    bottom_threshold = sorted_ys[-bottom_count]

    # Keep temporal order.
    bottom_slice_in_time_order = [
        y
        for y in timed_candidates
        if y >= bottom_threshold
    ]

    (
        start,
        end,
        run_mean,
    ) = find_longest_stable_run(
        bottom_slice_in_time_order,
        FLOOR_PLATEAU_TOLERANCE_NORM,
    )

    if run_mean is None:
        return None, "low", num_samples

    run_length = end - start + 1

    has_plateau = (
        run_length
        >= FLOOR_PLATEAU_MIN_RUN_FRAMES
    )

    has_enough_samples = (
        num_samples
        >= FLOOR_CONFIDENCE_MIN_SAMPLES
    )

    confidence = (
        "high"
        if (
            has_plateau
            and has_enough_samples
        )
        else "low"
    )

    return (
        run_mean,
        confidence,
        num_samples,
    )


def estimate_clip_floor(
    rows,
    manual_override=None,
):
    """
    Estimate floor/surface reference for both heels.

    Manual override takes priority over automatic estimation.
    """

    manual_override = (
        manual_override or {}
    )

    floor_estimates = {}

    for landmark_name in FLOOR_REFERENCE_LANDMARKS:

        if landmark_name in manual_override:

            floor_estimates[
                landmark_name
            ] = {
                "floor_y": float(
                    manual_override[
                        landmark_name
                    ]
                ),
                "confidence": "manual",
                "source": "manual_override",
                "num_samples": None,
            }

            continue

        (
            floor_y,
            confidence,
            num_samples,
        ) = estimate_floor_y(
            rows,
            landmark_name,
        )

        floor_estimates[
            landmark_name
        ] = {
            "floor_y": floor_y,
            "confidence": confidence,
            "source": "automatic_plateau",
            "num_samples": num_samples,
        }

    return floor_estimates


def add_floor_relative_columns(
    rows,
    floor_estimates,
):
    """
    Add surface-relative features.

    For each heel:

        floor_y
        height_above_floor
        floor_confidence

    Also add:

        heel_lifting_flag_left
        heel_lifting_flag_right

        heel_lifting_confirmed_flag_left
        heel_lifting_confirmed_flag_right

    IMPORTANT:

    These are DERIVED columns.

    Raw MediaPipe coordinates remain untouched.
    """

    # -----------------------------------------------------------------------
    # Per-frame floor-relative measurements
    # -----------------------------------------------------------------------

    for row in rows:

        if row.get("person_id", -1) == -1:
            continue

        for landmark_name in FLOOR_REFERENCE_LANDMARKS:

            landmark_id = FOOT_LANDMARKS[
                landmark_name
            ]

            y_key = f"lm{landmark_id}_y"

            floor_y_key = (
                f"lm{landmark_id}_floor_y"
            )

            height_key = (
                f"lm{landmark_id}"
                "_height_above_floor"
            )

            confidence_key = (
                f"lm{landmark_id}"
                "_floor_confidence"
            )

            estimate = floor_estimates[
                landmark_name
            ]

            floor_y = estimate[
                "floor_y"
            ]

            landmark_y = row.get(
                y_key
            )

            row[floor_y_key] = floor_y

            row[
                confidence_key
            ] = estimate[
                "confidence"
            ]

            if (
                floor_y is None
                or landmark_y is None
            ):
                row[
                    height_key
                ] = None

                continue

            # Image coordinates:
            #
            # y increases downward.
            #
            # Therefore:
            #
            # floor_y - landmark_y
            #
            # is positive when the heel is above
            # the reference surface.

            height_above_floor = (
                floor_y - landmark_y
            )

            row[
                height_key
            ] = height_above_floor

            side = (
                "left"
                if "left"
                in landmark_name
                else "right"
            )

            raw_flag_key = (
                f"heel_lifting_flag_{side}"
            )

            row[
                raw_flag_key
            ] = (
                height_above_floor
                > HEEL_LIFT_THRESHOLD_NORM
            )

    # -----------------------------------------------------------------------
    # Temporal confirmation
    # -----------------------------------------------------------------------

    for side in (
        "left",
        "right",
    ):

        raw_key = (
            f"heel_lifting_flag_{side}"
        )

        confirmed_key = (
            "heel_lifting_confirmed_flag_"
            f"{side}"
        )

        consecutive = 0

        for row in rows:

            if row.get(
                "person_id",
                -1,
            ) == -1:

                row[
                    confirmed_key
                ] = False

                consecutive = 0
                continue

            raw_flag = row.get(
                raw_key
            )

            if raw_flag is True:
                consecutive += 1
            else:
                consecutive = 0

            row[
                confirmed_key
            ] = (
                consecutive
                >= HEEL_LIFT_MIN_CONSECUTIVE_FRAMES
            )

    return rows


# ---------------------------------------------------------------------------
# MANUAL FLOOR OVERRIDES
# ---------------------------------------------------------------------------

def load_manual_overrides(
    override_json_path,
    video_path,
    input_root,
):
    """
    Load manual surface/floor overrides.

    Expected JSON:

    {
        "good/example.MOV": {
            "left_heel": 0.87,
            "right_heel": 0.86
        }
    }
    """

    if override_json_path is None:
        return {}

    if not override_json_path.is_file():
        return {}

    with open(
        override_json_path,
        "r",
        encoding="utf-8",
    ) as f:

        overrides = json.load(f)

    try:
        relative_key = str(
            video_path.relative_to(
                input_root
            )
        )
    except ValueError:
        relative_key = video_path.name

    return overrides.get(
        relative_key,
        {},
    )


# ---------------------------------------------------------------------------
# EMPTY LANDMARK VALUES
# ---------------------------------------------------------------------------

def add_empty_landmark_columns(row):
    """
    Ensure every frame has the same schema.

    Missing landmarks are represented as None.

    We DO NOT interpolate here.

    We DO NOT invent coordinates.

    We DO NOT permanently disable landmarks.
    """

    for landmark_id in range(
        NUM_LANDMARKS
    ):

        row.setdefault(
            f"lm{landmark_id}_x",
            None,
        )

        row.setdefault(
            f"lm{landmark_id}_y",
            None,
        )

        row.setdefault(
            f"lm{landmark_id}_z",
            None,
        )

        row.setdefault(
            f"lm{landmark_id}_visibility",
            None,
        )

        row.setdefault(
            f"lm{landmark_id}_presence",
            None,
        )

    return row


# ---------------------------------------------------------------------------
# PROCESS ONE VIDEO
# ---------------------------------------------------------------------------

def process_video(
    video_path,
    output_path,
    landmarker,
    manual_override=None,
):
    """
    Process exactly ONE video.

    One video = one independent MediaPipe VIDEO session.

    This is intentional because timestamps/tracking state must not
    leak between clips.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"[ERROR] Could not open: "
            f"{video_path}"
        )

        return False

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    frame_number = 0

    previous_timestamp_ms = -1

    rows = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ---------------------------------------------------------------
        # OpenCV BGR -> RGB
        # ---------------------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb,
        )

        # ---------------------------------------------------------------
        # Guaranteed monotonically increasing timestamp
        # ---------------------------------------------------------------

        timestamp_ms = int(
            (frame_number / fps)
            * 1000
        )

        if (
            timestamp_ms
            <= previous_timestamp_ms
        ):

            timestamp_ms = (
                previous_timestamp_ms
                + 1
            )

        previous_timestamp_ms = (
            timestamp_ms
        )

        # ---------------------------------------------------------------
        # MediaPipe inference
        # ---------------------------------------------------------------

        try:

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

        except Exception as e:

            print(
                f"[ERROR] Frame "
                f"{frame_number} failed "
                f"in {video_path}: {e}"
            )

            # Preserve frame continuity.
            row = {
                "frame": frame_number,
                "timestamp_ms": timestamp_ms,
                "person_id": -1,
                "mediapipe_error": True,
            }

            row = (
                add_empty_landmark_columns(
                    row
                )
            )

            rows.append(row)

            frame_number += 1

            continue

        # ---------------------------------------------------------------
        # Pose detected
        # ---------------------------------------------------------------

        if result.pose_landmarks:

            for (
                person_id,
                landmarks,
            ) in enumerate(
                result.pose_landmarks
            ):

                row = {
                    "frame": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "person_id": person_id,
                    "mediapipe_error": False,
                }

                for (
                    landmark_id,
                    landmark,
                ) in enumerate(
                    landmarks
                ):

                    # ---------------------------------------------------
                    # RAW VALUES
                    #
                    # NEVER discard based on visibility.
                    # NEVER permanently disable a landmark.
                    # ---------------------------------------------------

                    row[
                        f"lm{landmark_id}_x"
                    ] = float(
                        landmark.x
                    )

                    row[
                        f"lm{landmark_id}_y"
                    ] = float(
                        landmark.y
                    )

                    row[
                        f"lm{landmark_id}_z"
                    ] = float(
                        landmark.z
                    )

                    row[
                        f"lm{landmark_id}"
                        "_visibility"
                    ] = float(
                        landmark.visibility
                    )

                    row[
                        f"lm{landmark_id}"
                        "_presence"
                    ] = float(
                        landmark.presence
                    )

                row = (
                    add_empty_landmark_columns(
                        row
                    )
                )

                rows.append(row)

        # ---------------------------------------------------------------
        # No pose detected
        # ---------------------------------------------------------------

        else:

            row = {
                "frame": frame_number,
                "timestamp_ms": timestamp_ms,
                "person_id": -1,
                "mediapipe_error": False,
            }

            row = (
                add_empty_landmark_columns(
                    row
                )
            )

            rows.append(row)

        frame_number += 1

    cap.release()

    if not rows:

        print(
            f"[WARNING] No frames "
            f"processed: {video_path}"
        )

        return False

    # -------------------------------------------------------------------
    # FLOOR / SURFACE ESTIMATION
    # -------------------------------------------------------------------

    floor_estimates = (
        estimate_clip_floor(
            rows,
            manual_override,
        )
    )

    rows = (
        add_floor_relative_columns(
            rows,
            floor_estimates,
        )
    )

    # -------------------------------------------------------------------
    # FLOOR REPORT
    # -------------------------------------------------------------------

    for (
        landmark_name,
        estimate,
    ) in floor_estimates.items():

        print(
            f"    floor[{landmark_name}]: "
            f"y={estimate['floor_y']} "
            f"confidence="
            f"{estimate['confidence']} "
            f"source="
            f"{estimate['source']} "
            f"samples="
            f"{estimate['num_samples']}"
        )

        if (
            estimate["confidence"]
            == "low"
        ):

            print(
                "    [WARNING] "
                f"Low-confidence floor "
                f"estimate for "
                f"{landmark_name}."
            )

    # -------------------------------------------------------------------
    # BUILD DATAFRAME
    # -------------------------------------------------------------------

    dataframe = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------
    # EXPLICIT COLUMN ORDER
    # -------------------------------------------------------------------

    fieldnames = [
        "frame",
        "timestamp_ms",
        "person_id",
        "mediapipe_error",
    ]

    for i in range(
        NUM_LANDMARKS
    ):

        fieldnames.extend(
            [
                f"lm{i}_x",
                f"lm{i}_y",
                f"lm{i}_z",
                f"lm{i}_visibility",
                f"lm{i}_presence",
            ]
        )

    for landmark_name in (
        FLOOR_REFERENCE_LANDMARKS
    ):

        landmark_id = FOOT_LANDMARKS[
            landmark_name
        ]

        fieldnames.extend(
            [
                f"lm{landmark_id}"
                "_floor_y",

                f"lm{landmark_id}"
                "_height_above_floor",

                f"lm{landmark_id}"
                "_floor_confidence",
            ]
        )

    fieldnames.extend(
        [
            "heel_lifting_flag_left",
            "heel_lifting_flag_right",

            "heel_lifting_confirmed_flag_left",
            "heel_lifting_confirmed_flag_right",
        ]
    )

    # Add any missing columns explicitly.
    for column in fieldnames:

        if column not in dataframe.columns:

            dataframe[column] = None

    dataframe = dataframe[
        fieldnames
    ]

    # -------------------------------------------------------------------
    # DATA TYPES
    # -------------------------------------------------------------------

    dataframe["frame"] = (
        dataframe["frame"]
        .astype("int64")
    )

    dataframe["timestamp_ms"] = (
        dataframe["timestamp_ms"]
        .astype("int64")
    )

    dataframe["person_id"] = (
        dataframe["person_id"]
        .astype("int64")
    )

    dataframe[
        "mediapipe_error"
    ] = dataframe[
        "mediapipe_error"
    ].fillna(False).astype(bool)

    # -------------------------------------------------------------------
    # SAVE PARQUET
    # -------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    # -------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------

    detected_frames = int(
        (
            dataframe["person_id"]
            != -1
        ).sum()
    )

    total_frames = len(
        dataframe
    )

    print(
        f"[OK] {video_path} -> "
        f"{output_path} "
        f"({frame_number} frames, "
        f"{detected_frames} detected)"
    )

    return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract raw MediaPipe Pose "
            "Landmarker Heavy features "
            "from exactly one video and "
            "save them as Parquet."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input video file",
    )

    parser.add_argument(
        "output",
        type=Path,
        help=(
            "Output directory. "
            "The Parquet file will be "
            "written directly inside it."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=(
            "Path to "
            "pose_landmarker_heavy.task"
        ),
    )

    parser.add_argument(
        "--floor-override-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON containing "
            "manual floor/surface "
            "references."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------

    if not args.input.is_file():

        print(
            f"Error: input video does "
            f"not exist: {args.input}"
        )

        sys.exit(1)

    if (
        args.input.suffix.lower()
        not in VIDEO_EXTENSIONS
    ):

        print(
            "Error: unsupported video "
            f"format: {args.input.suffix}"
        )

        sys.exit(1)

    if not args.model.is_file():

        print(
            "Error: model does not exist: "
            f"{args.model}"
        )

        sys.exit(1)

    if (
        args.floor_override_json
        and not args.floor_override_json.is_file()
    ):

        print(
            "Error: floor override file "
            "does not exist: "
            f"{args.floor_override_json}"
        )

        sys.exit(1)

    # -------------------------------------------------------------------
    # MEDIAPIPE CONFIGURATION
    # -------------------------------------------------------------------

    BaseOptions = (
        mp.tasks.BaseOptions
    )

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

            min_pose_detection_confidence=0.5,

            min_pose_presence_confidence=0.5,

            min_tracking_confidence=0.5,
        )
    )

    # -------------------------------------------------------------------
    # ONE FRESH LANDMARKER PER VIDEO
    # -------------------------------------------------------------------

    with (
        mp.tasks.vision.PoseLandmarker
        .create_from_options(
            options
        )
        as landmarker
    ):

        # For a single-video invocation,
        # the video itself is the root.
        #
        # This also keeps compatibility
        # with the manual override format.

        input_root = (
            args.input.parent
        )

        manual_override = (
            load_manual_overrides(
                args.floor_override_json,
                args.input,
                input_root,
            )
        )

        # ---------------------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------------------

        output_path = (
            args.output
            / f"{args.input.stem}.parquet"
        )

        success = process_video(
            args.input,
            output_path,
            landmarker,
            manual_override,
        )

        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()


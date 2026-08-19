#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
}


def find_videos(input_root):
    """Find videos recursively, preserving their paths relative to input_root."""
    return sorted(
        [
            path
            for path in input_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda path: str(path.relative_to(input_root)).lower(),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Feed videos one at a time to extract_landmarks.py "
            "and preserve the input folder structure."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input dataset directory",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output dataset directory",
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to pose_landmarker_heavy.task",
    )

    parser.add_argument(
        "--extractor",
        type=Path,
        default=Path(__file__).with_name("extract_landmarks.py"),
        help="Path to extract_landmarks.py",
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: input directory does not exist: {args.input}")
        sys.exit(1)

    if not args.model.is_file():
        print(f"Error: model does not exist: {args.model}")
        sys.exit(1)

    if not args.extractor.is_file():
        print(f"Error: extractor does not exist: {args.extractor}")
        sys.exit(1)

    videos = find_videos(args.input)

    if not videos:
        print("No videos found.")
        return

    print("=" * 60)
    print(f"Found {len(videos)} video(s)")
    print("=" * 60)

    completed = 0
    failed = 0

    for index, video_path in enumerate(videos, start=1):
        relative_path = video_path.relative_to(args.input)

        # The extractor receives the output directory corresponding
        # to the video's original parent folder.
        #
        # Example:
        # input/heel slides/good/1.MOV
        #      -> output/heel slides/good/1.csv
        output_dir = args.output / relative_path.parent

        print()
        print("-" * 60)
        print(f"[{index}/{len(videos)}] Processing: {relative_path}")
        print("-" * 60)

        command = [
            sys.executable,
            str(args.extractor),
            "--model",
            str(args.model),
            str(video_path),
            str(output_dir),
        ]

        result = subprocess.run(command)

        if result.returncode == 0:
            print(f"[COMPLETED] {relative_path}")
            completed += 1
        else:
            print(
                f"[FAILED] {relative_path} "
                f"(extractor exited with code {result.returncode})"
            )
            failed += 1

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Videos found:       {len(videos)}")
    print(f"Completed:          {completed}")
    print(f"Failed:             {failed}")
    print(f"Output directory:   {args.output}")


if __name__ == "__main__":
    main()

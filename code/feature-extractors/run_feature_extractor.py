#!/usr/bin/env python3
"""
Generic batch pipeline for feature extractors.

Usage:
    python run_feature_extractor.py \
        --input /path/to/landmarks \
        --extractor /path/to/heel_slide_feature_extractor.py \
        --output /path/to/features

The extractor must accept:
    extractor.py INPUT_PARQUET OUTPUT_PARQUET

Folder structure and filenames are preserved exactly.

Example:
    landmarks/heel_slides/1.parquet
becomes:
    features/heel_slides/1.parquet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a feature extractor once per Parquet file while "
            "preserving the input directory structure."
        )
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Root folder containing input Parquet files.",
    )

    parser.add_argument(
        "--extractor", "-e",
        type=Path,
        required=True,
        help="Python extractor script.",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Root folder for generated feature Parquets.",
    )

    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help=(
            "Python interpreter to run the extractor with. "
            "Defaults to the current Python interpreter."
        ),
    )

    parser.add_argument(
        "--pattern",
        default="*.parquet",
        help="Recursive input filename pattern. Default: *.parquet",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one extractor fails.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show all planned jobs without running them.",
    )

    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not args.input.is_dir():
        raise SystemExit(
            f"[ERROR] Input directory does not exist: {args.input}"
        )

    if not args.extractor.is_file():
        raise SystemExit(
            f"[ERROR] Extractor does not exist: {args.extractor}"
        )

    if args.extractor.suffix.lower() != ".py":
        raise SystemExit(
            f"[ERROR] Extractor must be a .py file: {args.extractor}"
        )

    if args.python is not None and not args.python.is_file():
        raise SystemExit(
            f"[ERROR] Python interpreter does not exist: {args.python}"
        )


def find_parquets(root: Path, pattern: str) -> list[Path]:
    return sorted(
        [
            p for p in root.rglob(pattern)
            if p.is_file() and p.suffix.lower() in {".parquet", ".pq"}
        ],
        key=lambda p: p.as_posix().lower(),
    )


def output_path_for(
    input_root: Path,
    output_root: Path,
    input_file: Path,
) -> Path:
    return output_root / input_file.relative_to(input_root)


def run_extractor(
    python_executable: str,
    extractor: Path,
    input_file: Path,
    output_file: Path,
) -> subprocess.CompletedProcess[str]:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        python_executable,
        str(extractor),
        str(input_file),
        str(output_file),
    ]

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = parse_args()
    validate(args)

    input_root = args.input.resolve()
    extractor = args.extractor.resolve()
    output_root = args.output.resolve()

    python_executable = (
        str(args.python.resolve())
        if args.python is not None
        else sys.executable
    )

    files = find_parquets(
        input_root,
        args.pattern,
    )

    if not files:
        print(f"[WARNING] No Parquet files found under {input_root}")
        return 0

    print("=" * 72)
    print("FEATURE EXTRACTION PIPELINE")
    print("=" * 72)
    print(f"Input     : {input_root}")
    print(f"Extractor : {extractor}")
    print(f"Output    : {output_root}")
    print(f"Python    : {python_executable}")
    print(f"Files     : {len(files)}")
    print("=" * 72)

    if args.dry_run:
        print("\n[DRY RUN]\n")
        for i, input_file in enumerate(files, 1):
            output_file = output_path_for(
                input_root,
                output_root,
                input_file,
            )
            print(
                f"[{i:>4}/{len(files)}] "
                f"{input_file.relative_to(input_root)}"
                f" -> "
                f"{output_file.relative_to(output_root)}"
            )
        return 0

    success = 0
    skipped = 0
    failed = 0
    failures: list[tuple[Path, Path, str]] = []
    total_start = time.perf_counter()

    for index, input_file in enumerate(files, 1):
        output_file = output_path_for(
            input_root,
            output_root,
            input_file,
        )

        print("\n" + "-" * 72)
        print(
            f"[{index}/{len(files)}] "
            f"{input_file.relative_to(input_root)}"
        )

        if output_file.exists() and not args.overwrite:
            print(f"[SKIP] {output_file}")
            skipped += 1
            continue

        start = time.perf_counter()

        result = run_extractor(
            python_executable,
            extractor,
            input_file,
            output_file,
        )

        elapsed = time.perf_counter() - start

        if result.stdout.strip():
            print(result.stdout.rstrip())

        if result.stderr.strip():
            print(result.stderr.rstrip())

        if result.returncode != 0:
            failed += 1

            error = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"Exit code: {result.returncode}"
            )

            failures.append(
                (input_file, output_file, error)
            )

            print(
                f"[FAILED] Exit code {result.returncode} "
                f"after {elapsed:.2f}s"
            )

            # Never leave a partial output masquerading as a valid result.
            if output_file.exists():
                try:
                    output_file.unlink()
                    print("[CLEANUP] Removed failed output.")
                except OSError as cleanup_error:
                    print(
                        "[WARNING] Could not remove failed output: "
                        f"{cleanup_error}"
                    )

            if args.fail_fast:
                print("[STOP] --fail-fast enabled.")
                break

            continue

        if not output_file.exists():
            failed += 1

            error = (
                "Extractor returned exit code 0, "
                "but did not create the expected output file."
            )

            failures.append(
                (input_file, output_file, error)
            )

            print(f"[FAILED] {error}")

            if args.fail_fast:
                print("[STOP] --fail-fast enabled.")
                break

            continue

        success += 1
        print(f"[OK] Completed in {elapsed:.2f}s")
        print(f"     Output: {output_file}")

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)
    print(f"Found   : {len(files)}")
    print(f"Success : {success}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Time    : {total_elapsed:.2f}s")

    if failures:
        print("\nFAILED FILES")
        print("-" * 72)
        for input_file, output_file, error in failures:
            print(f"INPUT : {input_file}")
            print(f"OUTPUT: {output_file}")
            print(f"ERROR : {error}\n")

    print("=" * 72)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

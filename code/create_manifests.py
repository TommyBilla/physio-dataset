#!/usr/bin/env python3
"""
Create manifest CSV files for each exercise folder.
Scans subdirectories that contain 'good' and 'bad' subdirectories.
For each video file (.MOV) in those subdirectories, extracts metadata using ffprobe.
Writes a CSV manifest in the base directory with sanitized folder name.
"""

import os
import subprocess
import json
import csv
import re

BASE_DIR = "/home/billa-dakait/Desktop/test"
VIDEO_EXTENSIONS = {".MOV"}  # observed extension; can be extended

def sanitize_folder_name(name):
    """Convert folder name to lowercase and replace spaces with underscores."""
    return name.lower().replace(" ", "_")

def get_video_info(filepath):
    """Return duration (float), fps (float), frame_count (int) for video file."""
    # ffprobe command to get JSON output for duration and frame rate
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate",
        "-of", "json",
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        duration = float(stream.get("duration", 0))
        r_frame_rate = stream.get("r_frame_rate", "0/1")
        # parse fraction like "30/1"
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(r_frame_rate)
        frame_count = int(round(duration * fps))
        return duration, fps, frame_count
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    # Find immediate subdirectories that contain both 'good' and 'bad'
    main_folders = []
    for entry in os.listdir(BASE_DIR):
        entry_path = os.path.join(BASE_DIR, entry)
        if os.path.isdir(entry_path):
            good_path = os.path.join(entry_path, "good")
            bad_path = os.path.join(entry_path, "bad")
            if os.path.isdir(good_path) and os.path.isdir(bad_path):
                main_folders.append(entry)

    print(f"Found main folders: {main_folders}")

    for folder in main_folders:
        folder_path = os.path.join(BASE_DIR, folder)
        manifest_rows = []
        # Process good and bad subdirectories
        for quality in ["good", "bad"]:
            quality_path = os.path.join(folder_path, quality)
            if not os.path.isdir(quality_path):
                continue
            for filename in os.listdir(quality_path):
                if any(filename.lower().endswith(ext.lower()) for ext in VIDEO_EXTENSIONS):
                    filepath = os.path.join(quality_path, filename)
                    rel_path = os.path.relpath(filepath, BASE_DIR)
                    video_info = get_video_info(filepath)
                    if video_info is None:
                        continue
                    duration, fps, frame_count = video_info
                    # video_id: quality_filename_without_extension
                    name_without_ext = os.path.splitext(filename)[0]
                    video_id = f"{quality}_{name_without_ext}"
                    row = {
                        "video_id": video_id,
                        "exercise": folder,
                        "filename": filename,
                        "path": rel_path,
                        "duration": duration,
                        "fps": fps,
                        "frame_count": frame_count
                    }
                    manifest_rows.append(row)

        if manifest_rows:
            # Sanitize folder name for manifest filename
            safe_name = sanitize_folder_name(folder)
            manifest_filename = f"{safe_name}_manifest.csv"
            manifest_path = os.path.join(BASE_DIR, manifest_filename)
            fieldnames = ["video_id", "exercise", "filename", "path", "duration", "fps", "frame_count"]
            with open(manifest_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(manifest_rows)
            print(f"Created manifest: {manifest_path} with {len(manifest_rows)} entries")
        else:
            print(f"No video files found for folder {folder}")

if __name__ == "__main__":
    main()
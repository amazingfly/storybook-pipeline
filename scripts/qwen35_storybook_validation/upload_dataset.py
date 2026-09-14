#!/usr/bin/env python3
"""Upload and verify the Qwen validation dataset with rclone."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_REMOTE = (
    "gDrive:littlequeen/storybook_validation_v3_qwen35/input"
)


def run(command: list[str], attempts: int = 50) -> None:
    for attempt in range(1, attempts + 1):
        print(
            f"+ [attempt {attempt}/{attempts}]",
            " ".join(command),
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(completed.returncode, command)
        delay = min(60, 10 * attempt)
        print(
            f"rclone operation failed; retrying in {delay} seconds",
            flush=True,
        )
        time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if (source / "transfer" / "archive_manifest.json").is_file():
        source = source / "transfer"
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is not installed")
    manifest_path = source / "archive_manifest.json"
    index_path = source / "dataset_index.json"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("dataset build is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "archive_manifest.json",
        "dataset_index.json",
        *[
            part["name"]
            for item in manifest["archives"]
            for part in item.get("parts", [{"name": item["name"]}])
        ],
    }
    local_files = {path.name for path in source.iterdir() if path.is_file()}
    missing = sorted(name for name in expected if name not in local_files)
    if missing:
        raise FileNotFoundError(f"missing dataset files: {missing}")
    unexpected = sorted(local_files - expected)
    if unexpected:
        raise RuntimeError(f"unexpected files in dataset directory: {unexpected}")

    run(["rclone", "mkdir", args.remote])
    run(
        [
            "rclone",
            "copy",
            "--checksum",
            "--drive-chunk-size",
            "8M",
            "--transfers",
            "8",
            "--checkers",
            "4",
            "--retries",
            "5",
            "--low-level-retries",
            "4",
            "--retries-sleep",
            "30s",
            "--tpslimit",
            "2",
            "--tpslimit-burst",
            "1",
            "--drive-pacer-min-sleep",
            "500ms",
            "--drive-pacer-burst",
            "2",
            "--timeout",
            "90s",
            "--max-duration",
            "3m",
            "--cutoff-mode",
            "hard",
            "--stats",
            "30s",
            "--stats-one-line",
            "-v",
            str(source),
            args.remote,
        ]
    )
    run(
        [
            "rclone",
            "check",
            "--one-way",
            "--checkers",
            "2",
            str(source),
            args.remote,
        ]
    )
    print(
        json.dumps(
            {
                "source": str(source),
                "remote": args.remote,
                "verified_files": len(expected),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

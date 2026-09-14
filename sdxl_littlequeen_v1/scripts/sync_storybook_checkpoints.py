#!/usr/bin/env python3
"""Mirror immutable Colab checkpoint batches to local storage while a run is active."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import Any


CHECKPOINT_VERSION = "storybook-durable-checkpoint-v1"
TOKEN_REFRESHER = Path(__file__).with_name("refresh_colab_runtime_token.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def colab_python() -> str:
    launcher = shutil.which("colab")
    if launcher is None:
        raise RuntimeError("colab command is not available")
    first_line = Path(launcher).read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("#!"):
        raise RuntimeError(f"colab launcher has no Python shebang: {launcher}")
    interpreter = first_line[2:].strip()
    if not Path(interpreter).is_file():
        raise RuntimeError(f"Colab Python interpreter is missing: {interpreter}")
    return interpreter


def refresh_runtime_token(session: str) -> None:
    result = subprocess.run(
        [
            colab_python(),
            str(TOKEN_REFRESHER),
            "--session",
            session,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0:
        raise RuntimeError(output or "Colab runtime token refresh failed")
    if output:
        print(output, flush=True)


def download(session: str, remote: str, destination: Path) -> tuple[bool, str]:
    destination.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "colab",
            "download",
            "-s",
            session,
            remote,
            str(destination),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return result.returncode == 0 and destination.is_file(), output


def valid_batch(path: Path, record: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(record["bytes"])
        and sha256(path) == record["sha256"]
    )


def extract_batch(path: Path, record: dict, destination: Path) -> None:
    marker = path.with_suffix(path.suffix + ".extracted")
    identity = record["sha256"] + "\n"
    if marker.is_file() and marker.read_text(encoding="utf-8") == identity:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r") as archive:
        archive.extractall(destination, filter="data")
    temporary = marker.with_suffix(marker.suffix + ".part")
    temporary.write_text(identity, encoding="utf-8")
    temporary.replace(marker)


def merge_manifests(local: dict, remote: dict, compiled_sha256: str) -> dict:
    if local.get("compiled_story_sha256") != compiled_sha256:
        raise RuntimeError("local checkpoint belongs to a different compiled story")

    local_batches = {
        item["archive"]: item for item in local.get("batches", [])
    }
    remote_batches = {
        item["archive"]: item for item in remote.get("batches", [])
    }
    for archive_name in local_batches.keys() & remote_batches.keys():
        local_record = local_batches[archive_name]
        remote_record = remote_batches[archive_name]
        identity_fields = ("sequence", "bytes", "sha256", "file_count")
        if any(
            local_record.get(field) != remote_record.get(field)
            for field in identity_fields
        ):
            raise RuntimeError(
                "checkpoint batch conflict for "
                f"{archive_name}; refusing to overwrite the local archive"
            )

    merged_batches = dict(local_batches)
    merged_batches.update(remote_batches)
    sequences: dict[int, str] = {}
    for archive_name, record in merged_batches.items():
        sequence = int(record["sequence"])
        prior_archive = sequences.get(sequence)
        if prior_archive is not None and prior_archive != archive_name:
            raise RuntimeError(
                f"checkpoint sequence {sequence} is used by both "
                f"{prior_archive} and {archive_name}"
            )
        sequences[sequence] = archive_name

    merged_files = dict(local.get("files", {}))
    for relative, record in remote.get("files", {}).items():
        prior = merged_files.get(relative)
        if prior is not None and prior != record:
            raise RuntimeError(
                f"checkpoint file conflict for {relative}; "
                "refusing to replace the local completion record"
            )
        merged_files[relative] = record

    merged = dict(remote)
    merged["batches"] = sorted(
        merged_batches.values(),
        key=lambda item: int(item["sequence"]),
    )
    merged["files"] = merged_files
    if local.get("last_checkpoint_at"):
        merged["last_checkpoint_at"] = max(
            local["last_checkpoint_at"],
            remote.get("last_checkpoint_at", ""),
        )
    return merged


def sync_once(args: argparse.Namespace) -> dict[str, Any] | None:
    args.local_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.local_dir / ".remote_manifest.json.part"
    ok, output = download(
        args.session,
        f"{args.remote_dir}/checkpoint_manifest.json",
        temporary_manifest,
    )
    if not ok:
        print(
            "[checkpoint-download] manifest is not available yet"
            + (f": {output}" if output else ""),
            flush=True,
        )
        return None
    manifest = json.loads(temporary_manifest.read_text(encoding="utf-8"))
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError("unexpected checkpoint manifest version")
    if manifest.get("compiled_story_sha256") != args.compiled_sha256:
        raise RuntimeError("remote checkpoint belongs to a different compiled story")

    local_manifest_path = args.local_dir / "checkpoint_manifest.json"
    if local_manifest_path.is_file():
        try:
            local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            local_manifest = None
        if isinstance(local_manifest, dict):
            manifest = merge_manifests(
                local_manifest,
                manifest,
                args.compiled_sha256,
            )

    downloaded = 0
    for batch in manifest["batches"]:
        destination = args.local_dir / batch["archive"]
        if valid_batch(destination, batch):
            if args.extract_root is not None:
                extract_batch(destination, batch, args.extract_root)
            continue
        if destination.exists():
            raise RuntimeError(
                "checkpoint batch conflict for "
                f"{destination.name}; refusing to overwrite the local archive"
            )
        part = destination.with_suffix(destination.suffix + ".part")
        ok, output = download(
            args.session,
            f"{args.remote_dir}/{batch['archive']}",
            part,
        )
        if not ok:
            raise RuntimeError(
                f"failed to download checkpoint {batch['archive']}: {output}"
            )
        if not valid_batch(part, batch):
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"checkpoint verification failed for {batch['archive']}"
            )
        part.replace(destination)
        if args.extract_root is not None:
            extract_batch(destination, batch, args.extract_root)
        downloaded += 1

    local_manifest = args.local_dir / "checkpoint_manifest.json"
    temporary_local = local_manifest.with_suffix(".json.part")
    temporary_local.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_local.replace(local_manifest)
    temporary_manifest.unlink(missing_ok=True)
    print(
        f"[checkpoint-download] mirrored {len(manifest['batches'])} batches"
        f" ({downloaded} new) to {args.local_dir}",
        flush=True,
    )
    return manifest


def session_exists(session: str) -> bool | None:
    try:
        result = subprocess.run(
            ["colab", "sessions"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return f"[{session}]" in result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--compiled-sha256", required=True)
    parser.add_argument(
        "--extract-root",
        type=Path,
        help="Extract each verified immutable batch into this run directory.",
    )
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--recovery-seconds", type=float, default=43200.0)
    parser.add_argument("--session-miss-limit", type=int, default=3)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--token-refresh-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recovery_deadline = None
    session_misses = 0
    next_token_refresh = 0.0
    while True:
        watcher_alive = process_exists(args.watch_pid)
        if not watcher_alive and recovery_deadline is None:
            recovery_deadline = time.monotonic() + args.recovery_seconds
            print(
                "[checkpoint-download] Colab command disconnected; "
                f"continuing recovery sync for up to {args.recovery_seconds:.0f}s",
                flush=True,
            )

        now = time.monotonic()
        if now >= next_token_refresh:
            try:
                refresh_runtime_token(args.session)
                next_token_refresh = now + max(
                    60.0, args.token_refresh_seconds
                )
            except Exception as exc:
                print(
                    f"[checkpoint-download] token refresh failed; "
                    f"retrying in 60s: {exc}",
                    flush=True,
                )
                next_token_refresh = now + 60.0

        manifest = None
        try:
            manifest = sync_once(args)
        except Exception as exc:
            print(f"[checkpoint-download] retrying after error: {exc}", flush=True)

        if args.stop_file is not None and args.stop_file.exists():
            print(
                "[checkpoint-download] completion signal received after final sync",
                flush=True,
            )
            break
        if not watcher_alive:
            if manifest is not None and manifest.get("status") == "complete":
                print(
                    "[checkpoint-download] remote run reports complete",
                    flush=True,
                )
                break
            active = session_exists(args.session)
            if active is False:
                session_misses += 1
                if session_misses >= max(1, args.session_miss_limit):
                    print(
                        "[checkpoint-download] Colab session is no longer active",
                        flush=True,
                    )
                    break
            elif active is True:
                session_misses = 0
            if (
                recovery_deadline is not None
                and time.monotonic() >= recovery_deadline
            ):
                print(
                    "[checkpoint-download] recovery window expired",
                    flush=True,
                )
                break
        time.sleep(max(1.0, args.poll_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

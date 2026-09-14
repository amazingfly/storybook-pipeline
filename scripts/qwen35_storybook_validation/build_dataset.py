#!/usr/bin/env python3
"""Build durable, scene-sharded inputs for Qwen storybook validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


VERSION = "littlequeen-qwen35-validation-dataset-v3"
ARTIFACT_PRIORITY = ("grip_repaired", "wand_only", "regalia", "base")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_entry(path: Path, archive_path: str) -> dict:
    return {
        "archive_path": archive_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_tar(path: Path, files: dict[str, Path]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
        for archive_path, source in sorted(files.items()):
            archive.add(source, arcname=archive_path, recursive=False)
    temporary.replace(path)


def split_archive(
    source: Path,
    destination: Path,
    chunk_bytes: int,
) -> list[dict]:
    parts = []
    with source.open("rb") as stream:
        part_number = 0
        while chunk := stream.read(chunk_bytes):
            name = f"{source.name}.part_{part_number:03d}"
            path = destination / name
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(chunk)
            temporary.replace(path)
            parts.append(
                {
                    "name": name,
                    "bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                }
            )
            part_number += 1
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes-per-shard", type=int, default=25)
    parser.add_argument("--transfer-part-mb", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.scenes_per_shard < 1:
        raise ValueError("--scenes-per-shard must be positive")
    if args.transfer_part_mb < 8:
        raise ValueError("--transfer-part-mb must be at least 8")

    compiled_path = run_root / "compiled_story.json"
    summary_path = run_root / "summary.json"
    if not compiled_path.is_file():
        raise FileNotFoundError(compiled_path)
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    summary_records = {
        str(record["id"]): record for record in summary.get("records", [])
    }
    candidates = compiled.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("compiled story has no candidates")

    output.mkdir(parents=True, exist_ok=True)
    shard_files: dict[str, dict[str, Path]] = defaultdict(dict)
    records = []
    reference_sources: dict[str, Path] = {}
    reference_entries = []

    for scene in compiled.get("scenes", []):
        for lora in scene.get("base_loras", []):
            for reference in lora.get("validation_references", []):
                source = Path(reference).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"missing identity reference: {source}")
                key = sha256(source)
                archive_path = (
                    f"control/references/{key[:16]}_{source.name}"
                )
                if archive_path not in reference_sources:
                    reference_sources[archive_path] = source
                    entry = file_entry(source, archive_path)
                    entry["source_name"] = source.name
                    reference_entries.append(entry)

    for candidate in candidates:
        record_id = str(candidate["id"])
        scene_number = int(candidate["scene"])
        candidate_index = int(candidate["candidate_index"])
        shard_start = (
            ((scene_number - 1) // args.scenes_per_shard)
            * args.scenes_per_shard
            + 1
        )
        shard_end = min(
            shard_start + args.scenes_per_shard - 1,
            len(compiled["scenes"]),
        )
        shard_name = f"scenes_{shard_start:03d}_{shard_end:03d}.tar"
        artifacts = {}

        for artifact_type in ARTIFACT_PRIORITY:
            source = run_root / artifact_type / f"{record_id}.png"
            if not source.is_file():
                continue
            archive_path = f"dataset/{artifact_type}/{record_id}.png"
            artifacts[artifact_type] = file_entry(source, archive_path)
            shard_files[shard_name][archive_path] = source

        if "base" not in artifacts:
            raise FileNotFoundError(f"missing base image for {record_id}")
        summary_record = summary_records.get(record_id, {})
        final_relative = summary_record.get("artifacts", {}).get("final")
        final_type = (
            Path(final_relative).parts[0]
            if isinstance(final_relative, str) and final_relative
            else None
        )
        if final_type in artifacts:
            preferred_type = final_type
        else:
            variant_validation = summary_record.get("variant_validation", {})
            accepted_variants = [
                artifact_type
                for artifact_type in ("grip_repaired", "wand_only")
                if artifact_type in artifacts
                and variant_validation.get(artifact_type, {}).get("accepted")
            ]
            preferred_type = next(
                iter(accepted_variants),
                next(
                    artifact_type
                    for artifact_type in ARTIFACT_PRIORITY
                    if artifact_type in artifacts
                ),
            )

        metadata_source = run_root / "metadata" / f"{record_id}.json"
        metadata = None
        if metadata_source.is_file():
            archive_path = f"dataset/metadata/{record_id}.json"
            metadata = file_entry(metadata_source, archive_path)
            shard_files[shard_name][archive_path] = metadata_source

        scene_contract = compiled["scenes"][scene_number - 1]
        records.append(
            {
                "id": record_id,
                "scene": scene_number,
                "candidate_index": candidate_index,
                "seed": candidate.get("seed"),
                "prompt": scene_contract["prompt"],
                "script": scene_contract.get("script", ""),
                "lora_bindings": scene_contract.get("lora_bindings", {}),
                "base_loras": scene_contract.get("base_loras", []),
                "extra_loras": scene_contract.get("extra_loras", []),
                "artifacts": artifacts,
                "preferred_artifact": preferred_type,
                "metadata": metadata,
                "shard": shard_name,
            }
        )

    index = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_root": str(run_root),
        "title": compiled.get("title"),
        "story_version": compiled.get("version"),
        "scene_count": len(compiled["scenes"]),
        "candidate_count": len(records),
        "scenes_per_shard": args.scenes_per_shard,
        "artifact_priority": list(ARTIFACT_PRIORITY),
        "identity_references": reference_entries,
        "records": records,
    }
    index_path = output / "dataset_index.json"
    atomic_json(index_path, index)

    control_files = dict(reference_sources)
    control_files["control/compiled_story.json"] = compiled_path
    control_files["control/dataset_index.json"] = index_path
    if summary_path.is_file():
        control_files["control/source_summary.json"] = summary_path
    control_path = output / "control.tar"
    build_tar(control_path, control_files)

    archive_entries = []
    for shard_name, files in sorted(shard_files.items()):
        shard_path = output / shard_name
        build_tar(shard_path, files)
        scene_numbers = sorted(
            {
                int(path.split("/")[2].split("_")[1])
                for path in files
                if path.startswith("dataset/base/")
            }
        )
        archive_entries.append(
            {
                "name": shard_name,
                "kind": "scene_shard",
                "scene_start": min(scene_numbers),
                "scene_end": max(scene_numbers),
                "file_count": len(files),
                "bytes": shard_path.stat().st_size,
                "sha256": sha256(shard_path),
            }
        )

    archive_entries.insert(
        0,
        {
            "name": control_path.name,
            "kind": "control",
            "file_count": len(control_files),
            "bytes": control_path.stat().st_size,
            "sha256": sha256(control_path),
        },
    )
    transfer = output / "transfer"
    shutil.rmtree(transfer, ignore_errors=True)
    transfer.mkdir(parents=True)
    for entry in archive_entries:
        entry["parts"] = split_archive(
            output / entry["name"],
            transfer,
            args.transfer_part_mb * 1024 * 1024,
        )

    manifest = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transfer_part_bytes": args.transfer_part_mb * 1024 * 1024,
        "dataset_index": {
            "name": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256(index_path),
        },
        "archives": archive_entries,
        "total_archive_bytes": sum(item["bytes"] for item in archive_entries),
    }
    manifest_path = output / "archive_manifest.json"
    atomic_json(manifest_path, manifest)
    shutil.copy2(index_path, transfer / index_path.name)
    shutil.copy2(manifest_path, transfer / manifest_path.name)
    print(
        json.dumps(
            {
                "output": str(output),
                "scenes": index["scene_count"],
                "candidates": index["candidate_count"],
                "archives": len(archive_entries),
                "archive_bytes": manifest["total_archive_bytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

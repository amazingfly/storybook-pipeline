#!/usr/bin/env python3
"""Package only the inputs needed to resume FP16 accessory refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from PIL import Image


VERSION = "storybook-accessory-refinement-input-v1"
CHECKPOINT_VERSION = "storybook-durable-checkpoint-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-story", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_image(path: Path, size: tuple[int, int]) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == size
    except OSError:
        return False


def read_checkpoint(path: Path | None, compiled_sha256: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "version": CHECKPOINT_VERSION,
            "compiled_story_sha256": compiled_sha256,
            "batches": [],
            "files": {},
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(f"unexpected checkpoint version in {path}")
    if checkpoint.get("compiled_story_sha256") != compiled_sha256:
        raise RuntimeError("refinement checkpoint belongs to a different story")
    return checkpoint


def add_file(
    run_root: Path,
    relative_path: str,
    files: dict[str, dict[str, Any]],
) -> None:
    path = run_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    files[relative_path] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    args = parse_args()
    compiled_path = args.compiled_story.resolve()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    config = json.loads(compiled_path.read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    compiled_digest = sha256(compiled_path)
    checkpoint = read_checkpoint(args.checkpoint_manifest, compiled_digest)
    completed = checkpoint.get("files", {})
    width, height = (int(value) for value in config["settings"]["resolution"])
    size = (width, height)

    candidates = {candidate["id"]: candidate for candidate in config["candidates"]}
    records = []
    files: dict[str, dict[str, Any]] = {}
    for source_record in summary["records"]:
        if not source_record.get("uses_moonstar_accessories"):
            continue
        if not source_record.get("preflight_validation", {}).get("accepted", False):
            continue
        page_id = source_record["id"]
        candidate = candidates[page_id]
        artifacts = source_record["artifacts"]
        record = {
            "id": page_id,
            "seed": int(source_record["seed"]),
            "moonstar_weight": next(
                float(item["weight"])
                for item in candidate["extra_loras"]
                if item["mode"] == "moonstar_regional_set"
            ),
            "artifacts": {
                "composite": artifacts["composite"],
                "metadata": artifacts["metadata"],
                "regalia_mask": artifacts["regalia_mask"],
                "wand_mask": artifacts["wand_mask"],
                "hand_grip_mask": artifacts["hand_grip_mask"],
                "regalia": f"regalia/{page_id}.png",
                "wand_only": f"wand_only/{page_id}.png",
                "grip_repaired": f"grip_repaired/{page_id}.png",
            },
        }
        records.append(record)

        grip_relative = record["artifacts"]["grip_repaired"]
        wand_relative = record["artifacts"]["wand_only"]
        regalia_relative = record["artifacts"]["regalia"]
        if grip_relative in completed:
            if not valid_image(run_root / grip_relative, size):
                raise RuntimeError(
                    f"checkpoint marks {grip_relative} complete but local output is missing"
                )
            continue
        if wand_relative in completed:
            if not valid_image(run_root / wand_relative, size):
                raise RuntimeError(
                    f"checkpoint marks {wand_relative} complete but local output is missing"
                )
            add_file(run_root, wand_relative, files)
            add_file(run_root, artifacts["hand_grip_mask"], files)
            continue
        if regalia_relative in completed:
            if not valid_image(run_root / regalia_relative, size):
                raise RuntimeError(
                    f"checkpoint marks {regalia_relative} complete but local output is missing"
                )
            add_file(run_root, regalia_relative, files)
            add_file(run_root, artifacts["wand_mask"], files)
            add_file(run_root, artifacts["hand_grip_mask"], files)
            continue
        for key in (
            "composite",
            "metadata",
            "regalia_mask",
            "wand_mask",
            "hand_grip_mask",
        ):
            add_file(run_root, artifacts[key], files)

    if not records:
        raise RuntimeError("no accepted Moonstar accessory records were found")

    manifest = {
        "version": VERSION,
        "compiled_story_sha256": compiled_digest,
        "resolution": [width, height],
        "record_count": len(records),
        "completed_stage_file_count": sum(
            relative in completed
            for record in records
            for relative in (
                record["artifacts"]["regalia"],
                record["artifacts"]["wand_only"],
                record["artifacts"]["grip_repaired"],
            )
        ),
        "records": records,
        "files": files,
    }
    staging_manifest = run_root / ".refinement_input_manifest.json"
    staging_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with tarfile.open(temporary, "w") as archive:
        archive.add(compiled_path, arcname="compiled_story.json")
        archive.add(staging_manifest, arcname="refinement_input_manifest.json")
        for relative_path in sorted(files):
            archive.add(run_root / relative_path, arcname=relative_path)
    temporary.replace(output)
    staging_manifest.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "archive": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "records": len(records),
                "packaged_files": len(files),
                "completed_stage_files": manifest["completed_stage_file_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

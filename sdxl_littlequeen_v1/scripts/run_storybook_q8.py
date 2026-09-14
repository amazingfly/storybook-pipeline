#!/usr/bin/env python3
"""Generate compiled storybook base candidates with local Q8 SDXL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from run_sdxl_cpu import (
    ROOT,
    reject_conflicting_generators,
    render_one,
    valid_png,
    validate_inputs,
)


CHECKPOINT_VERSION = "storybook-durable-checkpoint-v1"
STORYBOOK_STYLE = "polished anime storybook illustration"
COMMON_NEGATIVE_PROMPT = (
    "photograph, photorealistic, 3d render, text, watermark, logo, duplicate character, "
    "extra limbs, malformed hands, distorted face"
)
MOONSTAR_BASE_NEGATIVE = (
    "crown, tiara, hat, hair ornament, ornate wand, staff, sword, weapon, floating rod, "
    "angled rod, open palm, jewelry, cropped head, cropped hands, cropped feet, hidden hands"
)
LORA_FILE_NAMES = {
    "little_queen_v2": "lqxl_sdxl_v2",
    "moon_dress_v1": "lqmoonfit_sdxl_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiled_story", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--limit-new", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    return parser.parse_args()


def has_moonstar(candidate: dict[str, Any]) -> bool:
    return any(
        item["mode"] == "moonstar_regional_set"
        for item in candidate["extra_loras"]
    )


def adapter_signature(candidate: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    return tuple(
        (item["adapter"], float(item["weight"]))
        for item in candidate["base_loras"]
    )


def build_prompt(candidate: dict[str, Any]) -> str:
    staging = []
    if has_moonstar(candidate):
        staging = [
            "Little Queen centered foreground, full body visible, bare head",
            "one hand gripping a vertical guide rod",
        ]
    return ", ".join(
        [
            *[item["trigger"] for item in candidate["base_loras"]],
            candidate["prompt"],
            *staging,
            STORYBOOK_STYLE,
        ]
    )


def build_negative(candidate: dict[str, Any]) -> str:
    parts = [COMMON_NEGATIVE_PROMPT]
    if has_moonstar(candidate):
        parts.append(MOONSTAR_BASE_NEGATIVE)
    return ", ".join(parts)


def local_loras(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    result = [{"name": "lcm-lora-sdxl", "weight": 1.0}]
    for item in candidate["base_loras"]:
        name = LORA_FILE_NAMES.get(item["name"])
        if name is None:
            raise RuntimeError(f"no local Q8 LoRA mapping for {item['name']}")
        result.append({"name": name, "weight": float(item["weight"])})
    return result


def valid_story_image(path: Path, size: tuple[int, int]) -> bool:
    if not valid_png(path):
        return False
    try:
        with Image.open(path) as image:
            return image.size == size
    except OSError:
        return False


def validate_checkpoint(
    checkpoint_dir: Path,
    compiled_sha: str,
) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError("unsupported checkpoint version")
    if manifest.get("compiled_story_sha256") != compiled_sha:
        raise RuntimeError("checkpoint belongs to a different compiled story")
    for batch in manifest["batches"]:
        archive = checkpoint_dir / batch["archive"]
        if not archive.is_file():
            raise RuntimeError(f"checkpoint archive is missing: {archive}")
        if archive.stat().st_size != int(batch["bytes"]):
            raise RuntimeError(f"checkpoint archive size mismatch: {archive}")
        if sha256(archive) != batch["sha256"]:
            raise RuntimeError(f"checkpoint archive checksum mismatch: {archive}")
    return manifest


def restore_checkpoint(
    checkpoint_dir: Path,
    run_root: Path,
    compiled_sha: str,
) -> dict[str, Any]:
    manifest = validate_checkpoint(checkpoint_dir, compiled_sha)
    restored = 0
    for batch in manifest["batches"]:
        archive = checkpoint_dir / batch["archive"]
        with tarfile.open(archive, "r") as stream:
            stream.extractall(run_root, filter="data")
        restored += 1
    print(
        f"[q8 checkpoint] restored {restored} verified batches with "
        f"{len(manifest['files'])} files",
        flush=True,
    )
    return manifest


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    compiled_path = args.compiled_story.resolve()
    run_root = args.run_root.resolve()
    config = json.loads(compiled_path.read_text(encoding="utf-8"))
    compiled_sha = sha256(compiled_path)
    width, height = (int(value) for value in config["settings"]["resolution"])
    image_size = (width, height)

    reject_conflicting_generators()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "base").mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs" / "q8_generation"
    log_dir.mkdir(parents=True, exist_ok=True)
    local_compiled = (run_root / "compiled_story.json").resolve()
    if compiled_path != local_compiled:
        shutil.copy2(compiled_path, local_compiled)

    checkpoint_manifest = None
    if args.checkpoint_dir is not None and not args.fresh:
        checkpoint_manifest = restore_checkpoint(
            args.checkpoint_dir.resolve(),
            run_root,
            compiled_sha,
        )

    state_path = run_root / "q8_generation_state.json"
    if state_path.is_file() and not args.fresh:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("compiled_story_sha256") != compiled_sha:
            raise RuntimeError("existing Q8 state belongs to a different compiled story")
    else:
        state = {
            "version": "storybook-q8-generation-v1",
            "title": config["title"],
            "compiled_story_sha256": compiled_sha,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "resolution": [width, height],
            "candidate_count": len(config["candidates"]),
            "results": {},
        }
    if checkpoint_manifest is not None:
        state["source_checkpoint"] = str(args.checkpoint_dir.resolve())
        state["source_checkpoint_files"] = len(checkpoint_manifest["files"])
    state["status"] = "running"
    state["resumed_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(state_path, state)

    cpu_config = {
        "name": "littlequeen_storybook_q8_local",
        "binary": str(
            ROOT / ".tooling/stable-diffusion.cpp/build-cpu/bin/sd-cli"
        ),
        "model": str(ROOT / "models/quantized/sd_xl_base_1.0-q8_0.gguf"),
        "lora_dir": str(ROOT / "models/loras"),
        "generation": {
            "width": width,
            "height": height,
            "steps": 8,
            "cfg_scale": 1.0,
            "sampling_method": "lcm",
            "scheduler": "discrete",
            "threads": 6,
            "rng": "cpu",
            "mmap": True,
            "vae_tiling": args.vae_tiling,
            "lora_apply_mode": "at_runtime",
        },
        "memory": {
            "sample_interval_seconds": 1.0,
            "minimum_available_mb": 1200,
            "maximum_process_rss_mb": 12500,
            "maximum_swap_used_mb": 4096,
        },
        "loras": [],
        "prompts": [
            {"prompt": "preflight", "seed": 1, "loras": local_loras(candidate)}
            for candidate in config["candidates"]
        ],
    }
    validate_inputs(cpu_config)

    generation_order = sorted(config["candidates"], key=adapter_signature)
    generated_now = 0
    completed = 0
    for index, candidate in enumerate(generation_order, start=1):
        relative = Path("base") / f"{candidate['id']}.png"
        destination = run_root / relative
        if valid_story_image(destination, image_size):
            completed += 1
            if candidate["id"] not in state["results"]:
                state["results"][candidate["id"]] = {
                    "index": index,
                    "id": candidate["id"],
                    "seed": int(candidate["seed"]),
                    "output": str(relative),
                    "status": "completed",
                    "source": "restored_checkpoint",
                }
            print(
                f"[q8 {index}/{len(generation_order)}] skip {candidate['id']}",
                flush=True,
            )
            continue
        if args.limit_new is not None and generated_now >= args.limit_new:
            break
        record = {
            "prompt": build_prompt(candidate),
            "negative_prompt": build_negative(candidate),
            "seed": int(candidate["seed"]),
            "loras": local_loras(candidate),
            "output_relative": str(relative),
            "log_stem": candidate["id"],
        }
        print(
            f"[q8 generate {index}/{len(generation_order)}] {candidate['id']}",
            flush=True,
        )
        result = render_one(cpu_config, record, index, run_root, log_dir)
        result.update(
            {
                "id": candidate["id"],
                "scene_number": int(candidate["scene"]),
                "candidate_index": int(candidate["candidate_index"]),
                "output": str(relative),
                "source": "local_q8",
            }
        )
        state["results"][candidate["id"]] = result
        state["last_completed_id"] = candidate["id"]
        state["last_completed_index"] = index
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(state_path, state)
        if result["status"] != "completed":
            state["status"] = "failed"
            write_json(state_path, state)
            return 1
        generated_now += 1
        completed += 1

    completed = sum(
        valid_story_image(
            run_root / "base" / f"{candidate['id']}.png",
            image_size,
        )
        for candidate in generation_order
    )
    state["completed_count"] = completed
    state["generated_this_invocation"] = generated_now
    state["status"] = (
        "base_complete" if completed == len(generation_order) else "paused"
    )
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(state_path, state)
    print(
        f"[q8] {state['status']}: {completed}/{len(generation_order)} valid base images; "
        f"{generated_now} generated this invocation",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise

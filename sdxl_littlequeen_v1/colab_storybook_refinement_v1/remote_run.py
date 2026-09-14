#!/usr/bin/env python3
"""Run resumable FP16 SDXL storybook accessory refinement on Colab."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


ROOT = Path("/content/lq_storybook_refinement_v1")
INPUT_ARCHIVE = Path("/content/storybook_refinement_inputs.tar")
INPUT_MANIFEST_VERSION = "storybook-accessory-refinement-input-v1"
CHECKPOINT_ROOT = Path("/content/storybook_refinement_checkpoints")
CHECKPOINT_UPLOAD = Path("/content/storybook_refinement_checkpoint_manifest.json")
SHARED_PATH = Path("/content/remote_storybook_mvp_v1_shared.py")
EXPECTED_INPUT_SHA256 = os.environ["STORYBOOK_REFINEMENT_INPUT_SHA256"]
INPUT_PREFIX = Path(
    f"/content/storybook_refinement_inputs_"
    f"{EXPECTED_INPUT_SHA256[:16]}.tar.part_"
)

REGALIA_PROMPT = (
    "lqmoonregalia4, exact five-peak polished gold Moonstar crown, centered oval "
    "rose-amber moonstone, two amber side gems, matched gold leaf-drop earrings, "
    "narrow gold moonstone collar, clean auburn hair and natural background above "
    "the crown, integrated anime storybook linework"
)
REGALIA_NEGATIVE = (
    "different crown, giant crown, extra crown peaks, flower crown, roses, hat, "
    "archway, cage, gold clothing, extra jewelry, changed face, changed hair, "
    "changed dress"
)
WAND_PROMPT = (
    "lqmoonwand4, exact single slender gold Moonstar scepter, straight engraved "
    "shaft, open crescent finial around one rose-amber oval moonstone, five leaf "
    "crystals, two round side gems, pointed rose crystal end cap, one small natural "
    "hand firmly wrapped around the shaft with visible fingers, anime storybook linework"
)
WAND_NEGATIVE = (
    "different wand, giant staff, sword, blade, star wand, flower, vine, archway, "
    "cage, freestanding object, floating wand, open palm, extra wand, duplicate, "
    "bent shaft, extra fingers, fused fingers, changed body"
)
GRIP_PROMPT = (
    "small natural child hand tightly closed around the slender vertical gold wand "
    "shaft, thumb crossing in front, four curled fingers visibly wrapped behind the "
    "shaft, correct wrist and anatomy, clean anime storybook linework"
)
GRIP_NEGATIVE = (
    "open palm, straight fingers, floating wand, hand beside wand, missing thumb, "
    "extra fingers, fused fingers, deformed hand, extra hand, changed sleeve"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble() -> None:
    parts = sorted(INPUT_PREFIX.parent.glob(INPUT_PREFIX.name + "*"))
    if not parts:
        raise FileNotFoundError(f"no chunks match {INPUT_PREFIX}*")
    temporary = INPUT_ARCHIVE.with_suffix(".tar.part")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    output.write(chunk)
    if sha256(temporary) != EXPECTED_INPUT_SHA256:
        raise RuntimeError("refinement input archive checksum mismatch")
    temporary.replace(INPUT_ARCHIVE)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pipe(shared):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLInpaintPipeline

    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        str(shared.MODEL),
        config=shared.MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.load_lora_weights(str(shared.REGALIA), adapter_name="regalia")
    pipe.load_lora_weights(str(shared.WAND), adapter_name="wand")
    pipe.to("cuda")
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    return pipe


def valid_image(path: Path, size: tuple[int, int]) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.size == size
    except OSError:
        return False


def exact_mask_merge(initial, generated, mask):
    from PIL import Image

    source = initial.convert("RGB")
    result = generated.convert("RGB")
    alpha = mask.convert("L")
    if result.size != source.size:
        result = result.resize(source.size, Image.Resampling.LANCZOS)
    return Image.composite(result, source, alpha)


def write_summary(summary: dict) -> None:
    path = ROOT / "refinement_summary.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    lock = Path("/content/storybook_refinement_v1.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another accessory refinement process is active") from exc

    shared = load_module("storybook_shared", SHARED_PATH)
    shared.install_dependencies()
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    shared.run(["nvidia-smi"])
    assemble()
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(INPUT_ARCHIVE, "r") as archive:
        archive.extractall(ROOT, filter="data")

    input_manifest_path = ROOT / "refinement_input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("version") != INPUT_MANIFEST_VERSION:
        raise RuntimeError("unexpected refinement input manifest version")
    compiled_path = ROOT / "compiled_story.json"
    if sha256(compiled_path) != input_manifest["compiled_story_sha256"]:
        raise RuntimeError("compiled story checksum mismatch")
    for relative, expected in input_manifest["files"].items():
        path = ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise RuntimeError(f"refinement input verification failed: {relative}")

    config = json.loads(compiled_path.read_text(encoding="utf-8"))
    records = input_manifest["records"]
    width, height = (int(value) for value in input_manifest["resolution"])
    size = (width, height)
    for directory in ("regalia", "wand_only", "grip_repaired"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    shared.CHECKPOINT_MANIFEST_UPLOAD = CHECKPOINT_UPLOAD
    shared.CHECKPOINT_DIRECTORIES = ("regalia", "wand_only", "grip_repaired")
    shared.GENERATION_ONLY = True
    checkpoint_config = {
        "title": config["title"],
        "candidates": [{"id": record["id"]} for record in records],
    }
    checkpoint = shared.DurableCheckpoint(
        checkpoint_config,
        run_root=ROOT,
        drive_root=CHECKPOINT_ROOT,
        interval=5,
    )
    checkpoint.restore()
    completed = checkpoint.manifest["files"]
    checkpoint.known = {
        relative: checkpoint._state(ROOT / relative)
        for relative in completed
        if (ROOT / relative).is_file()
    }

    shared.ensure_loras({"moonstar_regalia", "moonstar_wand"})
    shared.ensure_model()
    pipe = make_pipe(shared)
    shared.assert_prompts_fit(
        pipe,
        {
            "regalia positive": REGALIA_PROMPT,
            "regalia negative": REGALIA_NEGATIVE,
            "wand positive": WAND_PROMPT,
            "wand negative": WAND_NEGATIVE,
            "grip positive": GRIP_PROMPT,
            "grip negative": GRIP_NEGATIVE,
        },
    )

    total = len(records) * 3
    summary = {
        "version": "storybook-accessory-refinement-run-v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "stage_count": total,
        "precision": "fp16",
        "model": shared.MODEL_ID,
        "records": [],
    }
    write_summary(summary)
    regional = shared.REGIONAL

    def run_stage(
        *,
        position: int,
        stage: str,
        record: dict,
        initial_relative: str,
        mask_relative: str,
        destination_relative: str,
        prompt: str,
        negative_prompt: str,
        adapter: str,
        adapter_weight: float,
        strength: float,
        padding: int,
        seed_offset: int,
    ) -> None:
        destination = ROOT / destination_relative
        if destination_relative in completed:
            if not valid_image(destination, size):
                raise RuntimeError(
                    f"manifest-completed dependency is missing: {destination_relative}"
                )
            print(
                f"[refine {position}/{total}] manifest reuse {stage} {record['id']}",
                flush=True,
            )
            return
        initial_path = ROOT / initial_relative
        mask_path = ROOT / mask_relative
        if not valid_image(initial_path, size) or not valid_image(mask_path, size):
            raise RuntimeError(
                f"missing input for {stage} {record['id']}: "
                f"{initial_relative} or {mask_relative}"
            )
        print(f"[refine {position}/{total}] {stage} {record['id']}", flush=True)
        pipe.set_adapters([adapter], adapter_weights=[adapter_weight])
        initial = Image.open(initial_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        started = time.monotonic()
        with torch.inference_mode():
            generated = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial,
                mask_image=mask,
                width=width,
                height=height,
                strength=strength,
                num_inference_steps=int(regional["steps"]),
                guidance_scale=float(regional["guidance_scale"]),
                padding_mask_crop=padding,
                generator=torch.Generator(device="cuda").manual_seed(
                    int(record["seed"]) + seed_offset
                ),
            ).images[0]
        result = exact_mask_merge(initial, generated, mask)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.save(destination, format="PNG", compress_level=6)
        duration = time.monotonic() - started
        print(
            f"[refined {position}/{total}] {stage} {record['id']} "
            f"in {duration:.2f}s",
            flush=True,
        )
        summary["records"].append(
            {
                "id": record["id"],
                "stage": stage,
                "duration_seconds": round(duration, 2),
                "output": destination_relative,
            }
        )
        write_summary(summary)
        checkpoint.tick()
        del initial, mask, generated, result
        torch.cuda.empty_cache()

    try:
        for index, record in enumerate(records, start=1):
            artifacts = record["artifacts"]
            grip_relative = artifacts["grip_repaired"]
            wand_relative = artifacts["wand_only"]
            if grip_relative in completed:
                for offset, stage in enumerate(("regalia", "wand", "grip")):
                    print(
                        f"[refine {(index - 1) * 3 + offset + 1}/{total}] "
                        f"manifest skip {stage} {record['id']}",
                        flush=True,
                    )
                continue
            weight = float(record["moonstar_weight"])
            if wand_relative in completed:
                if not valid_image(ROOT / wand_relative, size):
                    raise RuntimeError(
                        f"manifest-completed dependency is missing: {wand_relative}"
                    )
                print(
                    f"[refine {(index - 1) * 3 + 1}/{total}] "
                    f"manifest skip regalia {record['id']}",
                    flush=True,
                )
                print(
                    f"[refine {(index - 1) * 3 + 2}/{total}] "
                    f"manifest reuse wand {record['id']}",
                    flush=True,
                )
                run_stage(
                    position=(index - 1) * 3 + 3,
                    stage="grip",
                    record=record,
                    initial_relative=artifacts["wand_only"],
                    mask_relative=artifacts["hand_grip_mask"],
                    destination_relative=artifacts["grip_repaired"],
                    prompt=GRIP_PROMPT,
                    negative_prompt=GRIP_NEGATIVE,
                    adapter="wand",
                    adapter_weight=float(regional["grip_wand_weight"]) * weight,
                    strength=float(regional["grip_strength"]),
                    padding=int(regional["grip_padding_mask_crop"]),
                    seed_offset=1200,
                )
                continue
            metadata = json.loads(
                (ROOT / artifacts["metadata"]).read_text(encoding="utf-8")
            ) if (ROOT / artifacts["metadata"]).is_file() else {}
            regalia_strength = (
                float(regional["crown_cleanup_strength"])
                if metadata.get("existing_crown_removed")
                else float(regional["regalia_strength"])
            )
            run_stage(
                position=(index - 1) * 3 + 1,
                stage="regalia",
                record=record,
                initial_relative=artifacts["composite"],
                mask_relative=artifacts["regalia_mask"],
                destination_relative=artifacts["regalia"],
                prompt=REGALIA_PROMPT,
                negative_prompt=REGALIA_NEGATIVE,
                adapter="regalia",
                adapter_weight=float(regional["regalia_weight"]) * weight,
                strength=regalia_strength,
                padding=int(regional["padding_mask_crop"]),
                seed_offset=500,
            )
            run_stage(
                position=(index - 1) * 3 + 2,
                stage="wand",
                record=record,
                initial_relative=artifacts["regalia"],
                mask_relative=artifacts["wand_mask"],
                destination_relative=artifacts["wand_only"],
                prompt=WAND_PROMPT,
                negative_prompt=WAND_NEGATIVE,
                adapter="wand",
                adapter_weight=float(regional["wand_weight"]) * weight,
                strength=float(regional["wand_strength"]),
                padding=int(regional["padding_mask_crop"]),
                seed_offset=900,
            )
            run_stage(
                position=(index - 1) * 3 + 3,
                stage="grip",
                record=record,
                initial_relative=artifacts["wand_only"],
                mask_relative=artifacts["hand_grip_mask"],
                destination_relative=artifacts["grip_repaired"],
                prompt=GRIP_PROMPT,
                negative_prompt=GRIP_NEGATIVE,
                adapter="wand",
                adapter_weight=float(regional["grip_wand_weight"]) * weight,
                strength=float(regional["grip_strength"]),
                padding=int(regional["grip_padding_mask_crop"]),
                seed_offset=1200,
            )
        checkpoint.flush(force=True)
        checkpoint.mark_complete()
        summary["status"] = "complete"
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_summary(summary)
        print(
            f"FP16 accessory refinement complete: {len(records)} records, "
            f"{total} stages.",
            flush=True,
        )
        return 0
    finally:
        try:
            checkpoint.flush(force=True)
        except Exception as exc:
            print(f"[checkpoint] final flush failed: {exc}", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

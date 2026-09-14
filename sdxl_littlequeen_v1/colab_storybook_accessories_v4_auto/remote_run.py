#!/usr/bin/env python3
"""Run unattended Little Queen generation, canonical masking, refinement, and validation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


ROOT = Path("/content/lq_storybook_accessories_v4_auto")
BUNDLE_PREFIX = Path("/content/storybook_accessories_v4_auto_bundle.tar.gz.part_")
BASES_PREFIX = Path("/content/storybook_accessories_v4_auto_bases.tar.gz.part_")
IDENTITY_PREFIX = Path("/content/lqxl_sdxl_v2.safetensors.part_")
OUTFIT_PREFIX = Path("/content/lqmoonfit_sdxl_v1.safetensors.part_")
REGALIA_PREFIX = Path("/content/lqmoonregalia_sdxl_v4.safetensors.part_")
WAND_PREFIX = Path("/content/lqmoonwand_sdxl_v4.safetensors.part_")
MODEL = ROOT / "models" / "sd_xl_base_1.0.safetensors"
IDENTITY = ROOT / "models" / "lqxl_sdxl_v2.safetensors"
OUTFIT = ROOT / "models" / "lqmoonfit_sdxl_v1.safetensors"
REGALIA = ROOT / "models" / "lqmoonregalia_sdxl_v4.safetensors"
WAND = ROOT / "models" / "lqmoonwand_sdxl_v4.safetensors"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
ARCHIVE = Path("/content/lq_storybook_accessories_v4_auto_results.tar.gz")

UPLOADS = (
    (
        IDENTITY_PREFIX,
        IDENTITY,
        170_552_852,
        "3799128d4bfd4fbc7848d5b1de099a3e99cf3f718845c98efa7d4d48a7953994",
    ),
    (
        OUTFIT_PREFIX,
        OUTFIT,
        170_545_956,
        "985e82df407690831ef8d27a9c6c3b061689278c334e3694e21eaf8aa88c36d7",
    ),
    (
        REGALIA_PREFIX,
        REGALIA,
        85_424_812,
        "8d3757aa6cfc0b55b5af5c653075b961709c321077c2afabcc79407e4ca29014",
    ),
    (
        WAND_PREFIX,
        WAND,
        85_424_748,
        "4714e064ae0bbc31c33b3398a21b98227d764fb8b0a0024932bb6f3359a34503",
    ),
)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(
    prefix: Path,
    destination: Path,
    expected_bytes: int | None = None,
    expected_sha: str | None = None,
) -> None:
    parts = sorted(prefix.parent.glob(prefix.name + "*"))
    if not parts:
        raise FileNotFoundError(f"no chunks match {prefix}*")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
        raise RuntimeError(f"size mismatch for {destination}")
    if expected_sha is not None and sha256(temporary) != expected_sha:
        raise RuntimeError(f"checksum mismatch for {destination}")
    temporary.replace(destination)


def install_dependencies() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "diffusers==0.34.0",
            "transformers==4.49.0",
            "accelerate==1.4.0",
            "peft==0.15.2",
            "safetensors>=0.4.5",
            "opencv-python-headless>=4.10,<5",
        ]
    )


def ensure_model() -> None:
    if MODEL.is_file() and sha256(MODEL) == MODEL_SHA256:
        return
    from huggingface_hub import hf_hub_download

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="sd_xl_base_1.0.safetensors",
            local_dir=MODEL.parent,
        )
    )
    if downloaded != MODEL:
        shutil.copy2(downloaded, MODEL)
    if sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("SDXL checksum mismatch")


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_txt2img(config: dict):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL),
        config=MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.load_lora_weights(str(IDENTITY), adapter_name="identity")
    pipe.load_lora_weights(str(OUTFIT), adapter_name="outfit")
    pipe.set_adapters(
        ["identity", "outfit"],
        adapter_weights=[
            float(config["generation"]["identity_weight"]),
            float(config["generation"]["outfit_weight"]),
        ],
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def make_inpaint(lora: Path, adapter: str, weight: float):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLInpaintPipeline

    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        str(MODEL),
        config=MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.load_lora_weights(str(lora), adapter_name=adapter)
    pipe.set_adapters([adapter], adapter_weights=[weight])
    pipe.to("cuda")
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    return pipe


def assert_prompts_fit(pipe, prompts: dict[str, str]) -> None:
    for tokenizer_name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipe, tokenizer_name, None)
        if tokenizer is None:
            continue
        limit = int(tokenizer.model_max_length)
        for label, prompt in prompts.items():
            count = len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                ).input_ids
            )
            if count > limit:
                raise RuntimeError(
                    f"{label} is {count} tokens for {tokenizer_name}; limit is {limit}"
                )


def gpu_sample() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    keys = ("gpu_used_mb", "gpu_total_mb", "gpu_util_pct", "gpu_temp_c")
    if result.returncode or not result.stdout.strip():
        return {key: "" for key in keys}
    return dict(
        zip(
            keys,
            (field.strip() for field in result.stdout.splitlines()[0].split(",")),
            strict=True,
        )
    )


def monitor_resources(destination: Path, stop: threading.Event) -> None:
    fields = [
        "timestamp_utc",
        "host_available_mb",
        "host_total_mb",
        "gpu_used_mb",
        "gpu_total_mb",
        "gpu_util_pct",
        "gpu_temp_c",
    ]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith(("MemAvailable:", "MemTotal:")):
                    key, value, _ = line.split()
                    memory[key.rstrip(":")] = round(int(value) / 1024)
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "host_available_mb": memory.get("MemAvailable", ""),
                    "host_total_mb": memory.get("MemTotal", ""),
                    **gpu_sample(),
                }
            )
            stream.flush()
            stop.wait(3)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def make_contact_sheet(
    records: list[dict],
    destination: Path,
    *,
    validation_labels: bool = False,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    columns = 4
    cell = (240, 370)
    rows = max(1, (len(records) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        artifacts = record.get("artifacts", {})
        chosen = artifacts.get("final") or artifacts.get("composite") or artifacts["base"]
        image = Image.open(ROOT / chosen).convert("RGB")
        thumb = ImageOps.contain(image, (cell[0] - 8, cell[1] - 48))
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        sheet.paste(thumb, (x + (cell[0] - thumb.width) // 2, y))
        if validation_labels:
            validation = record.get("validation", {})
            verdict = "PASS" if validation.get("accepted") else "REJECT"
            color = "#7ee787" if verdict == "PASS" else "#ff7b72"
            label = f"{record['id']} {verdict} score={validation.get('score', 0):.0f}"
            reason = "; ".join(validation.get("failures", [])[:1])
            draw.text((x + 4, y + cell[1] - 42), label, fill=color)
            if reason:
                draw.text((x + 4, y + cell[1] - 22), reason[:37], fill="white")
        else:
            draw.text((x + 4, y + cell[1] - 22), record["id"], fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=93)


def make_grip_contact_sheet(records: list[dict], destination: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    columns = 4
    cell = (280, 300)
    rows = max(1, (len(records) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        artifacts = record.get("artifacts", {})
        if "final" not in artifacts or "metadata" not in artifacts:
            continue
        metadata = json.loads(
            (ROOT / artifacts["metadata"]).read_text(encoding="utf-8")
        )
        hand_box = metadata["placements"]["staff"].get("detected_hand_box")
        if not hand_box:
            continue
        image = Image.open(ROOT / artifacts["final"]).convert("RGB")
        left, top, right, bottom = (float(value) for value in hand_box)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        side = min(
            max(max(right - left, bottom - top) * 4.2, 160.0),
            min(image.size) * 0.45,
        )
        crop = image.crop(
            (
                max(0, round(center_x - side / 2)),
                max(0, round(center_y - side / 2)),
                min(image.width, round(center_x + side / 2)),
                min(image.height, round(center_y + side / 2)),
            )
        )
        crop = ImageOps.contain(crop, (cell[0] - 12, cell[1] - 36))
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        sheet.paste(crop, (x + (cell[0] - crop.width) // 2, y))
        draw.text((x + 6, y + cell[1] - 27), record["id"], fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def write_summary(summary: dict) -> None:
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def build_archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for directory in (
            "base",
            "composite",
            "regalia",
            "final",
            "metadata",
            "review",
            "logs",
        ):
            path = ROOT / directory
            if path.is_dir():
                archive.add(path, arcname=directory)
        archive.add(ROOT / "summary.json", arcname="summary.json")
        archive.add(ROOT / "validation_report.json", arcname="validation_report.json")
        archive.add(ROOT / "auto_pipeline_config.json", arcname="auto_pipeline_config.json")
    print(f"Wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)", flush=True)


def main() -> int:
    install_dependencies()
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    run(["nvidia-smi"])
    resume = (
        (ROOT / "summary.json").is_file()
        and any((ROOT / "base").glob("candidate_*.png"))
        and MODEL.is_file()
    )
    if not resume:
        shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    assemble(BUNDLE_PREFIX, ROOT / "bundle.tar.gz")
    for upload in UPLOADS:
        _, destination, expected_bytes, expected_sha = upload
        if (
            destination.is_file()
            and destination.stat().st_size == expected_bytes
            and sha256(destination) == expected_sha
        ):
            continue
        assemble(*upload)
    with tarfile.open(ROOT / "bundle.tar.gz", "r:gz") as archive:
        archive.extractall(ROOT)
    if any(BASES_PREFIX.parent.glob(BASES_PREFIX.name + "*")):
        assemble(BASES_PREFIX, ROOT / "reuse_bases.tar.gz")
        with tarfile.open(ROOT / "reuse_bases.tar.gz", "r:gz") as archive:
            archive.extractall(ROOT)
    ensure_model()

    config = json.loads((ROOT / "auto_pipeline_config.json").read_text(encoding="utf-8"))
    width, height = (int(value) for value in config["resolution"])
    scenes = config["scenes"][: int(config["max_candidates"])]
    target = int(config["target_accepted"])
    seed_start = int(config["seed_start"])

    for directory in (
        "base",
        "composite",
        "regalia",
        "final",
        "metadata",
        "review",
        "logs",
    ):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    summary = {
        "name": "littlequeen_storybook_accessories_v4_unattended_colab_check",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "target_accepted": target,
        "max_candidates": len(scenes),
        "resolution": [width, height],
        "models": {
            "base": MODEL_ID,
            "identity_sha256": sha256(IDENTITY),
            "outfit_sha256": sha256(OUTFIT),
            "regalia_sha256": sha256(REGALIA),
            "wand_sha256": sha256(WAND),
        },
        "records": [],
    }
    if resume:
        summary["resumed_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(summary)
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_resources,
        args=(ROOT / "logs" / "resource_usage.csv", stop),
        daemon=True,
    )
    monitor.start()
    records: list[dict] = []
    summary["records"] = records

    positive_template = (
        "full body, solo, lqxl Little Queen, lqmoonfit, auburn hair, straight bangs, bare head, "
        "no jewelry, exactly two arms, one hand closed around a plain thin vertical wooden rod "
        "outside her body, other hand visible, head hands shoes visible, {scene}, anime storybook"
    )
    negative = (
        "crown, tiara, hat, hair ornament, ornate wand, jeweled staff, scepter, sword, weapon, "
        "floating rod, open palm beside rod, earrings, necklace, bracelet, cropped head, "
        "cropped hands, cropped feet, hidden hands, close-up, two people, duplicate, extra "
        "limbs, malformed hands, photorealistic, text, watermark"
    )

    try:
        pipe = None
        generation = config["generation"]
        for index, scene in enumerate(scenes, start=1):
            page_id = f"candidate_{index:02d}"
            seed = seed_start + index
            destination = ROOT / "base" / f"{page_id}.png"
            reused = False
            if destination.is_file():
                try:
                    with Image.open(destination) as existing:
                        reused = existing.size == (width, height)
                except Exception:
                    destination.unlink(missing_ok=True)
            if reused:
                print(f"[generate {index}/{len(scenes)}] reuse {page_id}", flush=True)
                duration = 0.0
            else:
                if pipe is None:
                    pipe = make_txt2img(config)
                    prompt_set = {"base negative": negative}
                    prompt_set.update(
                        {
                            f"base candidate {item_index}": positive_template.format(
                                scene=item_scene
                            )
                            for item_index, item_scene in enumerate(scenes, start=1)
                        }
                    )
                    assert_prompts_fit(pipe, prompt_set)
                print(f"[generate {index}/{len(scenes)}] {page_id}", flush=True)
                started = time.monotonic()
                with torch.inference_mode():
                    image = pipe(
                        prompt=positive_template.format(scene=scene),
                        negative_prompt=negative,
                        width=width,
                        height=height,
                        num_inference_steps=int(generation["steps"]),
                        guidance_scale=float(generation["guidance_scale"]),
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                    ).images[0]
                image.save(destination, format="PNG", compress_level=6)
                duration = round(time.monotonic() - started, 2)
                del image
            records.append(
                {
                    "id": page_id,
                    "seed": seed,
                    "scene": scene,
                    "duration_seconds": duration,
                    "reused_base": reused,
                    "artifacts": {"base": relative(destination)},
                }
            )
            write_summary(summary)
            torch.cuda.empty_cache()
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()
        make_contact_sheet(records, ROOT / "review" / "base_candidates.jpg")

        compositor = load_module(
            "lq_accessory_compositor",
            ROOT / "pipeline" / "apply_storybook_accessories.py",
        )
        validator = load_module(
            "lq_accessory_validator",
            ROOT / "pipeline" / "validate_storybook_accessories_v4.py",
        )
        set_path = ROOT / "storybook_accessories_v2" / "accessory_set.json"
        preflight_records = []
        for index, record in enumerate(records, start=1):
            page_id = record["id"]
            print(f"[mask {index}/{len(records)}] {page_id}", flush=True)
            composite = ROOT / "composite" / f"{page_id}.png"
            metadata = ROOT / "metadata" / f"{page_id}.json"
            regalia_mask = ROOT / "metadata" / f"{page_id}_regalia_mask.png"
            wand_mask = ROOT / "metadata" / f"{page_id}_wand_mask.png"
            try:
                arguments = argparse.Namespace(
                    input=ROOT / record["artifacts"]["base"],
                    output=composite,
                    set_path=set_path,
                    staff=True,
                    no_crown=False,
                    no_jewelry=False,
                    face_box=None,
                    staff_anchor=None,
                    staff_hand="auto",
                    staff_angle=0.0,
                    crown_scale=1.0,
                    staff_scale=1.0,
                    staff_anchor_outset=0.0,
                    remove_existing_crown=True,
                    existing_crown_min_score=0.45,
                    metadata=metadata,
                    blend_mask=ROOT / "metadata" / f"{page_id}_blend_mask.png",
                    cleanup_mask=ROOT / "metadata" / f"{page_id}_cleanup_mask.png",
                    regalia_mask=regalia_mask,
                    hand_prop_mask=wand_mask,
                    hand_prop_full_mask=ROOT
                    / "metadata"
                    / f"{page_id}_wand_full_mask.png",
                    hand_prop_visible_mask=ROOT
                    / "metadata"
                    / f"{page_id}_wand_visible_mask.png",
                    hand_occlusion_mask=ROOT
                    / "metadata"
                    / f"{page_id}_hand_occlusion_mask.png",
                    hand_grip_mask=ROOT
                    / "metadata"
                    / f"{page_id}_hand_grip_mask.png",
                    no_person_occlusion=False,
                    quiet=True,
                )
                compositor.main(arguments)
                record["artifacts"].update(
                    {
                        "composite": relative(composite),
                        "metadata": relative(metadata),
                        "regalia_mask": relative(regalia_mask),
                        "wand_mask": relative(wand_mask),
                        "wand_full_mask": relative(
                            ROOT / "metadata" / f"{page_id}_wand_full_mask.png"
                        ),
                        "wand_visible_mask": relative(
                            ROOT / "metadata" / f"{page_id}_wand_visible_mask.png"
                        ),
                        "hand_occlusion_mask": relative(
                            ROOT
                            / "metadata"
                            / f"{page_id}_hand_occlusion_mask.png"
                        ),
                        "hand_grip_mask": relative(
                            ROOT / "metadata" / f"{page_id}_hand_grip_mask.png"
                        ),
                    }
                )
                record["preflight_validation"] = validator.validate_preflight(
                    composite,
                    regalia_mask,
                    wand_mask,
                    metadata,
                )
                record["validation"] = record["preflight_validation"]
                if record["preflight_validation"]["accepted"]:
                    preflight_records.append(record)
            except Exception as exc:
                record["validation"] = {
                    "validator_version": validator.VALIDATOR_VERSION,
                    "stage": "preflight",
                    "accepted": False,
                    "score": 0.0,
                    "failures": [f"compositor failure: {exc}"],
                    "advisories": [],
                    "metrics": {},
                }
                print(f"[mask] rejected {page_id}: {exc}", flush=True)
            write_summary(summary)
        compositor.clear_model_caches()
        del compositor
        gc.collect()
        torch.cuda.empty_cache()
        make_contact_sheet(
            [record for record in records if "composite" in record["artifacts"]],
            ROOT / "review" / "composites.jpg",
        )

        regional = config["regional_refinement"]
        regalia_prompt = (
            "lqmoonregalia4, exact five-peak polished gold Moonstar crown, centered oval "
            "rose-amber moonstone, two amber side gems, matched gold leaf-drop earrings, "
            "narrow gold moonstone collar, clean auburn hair and natural background above "
            "the crown, integrated anime storybook linework"
        )
        regalia_negative = (
            "different crown, giant crown, extra crown peaks, flower crown, roses, hat, archway, "
            "cage, gold clothing, extra jewelry, changed face, changed hair, changed dress"
        )
        pipe = None
        for index, record in enumerate(preflight_records, start=1):
            page_id = record["id"]
            destination = ROOT / "regalia" / f"{page_id}.png"
            if destination.is_file():
                try:
                    with Image.open(destination) as existing:
                        reused = existing.size == (width, height)
                except Exception:
                    reused = False
                if reused:
                    print(
                        f"[regalia {index}/{len(preflight_records)}] reuse {page_id}",
                        flush=True,
                    )
                    record["artifacts"]["regalia"] = relative(destination)
                    continue
                destination.unlink(missing_ok=True)
            if pipe is None:
                pipe = make_inpaint(
                    REGALIA, "regalia", float(regional["regalia_weight"])
                )
                assert_prompts_fit(
                    pipe,
                    {
                        "regalia positive": regalia_prompt,
                        "regalia negative": regalia_negative,
                    },
                )
            print(f"[regalia {index}/{len(preflight_records)}] {page_id}", flush=True)
            composite = Image.open(ROOT / record["artifacts"]["composite"]).convert("RGB")
            mask = Image.open(ROOT / record["artifacts"]["regalia_mask"]).convert("L")
            metadata = json.loads(
                (ROOT / record["artifacts"]["metadata"]).read_text(encoding="utf-8")
            )
            regalia_strength = (
                float(regional["crown_cleanup_strength"])
                if metadata.get("existing_crown_removed")
                else float(regional["regalia_strength"])
            )
            with torch.inference_mode():
                result = pipe(
                    prompt=regalia_prompt,
                    negative_prompt=regalia_negative,
                    image=composite,
                    mask_image=mask,
                    width=width,
                    height=height,
                    strength=regalia_strength,
                    num_inference_steps=int(regional["steps"]),
                    guidance_scale=float(regional["guidance_scale"]),
                    padding_mask_crop=int(regional["padding_mask_crop"]),
                    generator=torch.Generator(device="cuda").manual_seed(record["seed"] + 500),
                ).images[0]
            result.save(destination, format="PNG", compress_level=6)
            record["artifacts"]["regalia"] = relative(destination)
            write_summary(summary)
            del composite, mask, result
            torch.cuda.empty_cache()
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()

        wand_prompt = (
            "lqmoonwand4, exact single slender gold Moonstar scepter, straight engraved shaft, "
            "open crescent finial around one rose-amber oval moonstone, five leaf crystals, "
            "two round side gems, pointed rose crystal end cap, one small natural hand firmly "
            "wrapped around the shaft with visible fingers, anime storybook linework"
        )
        wand_negative = (
            "different wand, giant staff, sword, blade, star wand, flower, vine, archway, cage, "
            "freestanding object, floating wand, open palm, extra wand, duplicate, bent shaft, "
            "extra fingers, fused fingers, changed body"
        )
        grip_prompt = (
            "small natural child hand tightly closed around the slender vertical gold wand "
            "shaft, thumb crossing in front, four curled fingers visibly wrapped behind the "
            "shaft, correct wrist and anatomy, clean anime storybook linework"
        )
        grip_negative = (
            "open palm, straight fingers, floating wand, hand beside wand, missing thumb, "
            "extra fingers, fused fingers, deformed hand, extra hand, changed sleeve"
        )
        pipe = None
        for index, record in enumerate(preflight_records, start=1):
            page_id = record["id"]
            destination = ROOT / "final" / f"{page_id}.png"
            if destination.is_file():
                try:
                    with Image.open(destination) as existing:
                        reused = existing.size == (width, height)
                except Exception:
                    reused = False
                if reused:
                    print(
                        f"[wand {index}/{len(preflight_records)}] reuse {page_id}",
                        flush=True,
                    )
                    record["artifacts"]["final"] = relative(destination)
                    record["validation"] = validator.validate_record(ROOT, record)
                    continue
                destination.unlink(missing_ok=True)
            if pipe is None:
                pipe = make_inpaint(WAND, "wand", float(regional["wand_weight"]))
                assert_prompts_fit(
                    pipe,
                    {
                        "wand positive": wand_prompt,
                        "wand negative": wand_negative,
                        "grip positive": grip_prompt,
                        "grip negative": grip_negative,
                    },
                )
            print(f"[wand {index}/{len(preflight_records)}] {page_id}", flush=True)
            regalia_image = Image.open(ROOT / record["artifacts"]["regalia"]).convert("RGB")
            mask = Image.open(ROOT / record["artifacts"]["wand_mask"]).convert("L")
            with torch.inference_mode():
                result = pipe(
                    prompt=wand_prompt,
                    negative_prompt=wand_negative,
                    image=regalia_image,
                    mask_image=mask,
                    width=width,
                    height=height,
                    strength=float(regional["wand_strength"]),
                    num_inference_steps=int(regional["steps"]),
                    guidance_scale=float(regional["guidance_scale"]),
                    padding_mask_crop=int(regional["padding_mask_crop"]),
                    generator=torch.Generator(device="cuda").manual_seed(record["seed"] + 900),
                ).images[0]
            grip_mask = Image.open(ROOT / record["artifacts"]["hand_grip_mask"]).convert("L")
            print(f"[grip {index}/{len(preflight_records)}] {page_id}", flush=True)
            pipe.set_adapters(
                ["wand"],
                adapter_weights=[float(regional["grip_wand_weight"])],
            )
            with torch.inference_mode():
                grip_result = pipe(
                    prompt=grip_prompt,
                    negative_prompt=grip_negative,
                    image=result,
                    mask_image=grip_mask,
                    width=width,
                    height=height,
                    strength=float(regional["grip_strength"]),
                    num_inference_steps=int(regional["steps"]),
                    guidance_scale=float(regional["guidance_scale"]),
                    padding_mask_crop=int(regional["grip_padding_mask_crop"]),
                    generator=torch.Generator(device="cuda").manual_seed(
                        record["seed"] + 1200
                    ),
                ).images[0]
            pipe.set_adapters(
                ["wand"],
                adapter_weights=[float(regional["wand_weight"])],
            )
            grip_result.save(destination, format="PNG", compress_level=6)
            record["artifacts"]["final"] = relative(destination)
            record["validation"] = validator.validate_record(ROOT, record)
            write_summary(summary)
            del regalia_image, mask, grip_mask, result, grip_result
            torch.cuda.empty_cache()
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()

        accepted = [record for record in records if record["validation"]["accepted"]]
        for record in records:
            record["selected"] = record in accepted[:target]
        validation_report = {
            "validator_version": validator.VALIDATOR_VERSION,
            "record_count": len(records),
            "accepted_count": len(accepted),
            "selected_count": min(target, len(accepted)),
            "records": [
                {"id": record["id"], "validation": record["validation"]}
                for record in records
            ],
        }
        (ROOT / "validation_report.json").write_text(
            json.dumps(validation_report, indent=2) + "\n",
            encoding="utf-8",
        )
        make_contact_sheet(
            [record for record in records if "final" in record["artifacts"]],
            ROOT / "review" / "final_candidates.jpg",
        )
        make_contact_sheet(
            records,
            ROOT / "review" / "validation_contact.jpg",
            validation_labels=True,
        )
        make_contact_sheet(
            accepted[:target],
            ROOT / "review" / "selected_finals.jpg",
            validation_labels=True,
        )
        make_grip_contact_sheet(
            [record for record in records if "final" in record["artifacts"]],
            ROOT / "review" / "grip_contacts.jpg",
        )

        summary["records"] = records
        summary["preflight_accepted_count"] = len(preflight_records)
        summary["accepted_count"] = len(accepted)
        summary["selected_count"] = min(target, len(accepted))
        summary["status"] = "complete" if len(accepted) >= target else "insufficient_accepted"
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_summary(summary)
        build_archive()
        print(
            f"Unattended run complete: {len(accepted)} accepted, "
            f"{min(target, len(accepted))} selected.",
            flush=True,
        )
        return 0
    finally:
        stop.set()
        monitor.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())

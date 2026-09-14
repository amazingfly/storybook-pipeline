#!/usr/bin/env python3
"""Resumable FLUX.2 Klein storybook comparison generation on a Colab T4."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


CONFIG_PATH = Path("/content/flux2_storybook_config.json")
MANIFEST_PATH = Path("/content/flux2_storybook_manifest.json")
TOKEN_PATH = Path("/content/hf_token")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_image(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == expected_size
    except (OSError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_report() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this test requires a Colab T4")
    properties = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    report = {
        "name": name,
        "total_vram_bytes": properties.total_memory,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    if "T4" not in name.upper():
        raise RuntimeError(f"Expected a T4 but Colab allocated {name}")
    return report


def save_archive(root: Path) -> Path:
    archive = Path("/content/flux2_storybook_results.tar.gz")
    temporary_base = Path("/content/flux2_storybook_results_partial")
    temporary_archive = Path(str(temporary_base) + ".tar.gz")
    temporary_archive.unlink(missing_ok=True)
    shutil.make_archive(str(temporary_base), "gztar", root_dir=root)
    temporary_archive.replace(archive)
    return archive


def main() -> int:
    config = read_json(CONFIG_PATH)
    manifest = read_json(MANIFEST_PATH)
    remote_root = Path(config["remote_root"])
    images_dir = remote_root / "images"
    metadata_dir = remote_root / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    progress_path = remote_root / "progress.json"
    failure_path = remote_root / "failure.json"
    failure_path.unlink(missing_ok=True)

    gpu = gpu_report()
    log(f"GPU verified: {gpu['name']} ({gpu['total_vram_bytes'] / 2**30:.2f} GiB)")
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("The Hugging Face token file is empty")

    jobs = manifest.get("jobs", [])
    if not jobs:
        raise ValueError("Manifest has no jobs")
    completed: list[dict[str, Any]] = []
    expected_size = (int(config["width"]), int(config["height"]))
    for job in jobs:
        output = images_dir / f"{job['id']}.png"
        metadata = metadata_dir / f"{job['id']}.json"
        if valid_image(output, expected_size) and metadata.is_file():
            completed.append(read_json(metadata))
    atomic_json(progress_path, {
        "version": config["version"], "updated_at": now(), "status": "loading",
        "gpu": gpu, "completed": completed, "completed_count": len(completed),
        "total_count": len(jobs),
    })
    if len(completed) == len(jobs):
        atomic_json(progress_path, {
            "version": config["version"], "updated_at": now(), "status": "complete",
            "gpu": gpu, "completed": completed, "completed_count": len(completed),
            "total_count": len(jobs),
        })
        archive = save_archive(remote_root)
        log(f"All {len(jobs)} jobs already complete; refreshed archive={archive}")
        return 0

    if config.get("memory_mode") != "text_encoder_4bit":
        raise ValueError(f"Unsupported memory_mode: {config.get('memory_mode')}")
    log("Loading the Qwen text encoder in 4-bit and the FLUX.2 transformer in float16")
    load_started = time.monotonic()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        subfolder="text_encoder",
        quantization_config=quantization,
        torch_dtype=torch.float16,
        token=token,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        config["base_model"],
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
        token=token,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    log(f"Base pipeline loaded in {time.monotonic() - load_started:.1f}s")

    log(f"Loading identity adapter from {config['lora_repo']}")
    pipe.load_lora_weights(
        config["lora_repo"],
        weight_name=config["lora_weight_name"],
        adapter_name=config["adapter_name"],
        token=token,
    )

    completed_ids = {record["id"] for record in completed}
    for index, job in enumerate(jobs, start=1):
        output = images_dir / f"{job['id']}.png"
        metadata = metadata_dir / f"{job['id']}.json"
        if job["id"] in completed_ids and valid_image(output, expected_size):
            log(f"[skip {index}/{len(jobs)}] {job['id']} already complete")
            continue

        scale = float(job["lora_scale"])
        pipe.set_adapters(config["adapter_name"], adapter_weights=scale)
        prompt = f"{config['trigger']}. {config['style_prefix']} {job['prompt']}"
        generator = torch.Generator(device="cuda").manual_seed(int(job["seed"]))
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        log(
            f"[generate {index}/{len(jobs)}] {job['id']} scene={job['scene']} "
            f"seed={job['seed']} lora={scale:.2f}"
        )
        result = pipe(
            prompt=prompt,
            width=int(config["width"]),
            height=int(config["height"]),
            num_inference_steps=int(config["num_inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            max_sequence_length=int(config["max_sequence_length"]),
            generator=generator,
        )
        elapsed = time.monotonic() - started
        temporary = output.with_suffix(".tmp.png")
        result.images[0].save(temporary, format="PNG")
        temporary.replace(output)
        del result
        record = {
            "id": job["id"], "scene": job["scene"], "seed": job["seed"],
            "lora_scale": scale, "original_prompt": job["prompt"],
            "effective_prompt": prompt, "image": f"images/{output.name}",
            "sha256": sha256(output), "width": expected_size[0],
            "height": expected_size[1], "elapsed_seconds": round(elapsed, 3),
            "memory_mode": config["memory_mode"],
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "completed_at": now(),
        }
        atomic_json(metadata, record)
        completed = [item for item in completed if item["id"] != job["id"]]
        completed.append(record)
        completed.sort(key=lambda item: item["id"])
        atomic_json(progress_path, {
            "version": config["version"], "updated_at": now(), "status": "running",
            "gpu": gpu, "completed": completed, "completed_count": len(completed),
            "total_count": len(jobs),
        })
        save_archive(remote_root)
        log(
            f"[generated {len(completed)}/{len(jobs)}] {job['id']} in {elapsed:.1f}s; "
            f"peak CUDA {record['peak_allocated_gib']:.2f} GiB"
        )
        gc.collect()
        torch.cuda.empty_cache()

    atomic_json(progress_path, {
        "version": config["version"], "updated_at": now(), "status": "complete",
        "gpu": gpu, "completed": completed, "completed_count": len(completed),
        "total_count": len(jobs),
    })
    archive = save_archive(remote_root)
    log(f"Comparison complete: {len(completed)}/{len(jobs)}; archive={archive}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        root = Path("/content/flux2_storybook_test")
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "failure.json", {
            "failed_at": now(), "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
        })
        try:
            save_archive(root)
        except Exception:
            pass
        log(f"FATAL {type(error).__name__}: {error}")
        raise
    if exit_code:
        raise SystemExit(exit_code)

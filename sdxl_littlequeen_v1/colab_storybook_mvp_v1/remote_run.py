#!/usr/bin/env python3
"""Generate a compiled JSON story with reusable base and regional LoRAs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


ROOT = Path("/content/lq_storybook_mvp_v1")
BUNDLE_PREFIX = Path("/content/storybook_mvp_v1_bundle.tar.gz.part_")
BASES_PREFIX = Path("/content/storybook_mvp_v1_bases.tar.gz.part_")
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
ARCHIVE = Path("/content/lq_storybook_mvp_v1_results.tar.gz")
CHECKPOINT_REMOTE_ROOT = Path("/content/storybook_checkpoints")
CHECKPOINT_RESTORE_PREFIX = Path("/content/storybook_checkpoint_restore.tar.gz.part_")
CHECKPOINT_MANIFEST_UPLOAD = Path("/content/storybook_checkpoint_manifest.json")
CHECKPOINT_INTERVAL = max(1, int(os.environ.get("STORYBOOK_CHECKPOINT_INTERVAL", "5")))
CHECKPOINT_VERSION = "storybook-durable-checkpoint-v1"
GENERATION_ONLY = os.environ.get("STORYBOOK_GENERATION_ONLY", "0") == "1"
LORA_DRIVE_FILE_ID = os.environ.get("STORYBOOK_LORA_DRIVE_FILE_ID", "")
LORA_ARCHIVE_SHA256 = os.environ.get("STORYBOOK_LORA_ARCHIVE_SHA256", "")
LORA_ARCHIVE = Path("/content/littlequeen_storybook_loras.tar")
CHECKPOINT_DIRECTORIES = (
    "base",
    "composite",
    "regalia",
    "wand_only",
    "grip_repaired",
    "metadata",
)
_CHECKPOINT = None
_RUN_LOCK = None

UPLOADS = {
    "little_queen_v2": (
        IDENTITY_PREFIX,
        IDENTITY,
        170_552_852,
        "3799128d4bfd4fbc7848d5b1de099a3e99cf3f718845c98efa7d4d48a7953994",
    ),
    "moon_dress_v1": (
        OUTFIT_PREFIX,
        OUTFIT,
        170_545_956,
        "985e82df407690831ef8d27a9c6c3b061689278c334e3694e21eaf8aa88c36d7",
    ),
    "moonstar_regalia": (
        REGALIA_PREFIX,
        REGALIA,
        85_424_812,
        "8d3757aa6cfc0b55b5af5c653075b961709c321077c2afabcc79407e4ca29014",
    ),
    "moonstar_wand": (
        WAND_PREFIX,
        WAND,
        85_424_748,
        "4714e064ae0bbc31c33b3398a21b98227d764fb8b0a0024932bb6f3359a34503",
    ),
}

BASE_LORA_FILES = {
    "little_queen_v2": (IDENTITY, "little_queen"),
    "moon_dress_v1": (OUTFIT, "moon_dress"),
}
GENERATION = {
    "steps": 28,
    "guidance_scale": 5.0,
}
REGIONAL = {
    "regalia_weight": 0.45,
    "wand_weight": 0.50,
    "grip_wand_weight": 0.20,
    "regalia_strength": 0.28,
    "crown_cleanup_strength": 0.45,
    "wand_strength": 0.30,
    "grip_strength": 0.68,
    "steps": 28,
    "guidance_scale": 3.5,
    "padding_mask_crop": 32,
    "grip_padding_mask_crop": 64,
}
STORYBOOK_STYLE = "polished anime storybook illustration"
COMMON_NEGATIVE_PROMPT = (
    "photograph, photorealistic, 3d render, text, watermark, logo, duplicate character, "
    "extra limbs, malformed hands, distorted face"
)
MOONSTAR_BASE_NEGATIVE = (
    "crown, tiara, hat, hair ornament, ornate wand, staff, sword, weapon, floating rod, "
    "angled rod, open palm, jewelry, cropped head, cropped hands, cropped feet, hidden hands"
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


def checkpoint_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "storybook"


class DurableCheckpoint:
    def __init__(
        self,
        config: dict,
        *,
        run_root: Path = ROOT,
        drive_root: Path = CHECKPOINT_REMOTE_ROOT,
        interval: int = CHECKPOINT_INTERVAL,
    ) -> None:
        self.run_root = run_root
        self.interval = max(1, interval)
        self.events_since_flush = 0
        self.compiled_path = run_root / "compiled_story.json"
        self.compiled_sha256 = sha256(self.compiled_path)
        checkpoint_name = (
            f"{checkpoint_slug(config['title'])}_{self.compiled_sha256[:16]}"
        )
        self.root = drive_root / checkpoint_name
        self.manifest_path = self.root / "checkpoint_manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if CHECKPOINT_MANIFEST_UPLOAD.is_file():
            incoming = json.loads(
                CHECKPOINT_MANIFEST_UPLOAD.read_text(encoding="utf-8")
            )
            expected = {
                "version": CHECKPOINT_VERSION,
                "compiled_story_sha256": self.compiled_sha256,
                "candidate_count": len(config["candidates"]),
            }
            for key, value in expected.items():
                if incoming.get(key) != value:
                    raise RuntimeError(
                        f"uploaded checkpoint manifest {key} mismatch: "
                        f"expected {value!r}, found {incoming.get(key)!r}"
                    )
            existing = None
            if self.manifest_path.is_file():
                try:
                    existing = json.loads(
                        self.manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    existing = None
            if (
                GENERATION_ONLY
                or existing is None
                or len(incoming.get("batches", []))
                > len(existing.get("batches", []))
            ):
                shutil.copy2(CHECKPOINT_MANIFEST_UPLOAD, self.manifest_path)
                print(
                    "[checkpoint] imported authoritative manifest-only "
                    f"completion index ({len(incoming.get('files', {}))} files)",
                    flush=True,
                )
        if self.manifest_path.is_file():
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            self._validate_manifest(config)
        else:
            self.manifest = {
                "version": CHECKPOINT_VERSION,
                "title": config["title"],
                "compiled_story_sha256": self.compiled_sha256,
                "candidate_count": len(config["candidates"]),
                "checkpoint_interval": self.interval,
                "batches": [],
                "files": {},
            }
            self._write_manifest()
        self.known: dict[str, tuple[int, int]] = {}

    def _validate_manifest(self, config: dict) -> None:
        expected = {
            "version": CHECKPOINT_VERSION,
            "compiled_story_sha256": self.compiled_sha256,
            "candidate_count": len(config["candidates"]),
        }
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise RuntimeError(
                    f"checkpoint manifest {key} mismatch: "
                    f"expected {value!r}, found {self.manifest.get(key)!r}"
                )

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(self.manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def _tracked_files(self) -> dict[str, Path]:
        tracked: dict[str, Path] = {}
        for directory in CHECKPOINT_DIRECTORIES:
            root = self.run_root / directory
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_file() and not path.name.endswith(".part"):
                    tracked[str(path.relative_to(self.run_root))] = path
        validation_report = self.run_root / "validation_report.json"
        if validation_report.is_file():
            tracked["validation_report.json"] = validation_report
        return tracked

    @staticmethod
    def _state(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def restore(self) -> int:
        if GENERATION_ONLY and CHECKPOINT_MANIFEST_UPLOAD.is_file():
            print(
                f"[checkpoint] indexed {len(self.manifest['files'])} completed files "
                "without uploading their image data",
                flush=True,
            )
            return len(self.manifest["batches"])
        restored = 0
        for batch in self.manifest["batches"]:
            archive_path = self.root / batch["archive"]
            if not archive_path.is_file():
                raise RuntimeError(f"checkpoint batch is missing: {archive_path}")
            if archive_path.stat().st_size != batch["bytes"]:
                raise RuntimeError(f"checkpoint batch size mismatch: {archive_path}")
            if sha256(archive_path) != batch["sha256"]:
                raise RuntimeError(f"checkpoint batch checksum mismatch: {archive_path}")
            with tarfile.open(archive_path, "r") as archive:
                archive.extractall(self.run_root, filter="data")
            restored += 1
        tracked = self._tracked_files()
        self.known = {
            relative: self._state(path)
            for relative, path in tracked.items()
            if relative in self.manifest["files"]
        }
        if restored:
            print(
                f"[checkpoint] restored {restored} batches with "
                f"{len(self.known)} files from {self.root}",
                flush=True,
            )
        else:
            print(f"[checkpoint] new durable run at {self.root}", flush=True)
        return restored

    def tick(self) -> None:
        self.events_since_flush += 1
        if self.events_since_flush >= self.interval:
            self.flush()

    def mark_complete(self) -> None:
        self.manifest["status"] = "complete"
        self.manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest()
        print("[checkpoint] durable run marked complete", flush=True)

    def flush(self, *, force: bool = False) -> None:
        if not force and self.events_since_flush < self.interval:
            return
        tracked = self._tracked_files()
        changed = {
            relative: path
            for relative, path in tracked.items()
            if self.known.get(relative) != self._state(path)
        }
        self.events_since_flush = 0
        if not changed:
            return
        sequence = max(
            (
                int(batch["sequence"])
                for batch in self.manifest["batches"]
            ),
            default=0,
        ) + 1
        archive_name = f"batch_{sequence:06d}.tar"
        local_archive = self.run_root.parent / f".{archive_name}.part"
        with tarfile.open(local_archive, "w") as archive:
            for relative, path in sorted(changed.items()):
                archive.add(path, arcname=relative)
        digest = sha256(local_archive)
        final_archive = self.root / archive_name
        if final_archive.exists():
            local_archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"refusing to overwrite existing checkpoint batch: {final_archive}"
            )
        drive_temporary = final_archive.with_suffix(".tar.part")
        last_error = None
        for attempt in range(1, 4):
            try:
                shutil.copyfile(local_archive, drive_temporary)
                drive_temporary.replace(final_archive)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                print(
                    f"[checkpoint] Drive write attempt {attempt}/3 failed: {exc}",
                    flush=True,
                )
                time.sleep(5 * attempt)
        local_archive.unlink(missing_ok=True)
        if last_error is not None:
            raise RuntimeError("durable checkpoint write failed") from last_error
        file_records = {}
        for relative, path in changed.items():
            state = self._state(path)
            self.known[relative] = state
            file_records[relative] = {
                "bytes": state[0],
                "batch": sequence,
            }
        self.manifest["files"].update(file_records)
        self.manifest["batches"].append(
            {
                "sequence": sequence,
                "archive": archive_name,
                "bytes": final_archive.stat().st_size,
                "sha256": digest,
                "file_count": len(changed),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.manifest["last_checkpoint_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest()
        print(
            f"[checkpoint] batch {sequence:06d}: {len(changed)} files, "
            f"{final_archive.stat().st_size} bytes",
            flush=True,
        )


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
            "gdown==6.1.0",
        ]
    )


def ensure_loras(required_names: set[str]) -> None:
    expanded_names = set(required_names)
    if "moonstar_accessories_v1" in expanded_names:
        expanded_names.update({"moonstar_regalia", "moonstar_wand"})
        expanded_names.remove("moonstar_accessories_v1")
    specifications = {}
    for name in sorted(expanded_names):
        upload = UPLOADS.get(name)
        if upload is None:
            raise RuntimeError(f"no LoRA mapping for {name}")
        _, destination, expected_bytes, expected_sha = upload
        specifications[name] = (destination, expected_bytes, expected_sha)
    if all(
        destination.is_file()
        and destination.stat().st_size == expected_bytes
        and sha256(destination) == expected_sha
        for destination, expected_bytes, expected_sha in specifications.values()
    ):
        print(
            f"[loras] reused {len(specifications)} verified runtime files",
            flush=True,
        )
        return
    if not LORA_DRIVE_FILE_ID or not LORA_ARCHIVE_SHA256:
        raise RuntimeError("Google Drive LoRA archive configuration is missing")

    import gdown

    temporary = LORA_ARCHIVE.with_suffix(".tar.part")
    for attempt in range(1, 4):
        try:
            print(
                f"[loras] downloading Google Drive archive "
                f"(attempt {attempt}/3, file {LORA_DRIVE_FILE_ID})",
                flush=True,
            )
            result = gdown.download(
                id=LORA_DRIVE_FILE_ID,
                output=str(temporary),
                quiet=False,
                resume=attempt > 1,
            )
            if result is None or not temporary.is_file():
                raise RuntimeError("gdown did not produce an archive")
            if sha256(temporary) != LORA_ARCHIVE_SHA256:
                raise RuntimeError("Google Drive LoRA archive checksum mismatch")
            temporary.replace(LORA_ARCHIVE)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * attempt)
    print(
        f"[loras] verified archive sha256 {LORA_ARCHIVE_SHA256}",
        flush=True,
    )

    expected_members = {
        upload[1].name
        for upload in UPLOADS.values()
    }
    with tarfile.open(LORA_ARCHIVE, "r") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        member_names = {member.name for member in members}
        if member_names != expected_members:
            raise RuntimeError(
                "Google Drive LoRA archive contents mismatch: "
                f"expected {sorted(expected_members)}, found {sorted(member_names)}"
            )
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        archive.extractall(MODEL.parent, members=members, filter="data")

    for name, (destination, expected_bytes, expected_sha) in specifications.items():
        if (
            not destination.is_file()
            or destination.stat().st_size != expected_bytes
            or sha256(destination) != expected_sha
        ):
            raise RuntimeError(f"extracted LoRA verification failed for {name}")
    LORA_ARCHIVE.unlink(missing_ok=True)
    print(
        f"[loras] extracted and verified {len(specifications)} LoRAs",
        flush=True,
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


def make_txt2img(config: dict, candidate: dict):
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
    for selection in config["required_loras"]:
        if selection["mode"] != "base":
            continue
        path, expected_adapter = BASE_LORA_FILES[selection["name"]]
        if selection["adapter"] != expected_adapter:
            raise RuntimeError(f"unexpected adapter name for {selection['name']}")
        pipe.load_lora_weights(str(path), adapter_name=selection["adapter"])
    names = [item["adapter"] for item in candidate["base_loras"]]
    weights = [float(item["weight"]) for item in candidate["base_loras"]]
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
    else:
        pipe.disable_lora()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def adapter_signature(candidate: dict) -> tuple[tuple[str, float], ...]:
    return tuple(
        (item["adapter"], float(item["weight"]))
        for item in candidate["base_loras"]
    )


def has_moonstar(candidate: dict) -> bool:
    return any(
        item["mode"] == "moonstar_regional_set"
        for item in candidate["extra_loras"]
    )


def moonstar_weight(candidate: dict) -> float:
    for item in candidate["extra_loras"]:
        if item["mode"] == "moonstar_regional_set":
            return float(item["weight"])
    return 0.0


def build_base_prompt(candidate: dict) -> str:
    triggers = [item["trigger"] for item in candidate["base_loras"]]
    staging = []
    if has_moonstar(candidate):
        staging.extend(
            [
                "Little Queen centered foreground, full body visible, bare head",
                "one hand gripping a vertical guide rod",
            ]
        )
    parts = [
        *triggers,
        candidate["prompt"],
        *staging,
        STORYBOOK_STYLE,
    ]
    return ", ".join(part for part in parts if part)


def build_negative_prompt(candidate: dict) -> str:
    parts = [COMMON_NEGATIVE_PROMPT]
    if has_moonstar(candidate):
        parts.append(MOONSTAR_BASE_NEGATIVE)
    return ", ".join(parts)


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


def make_variant_contact_sheet(
    records: list[dict],
    destination: Path,
    *,
    grip_crops: bool,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    variants = (("wand_only", "WAND ONLY"), ("grip_repaired", "GRIP REPAIR"))
    columns = 4
    cell = (280, 310 if grip_crops else 410)
    rows = max(1, (len(records) * len(variants) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    slot = 0
    for record in records:
        artifacts = record.get("artifacts", {})
        hand_box = None
        if grip_crops and "metadata" in artifacts:
            metadata = json.loads(
                (ROOT / artifacts["metadata"]).read_text(encoding="utf-8")
            )
            hand_box = metadata["placements"]["staff"].get("detected_hand_box")
        for key, variant_label in variants:
            if key not in artifacts:
                continue
            image = Image.open(ROOT / artifacts[key]).convert("RGB")
            if grip_crops and hand_box:
                left, top, right, bottom = (float(value) for value in hand_box)
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                side = min(
                    max(max(right - left, bottom - top) * 4.2, 160.0),
                    min(image.size) * 0.45,
                )
                image = image.crop(
                    (
                        max(0, round(center_x - side / 2)),
                        max(0, round(center_y - side / 2)),
                        min(image.width, round(center_x + side / 2)),
                        min(image.height, round(center_y + side / 2)),
                    )
                )
            thumb = ImageOps.contain(image, (cell[0] - 12, cell[1] - 48))
            x = (slot % columns) * cell[0]
            y = (slot // columns) * cell[1]
            sheet.paste(thumb, (x + (cell[0] - thumb.width) // 2, y))
            validation = record.get("variant_validation", {}).get(key, {})
            verdict = "PASS" if validation.get("accepted") else "REJECT"
            color = "#7ee787" if verdict == "PASS" else "#ff7b72"
            draw.text(
                (x + 6, y + cell[1] - 38),
                f"{record['id']} {variant_label}",
                fill="white",
            )
            draw.text(
                (x + 6, y + cell[1] - 20),
                f"GEOMETRY {verdict}",
                fill=color,
            )
            slot += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def validate_variant(validator, record: dict, artifact_key: str) -> dict:
    variant_record = {
        **record,
        "artifacts": {
            **record["artifacts"],
            "final": record["artifacts"][artifact_key],
        },
    }
    return validator.validate_record(ROOT, variant_record)


def write_summary(summary: dict) -> None:
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if _CHECKPOINT is not None:
        _CHECKPOINT.tick()


def build_archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for directory in (
            "base",
            "composite",
            "regalia",
            "wand_only",
            "grip_repaired",
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
        archive.add(ROOT / "compiled_story.json", arcname="compiled_story.json")
    print(f"Wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)", flush=True)


def main() -> int:
    global _CHECKPOINT, _RUN_LOCK
    _RUN_LOCK = Path("/content/storybook_mvp_v1.lock").open("w")
    try:
        fcntl.flock(_RUN_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            "another storybook generation process is already active"
        ) from exc
    print("[runtime] acquired exclusive storybook generation lock", flush=True)
    install_dependencies()
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    run(["nvidia-smi"])
    prior_compiled = ROOT / "compiled_story.json"
    prior_compiled_sha256 = (
        sha256(prior_compiled) if prior_compiled.is_file() else None
    )
    same_runtime_resume = (ROOT / "summary.json").is_file() and MODEL.is_file()
    if not same_runtime_resume:
        shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    assemble(BUNDLE_PREFIX, ROOT / "bundle.tar.gz")
    with tarfile.open(ROOT / "bundle.tar.gz", "r:gz") as archive:
        archive.extractall(ROOT)
    config = json.loads((ROOT / "compiled_story.json").read_text(encoding="utf-8"))
    current_compiled_sha256 = sha256(ROOT / "compiled_story.json")
    if (
        same_runtime_resume
        and prior_compiled_sha256 != current_compiled_sha256
    ):
        print(
            "[checkpoint] preserved runtime belongs to a different compiled story; "
            "discarding its local artifacts",
            flush=True,
        )
        for directory in (*CHECKPOINT_DIRECTORIES, "review", "logs", "final"):
            shutil.rmtree(ROOT / directory, ignore_errors=True)
        (ROOT / "summary.json").unlink(missing_ok=True)
        (ROOT / "validation_report.json").unlink(missing_ok=True)
        same_runtime_resume = False
    if any(
        CHECKPOINT_RESTORE_PREFIX.parent.glob(
            CHECKPOINT_RESTORE_PREFIX.name + "*"
        )
    ):
        restore_archive = Path("/content/storybook_checkpoint_restore.tar.gz")
        assemble(CHECKPOINT_RESTORE_PREFIX, restore_archive)
        CHECKPOINT_REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(restore_archive, "r:gz") as archive:
            archive.extractall(CHECKPOINT_REMOTE_ROOT, filter="data")
        print(
            f"[checkpoint] imported local mirror into {CHECKPOINT_REMOTE_ROOT}",
            flush=True,
        )
    _CHECKPOINT = DurableCheckpoint(config)
    restored_batches = _CHECKPOINT.restore()
    resume = same_runtime_resume or bool(restored_batches)
    required_names = {item["name"] for item in config["required_loras"]}
    ensure_loras(required_names)
    if any(BASES_PREFIX.parent.glob(BASES_PREFIX.name + "*")):
        assemble(BASES_PREFIX, ROOT / "reuse_bases.tar.gz")
        with tarfile.open(ROOT / "reuse_bases.tar.gz", "r:gz") as archive:
            archive.extractall(ROOT)
    ensure_model()

    width, height = (int(value) for value in config["settings"]["resolution"])
    candidates = config["candidates"]
    scene_count = len(config["scenes"])

    for directory in (
        "base",
        "composite",
        "regalia",
        "wand_only",
        "grip_repaired",
        "final",
        "metadata",
        "review",
        "logs",
    ):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    summary = {
        "name": "littlequeen_storybook_mvp_v1",
        "title": config["title"],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scene_count": scene_count,
        "candidate_count": len(candidates),
        "resolution": [width, height],
        "models": {
            "base": MODEL_ID,
            "loras": {
                item["name"]: (
                    item["sha256"]
                    if item["mode"] == "base"
                    else {file["role"]: file["sha256"] for file in item["files"]}
                )
                for item in config["required_loras"]
            },
        },
        "durable_checkpoint": {
            "version": CHECKPOINT_VERSION,
            "path": str(_CHECKPOINT.root),
            "interval": _CHECKPOINT.interval,
            "restored_batches": restored_batches,
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

    try:
        pipe = None
        pipe_signature = None
        generation_order = sorted(candidates, key=adapter_signature)
        prompts_checked = False
        for index, candidate in enumerate(generation_order, start=1):
            page_id = candidate["id"]
            seed = int(candidate["seed"])
            prompt = build_base_prompt(candidate)
            negative = build_negative_prompt(candidate)
            destination = ROOT / "base" / f"{page_id}.png"
            checkpoint_relative = str(destination.relative_to(ROOT))
            reused = False
            if destination.is_file():
                try:
                    with Image.open(destination) as existing:
                        reused = existing.size == (width, height)
                except Exception:
                    destination.unlink(missing_ok=True)
            manifest_only_reuse = (
                GENERATION_ONLY
                and checkpoint_relative in _CHECKPOINT.manifest["files"]
            )
            if manifest_only_reuse:
                reused = True
                print(
                    f"[generate {index}/{len(generation_order)}] "
                    f"manifest skip {page_id}",
                    flush=True,
                )
                duration = 0.0
            elif reused:
                print(
                    f"[generate {index}/{len(generation_order)}] reuse {page_id}",
                    flush=True,
                )
                duration = 0.0
            else:
                signature = adapter_signature(candidate)
                if pipe is None or signature != pipe_signature:
                    if pipe is not None:
                        del pipe
                        gc.collect()
                        torch.cuda.empty_cache()
                    pipe = make_txt2img(config, candidate)
                    pipe_signature = signature
                if not prompts_checked:
                    prompt_set = {}
                    for item in candidates:
                        prompt_set[f"base positive {item['id']}"] = build_base_prompt(
                            item
                        )
                        prompt_set[f"base negative {item['id']}"] = (
                            build_negative_prompt(item)
                        )
                    assert_prompts_fit(pipe, prompt_set)
                    prompts_checked = True
                print(
                    f"[generate {index}/{len(generation_order)}] {page_id}",
                    flush=True,
                )
                started = time.monotonic()
                with torch.inference_mode():
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative,
                        width=width,
                        height=height,
                        num_inference_steps=int(GENERATION["steps"]),
                        guidance_scale=float(GENERATION["guidance_scale"]),
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                    ).images[0]
                image.save(destination, format="PNG", compress_level=6)
                duration = round(time.monotonic() - started, 2)
                print(
                    f"[generated {index}/{len(generation_order)}] {page_id} "
                    f"in {duration:.2f}s",
                    flush=True,
                )
                del image
            records.append(
                {
                    "id": page_id,
                    "seed": seed,
                    "scene_number": int(candidate["scene"]),
                    "candidate_index": int(candidate["candidate_index"]),
                    "prompt": candidate["prompt"],
                    "script": candidate["script"],
                    "base_loras": candidate["base_loras"],
                    "extra_loras": candidate["extra_loras"],
                    "uses_moonstar_accessories": has_moonstar(candidate),
                    "duration_seconds": duration,
                    "reused_base": reused,
                    "artifacts": {"base": relative(destination)},
                }
            )
            write_summary(summary)
            torch.cuda.empty_cache()
        if GENERATION_ONLY:
            if pipe is not None:
                del pipe
            gc.collect()
            torch.cuda.empty_cache()
            summary["records"] = records
            summary["status"] = "base_generation_complete"
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_summary(summary)
            _CHECKPOINT.flush(force=True)
            _CHECKPOINT.mark_complete()
            print(
                f"Base generation complete: indexed {len(records)} candidates; "
                "validation and selection will run locally.",
                flush=True,
            )
            return 0
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
            if not record["uses_moonstar_accessories"]:
                record["artifacts"]["final"] = record["artifacts"]["base"]
                record["validation"] = {
                    "validator_version": validator.VALIDATOR_VERSION,
                    "stage": "base_generated",
                    "accepted": True,
                    "score": 100.0,
                    "failures": [],
                    "advisories": ["awaiting local visual ranking"],
                    "metrics": {},
                }
                write_summary(summary)
                continue
            print(f"[mask {index}/{len(records)}] {page_id}", flush=True)
            composite = ROOT / "composite" / f"{page_id}.png"
            metadata = ROOT / "metadata" / f"{page_id}.json"
            regalia_mask = ROOT / "metadata" / f"{page_id}_regalia_mask.png"
            wand_mask = ROOT / "metadata" / f"{page_id}_wand_mask.png"
            accessory_artifacts = {
                "composite": composite,
                "metadata": metadata,
                "regalia_mask": regalia_mask,
                "wand_mask": wand_mask,
                "wand_full_mask": (
                    ROOT / "metadata" / f"{page_id}_wand_full_mask.png"
                ),
                "wand_visible_mask": (
                    ROOT / "metadata" / f"{page_id}_wand_visible_mask.png"
                ),
                "hand_occlusion_mask": (
                    ROOT / "metadata" / f"{page_id}_hand_occlusion_mask.png"
                ),
                "hand_grip_mask": (
                    ROOT / "metadata" / f"{page_id}_hand_grip_mask.png"
                ),
            }
            try:
                reusable = metadata.is_file()
                for key, path in accessory_artifacts.items():
                    if key == "metadata":
                        continue
                    if not path.is_file():
                        reusable = False
                        break
                    with Image.open(path) as existing:
                        if existing.size != (width, height):
                            reusable = False
                            break
                if reusable:
                    json.loads(metadata.read_text(encoding="utf-8"))
                    print(
                        f"[mask {index}/{len(records)}] reuse {page_id}",
                        flush=True,
                    )
                else:
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
                        blend_mask=ROOT
                        / "metadata"
                        / f"{page_id}_blend_mask.png",
                        cleanup_mask=ROOT
                        / "metadata"
                        / f"{page_id}_cleanup_mask.png",
                        regalia_mask=regalia_mask,
                        hand_prop_mask=wand_mask,
                        hand_prop_full_mask=accessory_artifacts["wand_full_mask"],
                        hand_prop_visible_mask=accessory_artifacts[
                            "wand_visible_mask"
                        ],
                        hand_occlusion_mask=accessory_artifacts[
                            "hand_occlusion_mask"
                        ],
                        hand_grip_mask=accessory_artifacts["hand_grip_mask"],
                        no_person_occlusion=False,
                        quiet=True,
                    )
                    compositor.main(arguments)
                record["artifacts"].update(
                    {
                        key: relative(path)
                        for key, path in accessory_artifacts.items()
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

        regional = REGIONAL
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
                    REGALIA,
                    "regalia",
                    float(regional["regalia_weight"]) * moonstar_weight(record),
                )
                assert_prompts_fit(
                    pipe,
                    {
                        "regalia positive": regalia_prompt,
                        "regalia negative": regalia_negative,
                    },
                )
            print(f"[regalia {index}/{len(preflight_records)}] {page_id}", flush=True)
            pipe.set_adapters(
                ["regalia"],
                adapter_weights=[
                    float(regional["regalia_weight"]) * moonstar_weight(record)
                ],
            )
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
            wand_destination = ROOT / "wand_only" / f"{page_id}.png"
            grip_destination = ROOT / "grip_repaired" / f"{page_id}.png"
            wand_reused = False
            grip_reused = False
            if wand_destination.is_file():
                try:
                    with Image.open(wand_destination) as existing:
                        wand_reused = existing.size == (width, height)
                except Exception:
                    wand_reused = False
                if not wand_reused:
                    wand_destination.unlink(missing_ok=True)
            if grip_destination.is_file():
                try:
                    with Image.open(grip_destination) as existing:
                        grip_reused = existing.size == (width, height)
                except Exception:
                    grip_reused = False
                if not grip_reused:
                    grip_destination.unlink(missing_ok=True)
            if pipe is None:
                pipe = make_inpaint(
                    WAND,
                    "wand",
                    float(regional["wand_weight"]) * moonstar_weight(record),
                )
                assert_prompts_fit(
                    pipe,
                    {
                        "wand positive": wand_prompt,
                        "wand negative": wand_negative,
                        "grip positive": grip_prompt,
                        "grip negative": grip_negative,
                    },
                )
            if wand_reused:
                print(
                    f"[wand {index}/{len(preflight_records)}] reuse {page_id}",
                    flush=True,
                )
                result = Image.open(wand_destination).convert("RGB")
            else:
                print(f"[wand {index}/{len(preflight_records)}] {page_id}", flush=True)
                pipe.set_adapters(
                    ["wand"],
                    adapter_weights=[
                        float(regional["wand_weight"]) * moonstar_weight(record)
                    ],
                )
                regalia_image = Image.open(
                    ROOT / record["artifacts"]["regalia"]
                ).convert("RGB")
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
                        generator=torch.Generator(device="cuda").manual_seed(
                            record["seed"] + 900
                        ),
                    ).images[0]
                result.save(wand_destination, format="PNG", compress_level=6)
                del regalia_image, mask
            record["artifacts"]["wand_only"] = relative(wand_destination)
            if grip_reused:
                print(
                    f"[grip {index}/{len(preflight_records)}] reuse {page_id}",
                    flush=True,
                )
            else:
                grip_mask = Image.open(
                    ROOT / record["artifacts"]["hand_grip_mask"]
                ).convert("L")
                print(f"[grip {index}/{len(preflight_records)}] {page_id}", flush=True)
                pipe.set_adapters(
                    ["wand"],
                    adapter_weights=[
                        float(regional["grip_wand_weight"])
                        * moonstar_weight(record)
                    ],
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
                    adapter_weights=[
                        float(regional["wand_weight"]) * moonstar_weight(record)
                    ],
                )
                grip_result.save(grip_destination, format="PNG", compress_level=6)
                del grip_mask, grip_result
            record["artifacts"]["grip_repaired"] = relative(grip_destination)
            record["variant_validation"] = {
                "wand_only": validate_variant(validator, record, "wand_only"),
                "grip_repaired": validate_variant(
                    validator, record, "grip_repaired"
                ),
            }
            provisional_variant = (
                "grip_repaired"
                if record["variant_validation"]["grip_repaired"]["accepted"]
                else "wand_only"
            )
            record["provisional_variant"] = provisional_variant
            record["artifacts"]["final"] = record["artifacts"][provisional_variant]
            record["validation"] = record["variant_validation"][provisional_variant]
            write_summary(summary)
            del result
            torch.cuda.empty_cache()
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()

        review_pool = [
            record
            for record in records
            if (
                not record["uses_moonstar_accessories"]
                and record.get("validation", {}).get("accepted")
            )
            or any(
                validation["accepted"]
                for validation in record.get("variant_validation", {}).values()
            )
        ]
        for record in records:
            record["automatic_shortlist"] = record in review_pool
            record["selected"] = False
        validation_report = {
            "validator_version": validator.VALIDATOR_VERSION,
            "record_count": len(records),
            "geometry_shortlist_count": len(review_pool),
            "scene_count": scene_count,
            "scenes_with_candidates": sorted(
                {record["scene_number"] for record in review_pool}
            ),
            "selected_count": 0,
            "requires_visual_review": True,
            "records": [
                {
                    "id": record["id"],
                    "provisional_variant": record.get("provisional_variant"),
                    "variant_validation": record.get("variant_validation", {}),
                }
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
            review_pool,
            ROOT / "review" / "automatic_geometry_shortlist.jpg",
            validation_labels=True,
        )
        make_grip_contact_sheet(
            [record for record in records if "final" in record["artifacts"]],
            ROOT / "review" / "grip_contacts.jpg",
        )
        make_variant_contact_sheet(
            review_pool,
            ROOT / "review" / "variant_pages.jpg",
            grip_crops=False,
        )
        make_variant_contact_sheet(
            review_pool,
            ROOT / "review" / "variant_grips.jpg",
            grip_crops=True,
        )

        summary["records"] = records
        summary["preflight_accepted_count"] = len(preflight_records)
        summary["geometry_shortlist_count"] = len(review_pool)
        summary["selected_count"] = 0
        summary["requires_visual_review"] = True
        covered_scenes = {record["scene_number"] for record in review_pool}
        summary["scenes_with_candidates"] = sorted(covered_scenes)
        summary["status"] = (
            "awaiting_local_selection"
            if len(covered_scenes) == scene_count
            else "missing_scene_candidates"
        )
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_summary(summary)
        _CHECKPOINT.flush(force=True)
        build_archive()
        _CHECKPOINT.mark_complete()
        print(
            f"Story generation complete: {len(review_pool)} geometry candidates "
            f"across {len(covered_scenes)}/{scene_count} scenes; awaiting local ranking.",
            flush=True,
        )
        return 0
    finally:
        if _CHECKPOINT is not None:
            try:
                _CHECKPOINT.flush(force=True)
            except Exception as exc:
                print(f"[checkpoint] final flush failed: {exc}", flush=True)
        stop.set()
        monitor.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())

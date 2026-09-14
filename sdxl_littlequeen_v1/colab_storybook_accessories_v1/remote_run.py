#!/usr/bin/env python3
"""Run old-crown cleanup, canonical placement, and attachment blending on Colab."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/lq_storybook_accessories_v1")
BUNDLE_PREFIX = Path("/content/storybook_accessories_bundle.tar.gz.part_")
IDENTITY_PREFIX = Path("/content/lqxl_sdxl_v2.safetensors.part_")
IDENTITY = ROOT / "models" / "lqxl_sdxl_v2.safetensors"
MODEL = ROOT / "models" / "sd_xl_base_1.0.safetensors"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
ARCHIVE = Path("/content/lq_storybook_accessories_v1_results.tar.gz")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(prefix: Path, destination: Path, expected_bytes: int | None = None, expected_sha: str | None = None) -> None:
    parts = sorted(prefix.parent.glob(prefix.name + "*"))
    if not parts:
        raise FileNotFoundError(f"no chunks match {prefix}*")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if expected_bytes is not None and destination.stat().st_size != expected_bytes:
        raise RuntimeError(f"size mismatch for {destination}")
    if expected_sha is not None and sha256(destination) != expected_sha:
        raise RuntimeError(f"checksum mismatch for {destination}")


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


def make_pipeline():
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
    pipe.load_lora_weights(str(IDENTITY), adapter_name="identity")
    pipe.set_adapters(["identity"], adapter_weights=[0.45])
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def main() -> int:
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
        ]
    )
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    run(["nvidia-smi"])
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    bundle = ROOT / "bundle.tar.gz"
    assemble(BUNDLE_PREFIX, bundle)
    assemble(
        IDENTITY_PREFIX,
        IDENTITY,
        170_552_852,
        "3799128d4bfd4fbc7848d5b1de099a3e99cf3f718845c98efa7d4d48a7953994",
    )
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(ROOT)
    ensure_model()

    storybook = ROOT / "storybook"
    pages = json.loads((storybook / "test_pages.json").read_text(encoding="utf-8"))["pages"]
    clean_dir = ROOT / "clean"
    composite_dir = ROOT / "composite"
    final_dir = ROOT / "final"
    metadata_dir = ROOT / "metadata"
    for path in (clean_dir, composite_dir, final_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    cleanup_prompt = (
        "same Little Queen anime storybook page, bare uncovered auburn hair, no headwear, "
        "continuous natural hair and background where the crown was removed, preserve the "
        "same face, expression, pose, body, dress, lighting, composition, and scene"
    )
    cleanup_negative = (
        "crown, tiara, diadem, headband, hat, flowers on head, jewelry on head, extra person, "
        "changed face, changed pose, changed outfit, text, watermark"
    )
    pipe = make_pipeline()
    cleanup_records = []
    for index, record in enumerate(pages, start=1):
        page_id = record["id"]
        image = Image.open(storybook / "inputs" / f"{page_id}.png").convert("RGB")
        mask = Image.open(storybook / "cleanup_masks" / f"{page_id}.png").convert("L")
        print(f"[cleanup {index}/{len(pages)}] {page_id}", flush=True)
        with torch.inference_mode():
            cleaned = pipe(
                prompt=cleanup_prompt,
                negative_prompt=cleanup_negative,
                image=image,
                mask_image=mask,
                width=image.width,
                height=image.height,
                strength=0.99,
                num_inference_steps=30,
                guidance_scale=5.0,
                padding_mask_crop=64,
                generator=torch.Generator(device="cuda").manual_seed(int(record["seed"])),
            ).images[0]
        destination = clean_dir / f"{page_id}.png"
        cleaned.save(destination, format="PNG", compress_level=6)
        cleanup_records.append({"id": page_id, "output": str(destination)})
        del image, mask, cleaned
        torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()

    compositor = storybook / "apply_storybook_accessories.py"
    for index, record in enumerate(pages, start=1):
        page_id = record["id"]
        print(f"[composite {index}/{len(pages)}] {page_id}", flush=True)
        run(
            [
                sys.executable,
                str(compositor),
                str(clean_dir / f"{page_id}.png"),
                str(composite_dir / f"{page_id}.png"),
                "--set",
                str(storybook / "accessory_set.json"),
                "--staff",
                "--metadata",
                str(metadata_dir / f"{page_id}.json"),
                "--blend-mask",
                str(metadata_dir / f"{page_id}_blend_mask.png"),
                "--cleanup-mask",
                str(metadata_dir / f"{page_id}_cleanup_mask.png"),
            ]
        )

    blend_prompt = (
        "same exact Little Queen page and unchanged canonical accessories, natural hair contact "
        "along the gold crown band, matched earrings attached at the ears, necklace resting at "
        "the collar, fingers naturally wrapped around the unchanged straight gold staff shaft, "
        "harmonious polished anime storybook shading"
    )
    blend_negative = (
        "different crown, different wand, missing jewelry, extra crown, extra staff, duplicate, "
        "floating object, bent shaft, changed face, changed outfit, changed scene, text, watermark"
    )
    pipe = make_pipeline()
    final_records = []
    for index, record in enumerate(pages, start=1):
        page_id = record["id"]
        image = Image.open(composite_dir / f"{page_id}.png").convert("RGB")
        mask = Image.open(metadata_dir / f"{page_id}_blend_mask.png").convert("L")
        print(f"[blend {index}/{len(pages)}] {page_id}", flush=True)
        with torch.inference_mode():
            final = pipe(
                prompt=blend_prompt,
                negative_prompt=blend_negative,
                image=image,
                mask_image=mask,
                width=image.width,
                height=image.height,
                strength=0.20,
                num_inference_steps=24,
                guidance_scale=4.0,
                padding_mask_crop=48,
                generator=torch.Generator(device="cuda").manual_seed(int(record["seed"]) + 500),
            ).images[0]
        destination = final_dir / f"{page_id}.png"
        final.save(destination, format="PNG", compress_level=6)
        final_records.append({"id": page_id, "output": str(destination)})
        del image, mask, final
        torch.cuda.empty_cache()

    summary = {
        "name": "littlequeen_storybook_accessories_v1_inpaint_validation",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "base_model_sha256": sha256(MODEL),
        "identity_lora_sha256": sha256(IDENTITY),
        "cleanup": cleanup_records,
        "final": final_records,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(clean_dir, arcname="clean")
        archive.add(composite_dir, arcname="composite")
        archive.add(final_dir, arcname="final")
        archive.add(metadata_dir, arcname="metadata")
        archive.add(ROOT / "summary.json", arcname="summary.json")
    print(f"Wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

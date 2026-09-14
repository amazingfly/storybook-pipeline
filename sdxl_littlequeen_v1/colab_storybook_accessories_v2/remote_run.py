#!/usr/bin/env python3
"""Generate bare Little Queen bases, reject inherited props, and add canonical assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/lq_storybook_accessories_v2")
BUNDLE_PREFIX = Path("/content/storybook_accessories_v2_bundle.tar.gz.part_")
IDENTITY_PREFIX = Path("/content/lqxl_sdxl_v2.safetensors.part_")
IDENTITY = ROOT / "models" / "lqxl_sdxl_v2.safetensors"
MODEL = ROOT / "models" / "sd_xl_base_1.0.safetensors"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
ARCHIVE = Path("/content/lq_storybook_accessories_v2_results.tar.gz")
WIDTH = 832
HEIGHT = 1216
WEIGHTS = (0.35, 0.50, 0.65, 0.80)
SCENES = (
    "standing calmly in a quiet sunlit palace courtyard, both open hands visible",
    "walking beside a clear woodland stream, both empty hands visible",
    "turning gently in a moonlit castle garden, both empty hands visible",
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


def make_txt2img():
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
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def make_inpaint():
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


def make_contact_sheet(records: list[dict], key: str, destination: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    cell = (208, 328)
    columns = 4
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        image = Image.open(record[key]).convert("RGB")
        thumb = ImageOps.contain(image, (cell[0], cell[1] - 24))
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        sheet.paste(thumb, (x + (cell[0] - thumb.width) // 2, y))
        label = f"{record['id']} w={record['identity_weight']:.2f}"
        draw.text((x + 4, y + cell[1] - 20), label, fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=93)


def has_inherited_prop(metadata: dict) -> bool:
    blocked = ("crown", "staff", "wand", "sword", "earring", "necklace")
    return any(
        any(label in item["label"] for label in blocked)
        and float(item["score"]) >= 0.25
        for item in metadata["inherited_accessory_detections"]
    )


def main() -> int:
    install_dependencies()
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

    base_dir = ROOT / "base"
    composite_dir = ROOT / "composite"
    final_dir = ROOT / "final"
    metadata_dir = ROOT / "metadata"
    review_dir = ROOT / "review"
    for path in (base_dir, composite_dir, final_dir, metadata_dir, review_dir):
        path.mkdir(parents=True, exist_ok=True)

    positive_template = (
        "wide full-body storybook view with generous clear space above the hair and both shoes "
        "inside frame, solo, lqxl Little Queen, bare uncovered auburn hair with straight bangs, "
        "completely bare head, plain ears, bare neck and wrists, empty hands, {scene}, polished "
        "anime storybook illustration"
    )
    negative = (
        "crown, tiara, diadem, circlet, headband, hat, head ornament, hair ornament, flower in "
        "hair, wand, staff, scepter, sword, weapon, earrings, necklace, bracelet, wrist jewelry, "
        "cropped head, cropped feet, close-up, duplicate, two people, extra limbs, text, watermark"
    )
    records = []
    pipe = make_txt2img()
    candidate = 0
    for weight in WEIGHTS:
        pipe.set_adapters(["identity"], adapter_weights=[weight])
        for scene_index, scene in enumerate(SCENES, start=1):
            candidate += 1
            page_id = f"candidate_{candidate:02d}"
            seed = 140000 + candidate
            print(f"[generate {candidate}/{len(WEIGHTS) * len(SCENES)}] {page_id}", flush=True)
            with torch.inference_mode():
                image = pipe(
                    prompt=positive_template.format(scene=scene),
                    negative_prompt=negative,
                    width=WIDTH,
                    height=HEIGHT,
                    num_inference_steps=30,
                    guidance_scale=5.5,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ).images[0]
            path = base_dir / f"{page_id}.png"
            image.save(path, format="PNG", compress_level=6)
            records.append(
                {
                    "id": page_id,
                    "seed": seed,
                    "identity_weight": weight,
                    "scene": scene,
                    "base": str(path),
                }
            )
            del image
            torch.cuda.empty_cache()
    del pipe
    torch.cuda.empty_cache()
    make_contact_sheet(records, "base", review_dir / "base_candidates.jpg")

    compositor = ROOT / "storybook" / "apply_storybook_accessories.py"
    accepted = []
    for index, record in enumerate(records, start=1):
        page_id = record["id"]
        print(f"[inspect/composite {index}/{len(records)}] {page_id}", flush=True)
        metadata_path = metadata_dir / f"{page_id}.json"
        run(
            [
                sys.executable,
                str(compositor),
                record["base"],
                str(composite_dir / f"{page_id}.png"),
                "--set",
                str(ROOT / "storybook" / "accessory_set.json"),
                "--staff",
                "--metadata",
                str(metadata_path),
                "--blend-mask",
                str(metadata_dir / f"{page_id}_blend_mask.png"),
                "--cleanup-mask",
                str(metadata_dir / f"{page_id}_cleanup_mask.png"),
            ]
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record["metadata"] = str(metadata_path)
        record["composite"] = str(composite_dir / f"{page_id}.png")
        record["accepted_bare_base"] = not has_inherited_prop(metadata)
        if record["accepted_bare_base"]:
            accepted.append(record)
    if not accepted:
        raise RuntimeError("identity LoRA produced no bare, prop-free base candidates")
    make_contact_sheet(accepted, "composite", review_dir / "accepted_composites.jpg")

    blend_prompt = (
        "same exact page and unchanged accessories, natural hair contact at the crown band, "
        "earrings attached at the ears, necklace resting at the collar, fingers naturally "
        "wrapped around the unchanged straight gold staff, polished anime storybook shading"
    )
    blend_negative = (
        "different crown, different wand, changed jewelry, missing prop, extra prop, duplicate, "
        "bent staff, changed face, changed outfit, changed scene, text, watermark"
    )
    pipe = make_inpaint()
    for index, record in enumerate(accepted, start=1):
        page_id = record["id"]
        image = Image.open(record["composite"]).convert("RGB")
        mask = Image.open(metadata_dir / f"{page_id}_blend_mask.png").convert("L")
        print(f"[attachment blend {index}/{len(accepted)}] {page_id}", flush=True)
        with torch.inference_mode():
            final = pipe(
                prompt=blend_prompt,
                negative_prompt=blend_negative,
                image=image,
                mask_image=mask,
                width=image.width,
                height=image.height,
                strength=0.16,
                num_inference_steps=20,
                guidance_scale=3.5,
                padding_mask_crop=48,
                generator=torch.Generator(device="cuda").manual_seed(record["seed"] + 500),
            ).images[0]
        destination = final_dir / f"{page_id}.png"
        final.save(destination, format="PNG", compress_level=6)
        record["final"] = str(destination)
        del image, mask, final
        torch.cuda.empty_cache()
    make_contact_sheet(accepted, "final", review_dir / "accepted_final.jpg")

    summary = {
        "name": "littlequeen_storybook_accessories_v2_bare_base_sweep",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "base_model_sha256": sha256(MODEL),
        "identity_lora_sha256": sha256(IDENTITY),
        "candidate_count": len(records),
        "accepted_count": len(accepted),
        "records": records,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in (base_dir, composite_dir, final_dir, metadata_dir, review_dir):
            archive.add(path, arcname=path.name)
        archive.add(ROOT / "summary.json", arcname="summary.json")
    print(f"Wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

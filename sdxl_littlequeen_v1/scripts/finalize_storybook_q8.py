#!/usr/bin/env python3
"""Apply and validate storybook accessories locally with Q8 SDXL."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

import apply_storybook_accessories as compositor
import validate_storybook_accessories_v4 as validator
from run_sdxl_cpu import ROOT, render_one, valid_png, validate_inputs


REGIONAL = {
    "regalia_weight": 0.45,
    "wand_weight": 0.50,
    "grip_wand_weight": 0.20,
    "regalia_strength": 0.28,
    "crown_cleanup_strength": 0.45,
    "wand_strength": 0.30,
    "grip_strength": 0.68,
}
REGALIA_PROMPT = (
    "lqmoonregalia4, exact five-peak polished gold Moonstar crown, centered oval "
    "rose-amber moonstone, two amber side gems, matched gold leaf-drop earrings, "
    "narrow gold moonstone collar, clean auburn hair and natural background above "
    "the crown, integrated anime storybook linework"
)
REGALIA_NEGATIVE = (
    "different crown, giant crown, extra crown peaks, flower crown, roses, hat, archway, "
    "cage, gold clothing, extra jewelry, changed face, changed hair, changed dress"
)
WAND_PROMPT = (
    "lqmoonwand4, exact single slender gold Moonstar scepter, straight engraved shaft, "
    "open crescent finial around one rose-amber oval moonstone, five leaf crystals, "
    "two round side gems, pointed rose crystal end cap, one small natural hand firmly "
    "wrapped around the shaft with visible fingers, anime storybook linework"
)
WAND_NEGATIVE = (
    "different wand, giant staff, sword, blade, star wand, flower, vine, archway, cage, "
    "freestanding object, floating wand, open palm, extra wand, duplicate, bent shaft, "
    "extra fingers, fused fingers, changed body"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiled_story", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument(
        "--mask-workers",
        type=int,
        default=int(os.environ.get("STORYBOOK_MASK_WORKERS", "2")),
        help="Concurrent local mask workers (default: 2).",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(run_root: Path, path: Path) -> str:
    return str(path.relative_to(run_root))


def has_moonstar(candidate: dict[str, Any]) -> bool:
    return any(
        item["mode"] == "moonstar_regional_set"
        for item in candidate["extra_loras"]
    )


def moonstar_weight(candidate: dict[str, Any]) -> float:
    for item in candidate["extra_loras"]:
        if item["mode"] == "moonstar_regional_set":
            return float(item["weight"])
    return 0.0


def valid_image(path: Path, size: tuple[int, int]) -> bool:
    if not valid_png(path):
        return False
    try:
        with Image.open(path) as image:
            return image.size == size
    except OSError:
        return False


def base_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "seed": int(candidate["seed"]),
        "scene_number": int(candidate["scene"]),
        "candidate_index": int(candidate["candidate_index"]),
        "prompt": candidate["prompt"],
        "script": candidate["script"],
        "base_loras": candidate["base_loras"],
        "extra_loras": candidate["extra_loras"],
        "uses_moonstar_accessories": has_moonstar(candidate),
        "artifacts": {"base": f"base/{candidate['id']}.png"},
        "selected": False,
    }


def accessory_paths(run_root: Path, page_id: str) -> dict[str, Path]:
    return {
        "composite": run_root / "composite" / f"{page_id}.png",
        "metadata": run_root / "metadata" / f"{page_id}.json",
        "blend_mask": run_root / "metadata" / f"{page_id}_blend_mask.png",
        "cleanup_mask": run_root / "metadata" / f"{page_id}_cleanup_mask.png",
        "regalia_mask": run_root / "metadata" / f"{page_id}_regalia_mask.png",
        "wand_mask": run_root / "metadata" / f"{page_id}_wand_mask.png",
        "wand_full_mask": run_root / "metadata" / f"{page_id}_wand_full_mask.png",
        "wand_visible_mask": (
            run_root / "metadata" / f"{page_id}_wand_visible_mask.png"
        ),
        "hand_occlusion_mask": (
            run_root / "metadata" / f"{page_id}_hand_occlusion_mask.png"
        ),
        "hand_grip_mask": run_root / "metadata" / f"{page_id}_hand_grip_mask.png",
    }


def reusable_accessory_paths(
    paths: dict[str, Path],
    size: tuple[int, int],
) -> bool:
    if not paths["metadata"].is_file():
        return False
    try:
        json.loads(paths["metadata"].read_text(encoding="utf-8"))
        for key, path in paths.items():
            if key == "metadata":
                continue
            with Image.open(path) as image:
                if image.size != size:
                    return False
    except (OSError, ValueError):
        return False
    return True


def apply_masks(
    run_root: Path,
    record: dict[str, Any],
    size: tuple[int, int],
    progress: tuple[int, int],
) -> dict[str, Path]:
    page_id = record["id"]
    position, total = progress
    paths = accessory_paths(run_root, page_id)
    if reusable_accessory_paths(paths, size):
        print(
            f"[local mask {position}/{total}] reuse {page_id}",
            flush=True,
        )
    else:
        print(f"[local mask {position}/{total}] {page_id}", flush=True)
        arguments = argparse.Namespace(
            input=run_root / record["artifacts"]["base"],
            output=paths["composite"],
            set_path=ROOT / "storybook_accessories_v2" / "accessory_set.json",
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
            metadata=paths["metadata"],
            blend_mask=paths["blend_mask"],
            cleanup_mask=paths["cleanup_mask"],
            regalia_mask=paths["regalia_mask"],
            hand_prop_mask=paths["wand_mask"],
            hand_prop_full_mask=paths["wand_full_mask"],
            hand_prop_visible_mask=paths["wand_visible_mask"],
            hand_occlusion_mask=paths["hand_occlusion_mask"],
            hand_grip_mask=paths["hand_grip_mask"],
            no_person_occlusion=False,
            quiet=True,
        )
        compositor.main(arguments)
    return paths


def mask_one(
    run_root: Path,
    record: dict[str, Any],
    size: tuple[int, int],
    progress: tuple[int, int],
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = apply_masks(run_root, record, size, progress)
    validation = validator.validate_preflight(
        paths["composite"],
        paths["regalia_mask"],
        paths["wand_mask"],
        paths["metadata"],
    )
    return paths, validation


def make_cpu_config(
    width: int,
    height: int,
    vae_tiling: bool,
) -> dict[str, Any]:
    config = {
        "name": "littlequeen_storybook_q8_local_accessories",
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
            "vae_tiling": vae_tiling,
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
            {
                "prompt": "preflight",
                "seed": 1,
                "loras": [
                    {"name": "lcm-lora-sdxl", "weight": 1.0},
                    {"name": "lqmoonregalia_sdxl_v4", "weight": 0.45},
                ],
            },
            {
                "prompt": "preflight",
                "seed": 1,
                "loras": [
                    {"name": "lcm-lora-sdxl", "weight": 1.0},
                    {"name": "lqmoonwand_sdxl_v4", "weight": 0.50},
                ],
            },
        ],
    }
    validate_inputs(config)
    return config


def exact_mask_merge(
    initial: Path,
    generated: Path,
    mask_path: Path,
    destination: Path,
) -> None:
    with Image.open(initial) as source_image:
        source = source_image.convert("RGB")
    with Image.open(generated) as generated_image:
        result = generated_image.convert("RGB")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
    if result.size != source.size:
        result = result.resize(source.size, Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.composite(result, source, mask).save(
        destination,
        format="PNG",
        compress_level=6,
    )


def render_inpaint(
    cpu_config: dict[str, Any],
    run_root: Path,
    record: dict[str, Any],
    *,
    stage: str,
    stage_index: int,
    prompt: str,
    negative_prompt: str,
    initial: Path,
    mask: Path,
    destination: Path,
    strength: float,
    lora_name: str,
    lora_weight: float,
    seed: int,
    size: tuple[int, int],
) -> dict[str, Any]:
    if valid_image(destination, size):
        print(f"[local {stage}] reuse {record['id']}", flush=True)
        return {"status": "completed", "reused": True}
    raw = (
        run_root
        / "logs"
        / "q8_accessories"
        / "raw"
        / f"{record['id']}_{stage}.png"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    relative_raw = relative(run_root, raw)
    render_record = {
        "prompt": prompt,
        "seed": seed,
        "loras": [
            {"name": "lcm-lora-sdxl", "weight": 1.0},
            {"name": lora_name, "weight": lora_weight},
        ],
        "output_relative": relative_raw,
        "log_stem": f"{record['id']}_{stage}",
        "extra_args": [
            "--init-img",
            str(initial),
            "--mask",
            str(mask),
            "--strength",
            str(strength),
        ],
    }
    print(f"[local {stage}] {record['id']}", flush=True)
    result = render_one(
        {
            **cpu_config,
            "negative_prompt": negative_prompt,
        },
        render_record,
        stage_index,
        run_root,
        run_root / "logs" / "q8_accessories",
    )
    if result["status"] != "completed":
        raise RuntimeError(f"{stage} Q8 refinement failed for {record['id']}")
    exact_mask_merge(initial, raw, mask, destination)
    raw.unlink(missing_ok=True)
    result["output"] = relative(run_root, destination)
    return result


def main() -> int:
    args = parse_args()
    if args.mask_workers < 1:
        raise ValueError("--mask-workers must be at least 1")
    compiled_path = args.compiled_story.resolve()
    run_root = args.run_root.resolve()
    config = json.loads(compiled_path.read_text(encoding="utf-8"))
    width, height = (int(value) for value in config["settings"]["resolution"])
    size = (width, height)
    missing = [
        candidate["id"]
        for candidate in config["candidates"]
        if not valid_image(run_root / "base" / f"{candidate['id']}.png", size)
    ]
    if missing:
        raise RuntimeError(
            f"cannot finalize before base generation completes; {len(missing)} missing"
        )

    for directory in (
        "composite",
        "regalia",
        "wand_only",
        "grip_repaired",
        "final",
        "metadata",
        "review",
        "logs/q8_accessories",
    ):
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    local_compiled = run_root / "compiled_story.json"
    if compiled_path != local_compiled.resolve():
        shutil.copy2(compiled_path, local_compiled)

    summary_path = run_root / "summary.json"
    prior_records = {}
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        prior_records = {
            record["id"]: record for record in prior.get("records", [])
        }
    records = [
        prior_records.get(candidate["id"], base_record(candidate))
        for candidate in config["candidates"]
    ]
    candidates = {
        candidate["id"]: candidate for candidate in config["candidates"]
    }
    summary = {
        "name": "littlequeen_storybook_mvp_v1_q8_local",
        "title": config["title"],
        "status": "local_masking",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "scene_count": len(config["scenes"]),
        "candidate_count": len(records),
        "resolution": [width, height],
        "models": {
            "base": "sd_xl_base_1.0-q8_0.gguf",
            "sampler": "LCM 8-step",
            "validation": validator.VALIDATOR_VERSION,
        },
        "records": records,
    }
    write_json(summary_path, summary)

    for record in records:
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

    mask_records = [
        record for record in records if record["uses_moonstar_accessories"]
    ]
    mask_total = len(mask_records)
    worker_count = min(args.mask_workers, max(mask_total, 1))
    print(
        f"[local mask] workers={worker_count}, total={mask_total}",
        flush=True,
    )
    if worker_count > 1 and any(
        not reusable_accessory_paths(
            accessory_paths(run_root, record["id"]),
            size,
        )
        for record in mask_records
    ):
        compositor.detector_components()
        compositor.sam_components()

    completed_masks = 0
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="storybook-mask",
    ) as executor:
        futures = {
            executor.submit(
                mask_one,
                run_root,
                record,
                size,
                (position, mask_total),
            ): (position, record)
            for position, record in enumerate(mask_records, start=1)
        }
        for future in as_completed(futures):
            position, record = futures[future]
            try:
                paths, preflight_validation = future.result()
                record["artifacts"].update(
                    {
                        key: relative(run_root, path)
                        for key, path in paths.items()
                        if key
                        in {
                            "composite",
                            "metadata",
                            "regalia_mask",
                            "wand_mask",
                            "wand_full_mask",
                            "wand_visible_mask",
                            "hand_occlusion_mask",
                            "hand_grip_mask",
                        }
                    }
                )
                record["preflight_validation"] = preflight_validation
                record["validation"] = preflight_validation
            except Exception as exc:
                record.pop("preflight_validation", None)
                record["validation"] = {
                    "validator_version": validator.VALIDATOR_VERSION,
                    "stage": "preflight",
                    "accepted": False,
                    "score": 0.0,
                    "failures": [f"compositor failure: {exc}"],
                    "advisories": [],
                    "metrics": {},
                }
                print(
                    f"[local mask {position}/{mask_total}] "
                    f"rejected {record['id']}: {exc}",
                    flush=True,
                )
            completed_masks += 1
            if completed_masks % 5 == 0:
                summary["local_masking_completed_count"] = completed_masks
                summary["local_masking_total_count"] = mask_total
                summary["local_masking_workers"] = worker_count
                write_json(summary_path, summary)
    summary["local_masking_completed_count"] = completed_masks
    summary["local_masking_total_count"] = mask_total
    summary["local_masking_workers"] = worker_count
    preflight_records = [
        record
        for record in mask_records
        if record.get("preflight_validation", {}).get("accepted", False)
    ]
    if completed_masks:
        write_json(summary_path, summary)
    compositor.clear_model_caches()
    gc.collect()

    summary["status"] = "local_q8_accessory_refinement"
    write_json(summary_path, summary)
    cpu_config = make_cpu_config(width, height, args.vae_tiling)
    for index, record in enumerate(preflight_records, start=1):
        candidate = candidates[record["id"]]
        weight = moonstar_weight(candidate)
        artifacts = record["artifacts"]
        metadata = json.loads(
            (run_root / artifacts["metadata"]).read_text(encoding="utf-8")
        )
        regalia_destination = run_root / "regalia" / f"{record['id']}.png"
        regalia_strength = (
            REGIONAL["crown_cleanup_strength"]
            if metadata.get("existing_crown_removed")
            else REGIONAL["regalia_strength"]
        )
        metrics = record.setdefault("q8_stage_metrics", {})
        metrics["regalia"] = render_inpaint(
            cpu_config,
            run_root,
            record,
            stage="regalia",
            stage_index=index * 3 - 2,
            prompt=REGALIA_PROMPT,
            negative_prompt=REGALIA_NEGATIVE,
            initial=run_root / artifacts["composite"],
            mask=run_root / artifacts["regalia_mask"],
            destination=regalia_destination,
            strength=regalia_strength,
            lora_name="lqmoonregalia_sdxl_v4",
            lora_weight=REGIONAL["regalia_weight"] * weight,
            seed=int(record["seed"]) + 500,
            size=size,
        )
        artifacts["regalia"] = relative(run_root, regalia_destination)

        wand_destination = run_root / "wand_only" / f"{record['id']}.png"
        metrics["wand_only"] = render_inpaint(
            cpu_config,
            run_root,
            record,
            stage="wand",
            stage_index=index * 3 - 1,
            prompt=WAND_PROMPT,
            negative_prompt=WAND_NEGATIVE,
            initial=regalia_destination,
            mask=run_root / artifacts["wand_mask"],
            destination=wand_destination,
            strength=REGIONAL["wand_strength"],
            lora_name="lqmoonwand_sdxl_v4",
            lora_weight=REGIONAL["wand_weight"] * weight,
            seed=int(record["seed"]) + 900,
            size=size,
        )
        artifacts["wand_only"] = relative(run_root, wand_destination)

        grip_destination = run_root / "grip_repaired" / f"{record['id']}.png"
        metrics["grip_repaired"] = render_inpaint(
            cpu_config,
            run_root,
            record,
            stage="grip",
            stage_index=index * 3,
            prompt=GRIP_PROMPT,
            negative_prompt=GRIP_NEGATIVE,
            initial=wand_destination,
            mask=run_root / artifacts["hand_grip_mask"],
            destination=grip_destination,
            strength=REGIONAL["grip_strength"],
            lora_name="lqmoonwand_sdxl_v4",
            lora_weight=REGIONAL["grip_wand_weight"] * weight,
            seed=int(record["seed"]) + 1200,
            size=size,
        )
        artifacts["grip_repaired"] = relative(run_root, grip_destination)
        record["variant_validation"] = {
            "wand_only": validator.validate_final(
                run_root / artifacts["composite"],
                wand_destination,
                run_root / artifacts["regalia_mask"],
                run_root / artifacts["wand_mask"],
                record["preflight_validation"],
            ),
            "grip_repaired": validator.validate_final(
                run_root / artifacts["composite"],
                grip_destination,
                run_root / artifacts["regalia_mask"],
                run_root / artifacts["wand_mask"],
                record["preflight_validation"],
            ),
        }
        provisional = (
            "grip_repaired"
            if record["variant_validation"]["grip_repaired"]["accepted"]
            else "wand_only"
        )
        record["provisional_variant"] = provisional
        record["artifacts"]["final"] = record["artifacts"][provisional]
        record["validation"] = record["variant_validation"][provisional]
        write_json(summary_path, summary)

    review_pool = []
    for record in records:
        accepted = (
            not record["uses_moonstar_accessories"]
            and record.get("validation", {}).get("accepted", False)
        ) or any(
            item.get("accepted", False)
            for item in record.get("variant_validation", {}).values()
        )
        record["automatic_shortlist"] = accepted
        record["selected"] = False
        if accepted:
            review_pool.append(record)
    covered_scenes = {record["scene_number"] for record in review_pool}
    summary["preflight_accepted_count"] = len(preflight_records)
    summary["geometry_shortlist_count"] = len(review_pool)
    summary["selected_count"] = 0
    summary["requires_visual_review"] = True
    summary["scenes_with_candidates"] = sorted(covered_scenes)
    summary["status"] = (
        "awaiting_local_selection"
        if len(covered_scenes) == len(config["scenes"])
        else "missing_scene_candidates"
    )
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(summary_path, summary)
    validation_report = {
        "validator_version": validator.VALIDATOR_VERSION,
        "record_count": len(records),
        "geometry_shortlist_count": len(review_pool),
        "scene_count": len(config["scenes"]),
        "scenes_with_candidates": sorted(covered_scenes),
        "records": [
            {
                "id": record["id"],
                "validation": record.get("validation", {}),
                "variant_validation": record.get("variant_validation", {}),
            }
            for record in records
        ],
    }
    write_json(run_root / "validation_report.json", validation_report)
    print(
        f"[local validation] {len(review_pool)} candidates across "
        f"{len(covered_scenes)}/{len(config['scenes'])} scenes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise

#!/usr/bin/env python3
"""Validate downloaded FP16 accessory refinements and update the run summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

import validate_storybook_accessories_v4 as validator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-story", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_image(path: Path, size: tuple[int, int]) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == size
    except OSError:
        return False


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    config = json.loads(args.compiled_story.read_text(encoding="utf-8"))
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    size = tuple(int(value) for value in config["settings"]["resolution"])
    missing = []

    for record in summary["records"]:
        if not record.get("uses_moonstar_accessories"):
            record["artifacts"]["final"] = record["artifacts"]["base"]
            continue
        if not record.get("preflight_validation", {}).get("accepted", False):
            continue
        page_id = record["id"]
        artifacts = record["artifacts"]
        expected = {
            "regalia": f"regalia/{page_id}.png",
            "wand_only": f"wand_only/{page_id}.png",
            "grip_repaired": f"grip_repaired/{page_id}.png",
        }
        record_missing = []
        for key, relative_path in expected.items():
            if not valid_image(run_root / relative_path, size):
                missing.append(relative_path)
                record_missing.append(relative_path)
            artifacts[key] = relative_path
        if record_missing:
            continue
        record["variant_validation"] = {
            "wand_only": validator.validate_final(
                run_root / artifacts["composite"],
                run_root / artifacts["wand_only"],
                run_root / artifacts["regalia_mask"],
                run_root / artifacts["wand_mask"],
                record["preflight_validation"],
            ),
            "grip_repaired": validator.validate_final(
                run_root / artifacts["composite"],
                run_root / artifacts["grip_repaired"],
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
        artifacts["final"] = artifacts[provisional]
        record["validation"] = record["variant_validation"][provisional]

    if missing:
        raise RuntimeError(
            f"FP16 refinement is incomplete; {len(set(missing))} outputs are missing"
        )

    review_pool = []
    for record in summary["records"]:
        accepted = (
            not record.get("uses_moonstar_accessories")
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
    summary["name"] = "littlequeen_storybook_mvp_v1_fp16_refinement"
    summary["models"]["accessory_refinement"] = {
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
        "precision": "fp16",
        "steps": 28,
        "device": "Colab GPU",
    }
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
    write_json(
        run_root / "validation_report.json",
        {
            "validator_version": validator.VALIDATOR_VERSION,
            "record_count": len(summary["records"]),
            "geometry_shortlist_count": len(review_pool),
            "scene_count": len(config["scenes"]),
            "scenes_with_candidates": sorted(covered_scenes),
            "records": [
                {
                    "id": record["id"],
                    "validation": record.get("validation", {}),
                    "variant_validation": record.get("variant_validation", {}),
                }
                for record in summary["records"]
            ],
        },
    )
    print(
        f"[fp16 validation] {len(review_pool)} candidates across "
        f"{len(covered_scenes)}/{len(config['scenes'])} scenes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

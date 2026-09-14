#!/usr/bin/env python3
"""Materialize personally reviewed V5 accessory variants as approved pages."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


VARIANTS = {"wand_only", "grip_repaired"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    return parser.parse_args()


def make_contact_sheet(items: list[dict], destination: Path) -> None:
    columns = 4
    cell = (240, 370)
    rows = max(1, (len(items) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(items):
        image = Image.open(item["destination"]).convert("RGB")
        thumb = ImageOps.contain(image, (cell[0] - 8, cell[1] - 48))
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        sheet.paste(thumb, (x + (cell[0] - thumb.width) // 2, y))
        draw.text(
            (x + 4, y + cell[1] - 22),
            f"{item['id']} {item['variant']}",
            fill="white",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def finalize(run_root: Path, review_path: Path) -> dict:
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    source_records = {record["id"]: record for record in summary["records"]}
    approved_dir = run_root / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)
    for stale in approved_dir.glob("*.png"):
        stale.unlink()

    approved = []
    seen = set()
    for decision in review["records"]:
        if decision.get("decision") != "accept":
            continue
        page_id = decision["id"]
        variant = decision.get("variant")
        if page_id in seen:
            raise ValueError(f"duplicate accepted record: {page_id}")
        if page_id not in source_records:
            raise ValueError(f"unknown record: {page_id}")
        if variant not in VARIANTS:
            raise ValueError(f"{page_id} has invalid variant: {variant}")
        source_record = source_records[page_id]
        artifact = source_record.get("artifacts", {}).get(variant)
        if not artifact:
            raise ValueError(f"{page_id} is missing its {variant} artifact")
        source = run_root / artifact
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = approved_dir / f"{len(approved) + 1:02d}_{page_id}_{variant}.png"
        shutil.copy2(source, destination)
        approved.append(
            {
                "id": page_id,
                "variant": variant,
                "reason": decision.get("reason", ""),
                "source": str(source.relative_to(run_root)),
                "destination": str(destination.relative_to(run_root)),
            }
        )
        seen.add(page_id)

    for record in summary["records"]:
        selected = next((item for item in approved if item["id"] == record["id"]), None)
        record["selected"] = selected is not None
        record["selected_variant"] = selected["variant"] if selected else None

    target = int(summary["target_accepted"])
    summary["selected_count"] = len(approved)
    summary["requires_visual_review"] = False
    summary["status"] = (
        "complete" if len(approved) >= target else "insufficient_human_accepted"
    )
    summary["human_review"] = str(review_path.relative_to(run_root))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "version": "littlequeen-storybook-accessories-v5-human-selection",
        "target": target,
        "selected_count": len(approved),
        "status": summary["status"],
        "records": approved,
    }
    manifest_path = run_root / "approved_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(
        [
            {
                **item,
                "destination": run_root / item["destination"],
            }
            for item in approved
        ],
        run_root / "review" / "human_selected_finals.jpg",
    )
    return manifest


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    review_path = (args.review or run_root / "review" / "human_review.json").resolve()
    manifest = finalize(run_root, review_path)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

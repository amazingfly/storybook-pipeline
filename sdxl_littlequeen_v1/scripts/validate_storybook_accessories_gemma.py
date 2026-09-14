#!/usr/bin/env python3
"""Add a strict Gemma vision anatomy and wand-grip check to mask validation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SEMANTIC_VERSION = "littlequeen-accessory-gemma4-semantic-v1"
DEFAULT_START = Path(os.environ.get("LTX_REPO", "/home/derek/projects/agentic/ltxVideo")) / "scripts/start_gemma_vision.sh"
DEFAULT_STOP = Path(os.environ.get("LTX_REPO", "/home/derek/projects/agentic/ltxVideo")) / "scripts/stop_gemma_vision.sh"
PROMPT = """
Judge this Little Queen storybook image strictly for automatic production use.

Accept only when every condition is true:
1. Exactly one Little Queen is present with coherent face and body, exactly two arms,
   and no obvious extra hand, arm, leg, face, or merged limb.
2. There is one clean gold crown naturally placed on her head, with no second crown
   or inherited headwear visibly sticking out behind it.
3. The gold crescent wand is easy to read: its decorated top and a useful length of
   shaft are visible rather than mostly hidden behind hair or the body.
4. A visible hand convincingly holds or closes around the wand shaft. Reject a wand
   that floats beside her, merely passes behind her, or only happens to be nearby.
   The second image is a wand/hand detail crop: hand pixels must touch or overlap the
   shaft in that crop. Even a small visible gap means wand_held_by_hand is false.

Do not reject for dress color/style variation, scene variation, or minor stylization.
Return only this JSON object with no markdown:
{
  "single_little_queen": true,
  "coherent_anatomy": true,
  "single_clean_crown": true,
  "wand_readable": true,
  "wand_held_by_hand": true,
  "decision": "accept",
  "failed_checks": [],
  "notes": "short reason, 20 words or fewer"
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--deterministic-report",
        type=Path,
        help="Optional newer deterministic report to use instead of summary verdicts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--start-command", type=Path, default=DEFAULT_START)
    parser.add_argument("--stop-command", type=Path, default=DEFAULT_STOP)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--ids", nargs="*", help="Optional candidate IDs for a focused check.")
    return parser.parse_args()


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def ensure_server(args: argparse.Namespace) -> bool:
    if healthy(args.base_url):
        return False
    subprocess.run([str(args.start_command)], check=True)
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        if healthy(args.base_url):
            return True
        time.sleep(1)
    raise RuntimeError("Gemma vision server did not become ready")


def encode_image(
    path: Path, crop_box: tuple[int, int, int, int] | None = None
) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if crop_box is not None:
            image = image.crop(crop_box)
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_result(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise RuntimeError(f"Gemma returned no JSON object: {text!r}")
    result = json.loads(match.group(0))
    keys = (
        "single_little_queen",
        "coherent_anatomy",
        "single_clean_crown",
        "wand_readable",
        "wand_held_by_hand",
    )
    if any(not isinstance(result.get(key), bool) for key in keys):
        raise RuntimeError(f"Gemma response has missing booleans: {result}")
    result["decision"] = "accept" if all(result[key] for key in keys) else "reject"
    result["failed_checks"] = [key for key in keys if not result[key]]
    result["notes"] = str(result.get("notes", "")).strip()
    return result


def grip_crop(metadata_path: Path, image_path: Path) -> tuple[int, int, int, int]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    staff = metadata["placements"]["staff"]
    boxes = [staff["box"]]
    if staff.get("detected_hand_box"):
        boxes.append(staff["detected_hand_box"])
    left = min(float(box[0]) for box in boxes)
    top = min(float(box[1]) for box in boxes)
    right = max(float(box[2]) for box in boxes)
    bottom = max(float(box[3]) for box in boxes)
    with Image.open(image_path) as image:
        width, height = image.size
    padding = max(24, round((right - left) * 0.18))
    return (
        max(0, round(left - padding)),
        max(0, round(top - padding)),
        min(width, round(right + padding)),
        min(height, round(bottom + padding)),
    )


def judge(base_url: str, image_path: Path, metadata_path: Path) -> dict:
    crop_box = grip_crop(metadata_path, image_path)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + encode_image(image_path)
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + encode_image(image_path, crop_box)
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 256,
        "reasoning_budget_tokens": 64,
        "reasoning_format": "deepseek",
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        data = json.load(response)
    return parse_result(data["choices"][0]["message"]["content"])


def combined_validation(deterministic: dict, semantic: dict | None) -> dict:
    semantic_accepted = semantic is not None and semantic.get("decision") == "accept"
    failures = list(deterministic.get("failures", []))
    if semantic is None:
        failures.append("semantic validation was unavailable")
    else:
        failures.extend(
            f"Gemma: {item.replace('_', ' ')}" for item in semantic["failed_checks"]
        )
    failures = list(dict.fromkeys(failures))
    return {
        "validator_version": f"{deterministic.get('validator_version')}+{SEMANTIC_VERSION}",
        "stage": "combined",
        "accepted": bool(deterministic.get("accepted")) and semantic_accepted,
        "score": deterministic.get("score", 0) if semantic_accepted else 0,
        "failures": failures,
        "advisories": deterministic.get("advisories", []),
        "metrics": deterministic.get("metrics", {}),
    }


def make_contact(run_root: Path, records: list[dict], destination: Path) -> None:
    columns, cell = 4, (240, 370)
    rows = max(1, (len(records) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        artifacts = record["artifacts"]
        source = artifacts.get("final") or artifacts.get("composite") or artifacts["base"]
        with Image.open(run_root / source) as opened:
            image = ImageOps.fit(opened.convert("RGB"), (240, 330))
        x = index % columns * cell[0]
        y = index // columns * cell[1]
        sheet.paste(image, (x, y))
        accepted = record["validation"]["accepted"]
        color = "#7ee787" if accepted else "#ff7b72"
        draw.text(
            (x + 4, y + 334),
            f"{record['id']} {'PASS' if accepted else 'REJECT'}",
            fill=color,
        )
        reason = "; ".join(record["validation"]["failures"][:1])
        draw.text((x + 4, y + 351), reason[:38], fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=93)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    deterministic_overrides = {}
    if args.deterministic_report:
        deterministic_data = json.loads(
            args.deterministic_report.read_text(encoding="utf-8")
        )
        deterministic_overrides = {
            record["id"]: record["validation"]
            for record in deterministic_data["records"]
        }
    started = ensure_server(args)
    output_records = []
    try:
        for record in summary["records"]:
            if args.ids and record["id"] not in args.ids:
                continue
            artifacts = record["artifacts"]
            image_key = "final" if "final" in artifacts else "composite"
            semantic = None
            if image_key in artifacts:
                image_path = run_root / artifacts[image_key]
                metadata_path = run_root / artifacts["metadata"]
                print(f"[gemma] {record['id']} {image_path.name}", flush=True)
                try:
                    semantic = judge(args.base_url, image_path, metadata_path)
                except Exception as exc:
                    semantic = {
                        "decision": "error",
                        "failed_checks": ["semantic_validation_error"],
                        "notes": f"{type(exc).__name__}: {exc}",
                    }
            deterministic = deterministic_overrides.get(
                record["id"], record["validation"]
            )
            combined = combined_validation(deterministic, semantic)
            output_records.append(
                {
                    "id": record["id"],
                    "artifacts": artifacts,
                    "deterministic_validation": deterministic,
                    "semantic_validation": semantic,
                    "validation": combined,
                }
            )
    finally:
        if started and not args.keep_server:
            subprocess.run([str(args.stop_command)], check=False)

    report = {
        "validator_version": SEMANTIC_VERSION,
        "record_count": len(output_records),
        "accepted_count": sum(item["validation"]["accepted"] for item in output_records),
        "records": output_records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_contact(
        run_root,
        output_records,
        run_root / "review" / "semantic_validation_contact.jpg",
    )
    print(
        json.dumps(
            {
                "validator_version": SEMANTIC_VERSION,
                "record_count": report["record_count"],
                "accepted_count": report["accepted_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

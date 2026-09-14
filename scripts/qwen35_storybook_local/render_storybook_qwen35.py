#!/usr/bin/env python3
"""Rank story images with Qwen3.5-VL, narrate scenes, and assemble an MP4."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


DEFAULT_PIPER = Path("/home/derek/projects/readSelectedText/.venv/bin/piper")
DEFAULT_VOICES = Path("/home/derek/.local/share/piper-tts/voices")
DEFAULT_FFMPEG = Path("/home/derek/miniforge3/bin/ffmpeg")
DEFAULT_FFPROBE = Path("/home/derek/miniforge3/bin/ffprobe")
WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_QWEN_START = WORKFLOW_DIR / "start_qwen_storybook.sh"
DEFAULT_QWEN_STOP = WORKFLOW_DIR / "stop_qwen_storybook.sh"
HARD_CHECKS = (
    "single_character",
    "coherent_anatomy",
    "character_consistency",
    "clean_headwear_and_accessories",
    "wand_readable",
    "wand_held_by_hand",
    "complete_framing",
    "scene_match",
    "no_text_or_watermark",
)
QWEN_ENGINE_VERSION = "storybook-qwen35-4b-q6kl-evidence-v1"
QWEN_CRITERIA = (
    {
        "key": "single_character",
        "weight": 10,
        "source": "full",
        "requires_character": True,
        "question": (
            "Is there exactly one Little Queen likeness? Pass only when there is one "
            "Queen, one Queen head, and no duplicate, reflected, miniature, extra-face, "
            "or background copy of her. Other requested people and animals are allowed."
        ),
    },
    {
        "key": "coherent_anatomy",
        "weight": 15,
        "source": "full",
        "question": (
            "Does the visible body form one coherent person? Inspect the face, torso, "
            "arms, hands, and legs. Fail obvious extra, fused, missing, disconnected, "
            "or severely malformed anatomy. Do not fail merely stylized storybook "
            "proportions."
        ),
    },
    {
        "key": "character_consistency",
        "weight": 12,
        "source": "identity",
        "requires_character": True,
        "question": (
            "Does the candidate clearly depict the same Little Queen identity as the "
            "trusted reference images? Compare the round young face, large dark eyes, "
            "small nose and mouth, auburn hair, and childlike proportions. Ignore "
            "harmless changes in pose, lighting, dress color, jewelry, and hairstyle."
        ),
    },
    {
        "key": "clean_headwear_and_accessories",
        "weight": 8,
        "source": "full",
        "question": (
            "Are visible hats, crowns, jewelry, carried props, and other accessories "
            "clean and coherent? Fail fused or duplicated items, objects merged into a "
            "face or body, or conspicuously broken accessory geometry."
        ),
    },
    {
        "key": "exactly_one_requested_wand",
        "weight": 12,
        "source": "base_prop",
        "requires_accessories": True,
        "question": (
            "Before compositing, are any reported long shafts far enough from the "
            "official wand axis that the overlay would leave a second prop visible?"
        ),
    },
    {
        "key": "wand_readable",
        "weight": 8,
        "source": "full",
        "requires_accessories": True,
        "question": (
            "Is the single requested Moonstar wand visually readable as one coherent "
            "gold shaft with a deliberate decorative head? Fail broken, doubled, "
            "melted, discontinuous, or unrecognizable wand geometry."
        ),
    },
    {
        "key": "wand_held_by_hand",
        "weight": 15,
        "source": "base_hands",
        "requires_accessories": True,
        "question": (
            "Before the wand is composited, is the recorded grip anchor spatially "
            "close to a hand that Qwen independently localized?"
        ),
    },
    {
        "key": "complete_framing",
        "weight": 5,
        "source": "full",
        "question": (
            "Is the illustration framed cleanly for a storybook page? Important "
            "subjects must have complete heads, with enough body, hands, and props "
            "shown to understand the action. Fail a cropped head, crown, or key prop."
        ),
    },
    {
        "key": "scene_match",
        "weight": 12,
        "source": "full",
        "question": (
            "Does the candidate visibly contain the requested location, named subject "
            "types and counts, major action, and important story objects? First inventory "
            "what is actually visible. Fail a missing, substituted, or duplicated major "
            "subject. Never treat words in the request as visual evidence."
        ),
    },
    {
        "key": "no_text_or_watermark",
        "weight": 3,
        "source": "full",
        "question": (
            "Is the artwork free of visible writing, captions, signatures, watermarks, "
            "panel borders, and image-grid artifacts? Incidental book-page marks that "
            "are not readable words are allowed."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--compiled-story", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument(
        "--story-dir-name",
        default="story_qwen35_q6kl",
        help="Run-relative output directory, kept separate from Gemma results.",
    )
    parser.add_argument("--piper", type=Path, default=DEFAULT_PIPER)
    parser.add_argument("--voices-dir", type=Path, default=DEFAULT_VOICES)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--qwen-start", type=Path, default=DEFAULT_QWEN_START)
    parser.add_argument("--qwen-stop", type=Path, default=DEFAULT_QWEN_STOP)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument(
        "--reuse-criteria-report-name",
        help="Reuse non-spatial criterion runs from a report in each scene directory.",
    )
    parser.add_argument(
        "--rerun-criteria",
        default="exactly_one_requested_wand,wand_held_by_hand",
        help="Comma-separated criteria to rerun when a reuse report is supplied.",
    )
    return parser.parse_args()


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def ensure_qwen(args: argparse.Namespace) -> bool:
    if healthy(args.base_url):
        return False
    run([str(args.qwen_start)])
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if healthy(args.base_url):
            return True
        time.sleep(1)
    raise RuntimeError("Qwen vision server did not become ready")


def encode_image(
    path: Path,
    crop_box: tuple[int, int, int, int] | None = None,
    *,
    max_size: tuple[int, int] = (960, 960),
) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if crop_box is not None:
            image = image.crop(crop_box)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=93, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def grip_crop(
    run_root: Path, record: dict, image_path: Path
) -> tuple[int, int, int, int] | None:
    metadata_name = record["artifacts"].get("metadata")
    if not metadata_name:
        return None
    metadata = json.loads((run_root / metadata_name).read_text(encoding="utf-8"))
    staff = metadata.get("placements", {}).get("staff", {})
    boxes = [box for box in (staff.get("box"), staff.get("detected_hand_box")) if box]
    if not boxes:
        return None
    left = min(float(box[0]) for box in boxes)
    top = min(float(box[1]) for box in boxes)
    right = max(float(box[2]) for box in boxes)
    bottom = max(float(box[3]) for box in boxes)
    with Image.open(image_path) as image:
        width, height = image.size
    padding = max(28, round((right - left) * 0.2))
    return (
        max(0, round(left - padding)),
        max(0, round(top - padding)),
        min(width, round(right + padding)),
        min(height, round(bottom + padding)),
    )


def grip_contact_crop(
    run_root: Path, record: dict, image_path: Path
) -> tuple[int, int, int, int] | None:
    metadata_name = record["artifacts"].get("metadata")
    if not metadata_name:
        return None
    metadata = json.loads((run_root / metadata_name).read_text(encoding="utf-8"))
    staff = metadata.get("placements", {}).get("staff", {})
    anchor = staff.get("grip_anchor")
    hand_box = staff.get("detected_hand_box")
    if not anchor or not hand_box:
        return None
    center_x = (float(anchor[0]) + (float(hand_box[0]) + float(hand_box[2])) / 2) / 2
    center_y = (float(anchor[1]) + (float(hand_box[1]) + float(hand_box[3])) / 2) / 2
    extent = max(
        96.0,
        (float(hand_box[2]) - float(hand_box[0])) * 2.2,
        (float(hand_box[3]) - float(hand_box[1])) * 1.8,
    )
    with Image.open(image_path) as image:
        width, height = image.size
    return (
        max(0, round(center_x - extent)),
        max(0, round(center_y - extent)),
        min(width, round(center_x + extent)),
        min(height, round(center_y + extent)),
    )


def parse_qwen_json(text: str) -> dict:
    result = extract_json_object(text)
    for key in HARD_CHECKS:
        if not isinstance(result.get(key), bool):
            raise RuntimeError(f"Qwen response is missing boolean {key}: {result}")
    score = result.get("overall_score")
    if not isinstance(score, (int, float)):
        raise RuntimeError(f"Qwen response is missing overall_score: {result}")
    result["overall_score"] = max(0, min(100, round(float(score), 1)))
    result["notes"] = str(result.get("notes", "")).strip()
    result["failed_checks"] = [key for key in HARD_CHECKS if not result[key]]
    result["accepted"] = not result["failed_checks"] and result["overall_score"] >= 70
    return result


def parse_scene_qwen_json(text: str, expected_ids: set[str]) -> dict:
    result = extract_json_object(text)
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError(f"Qwen response has no candidates array: {result}")
    parsed = {}
    for item in candidates:
        candidate_id = str(item.get("id", "")).upper()
        if candidate_id not in expected_ids or candidate_id in parsed:
            continue
        for key in HARD_CHECKS:
            if not isinstance(item.get(key), bool):
                raise RuntimeError(
                    f"Qwen candidate {candidate_id} is missing boolean {key}"
                )
        score = item.get("overall_score")
        if not isinstance(score, (int, float)):
            raise RuntimeError(
                f"Qwen candidate {candidate_id} is missing overall_score"
            )
        item["id"] = candidate_id
        item["overall_score"] = max(0, min(100, round(float(score), 1)))
        item["notes"] = str(item.get("notes", "")).strip()
        item["failed_checks"] = [key for key in HARD_CHECKS if not item[key]]
        item["accepted"] = not item["failed_checks"] and item["overall_score"] >= 70
        parsed[candidate_id] = item
    missing = expected_ids - set(parsed)
    if missing:
        raise RuntimeError(f"Qwen omitted candidates: {sorted(missing)}")
    selected_id = str(result.get("selected_id", "")).upper()
    if selected_id not in expected_ids:
        selected_id = ""
    return {"candidates": parsed, "selected_id": selected_id}


def selection_prompt(record: dict, candidate_ids: list[str] | None = None) -> str:
    character = ", ".join(item["label"] for item in record["base_loras"] if item)
    extras = ", ".join(item["label"] for item in record["extra_loras"])
    accessory_required = record["uses_moonstar_accessories"]
    accessory_rule = (
        """
The requested Moonstar set must contain exactly one clean gold crown, coherent
matching jewelry, and one readable gold crescent wand. A visible hand must
convincingly close around and touch the wand shaft. Reject floating, nearby,
behind-body, or gap-separated wands. Reject any second wand, staff, or long
star-tipped rod, even when the scene contains a magical star. The second image
is a hand/wand detail crop."""
        if accessory_required
        else """
No special crown or wand set is required. Set both wand fields true when no wand
was requested and none creates a visual defect. Judge any visible accessories for
basic coherence only."""
    )
    comparison = ""
    if candidate_ids:
        comparison = f"""
The first supplied image is a labeled comparison sheet. The second is a labeled
close view of the wand-hand region in the same order. Evaluate every ID exactly
once: {", ".join(candidate_ids)}. Compare candidates directly; do not assume the
first image is best."""
    return f"""
You are selecting one publishable illustration from up to ten attempts for a children's
storybook. Inspect strictly, but do not invent defects.

Scene {record['scene_number']} request:
{record['prompt']}

Requested character/outfit LoRAs: {character or 'none'}
Other requested LoRAs: {extras or 'none'}
{accessory_rule}
{comparison}

Rules:
1. Exactly one intended protagonist. Reject doubles, extra faces, or background
   copies of the protagonist.
2. Face, eyes, hair, torso, arms, hands, and legs must form one coherent person.
   Reject obvious extra/fused/missing limbs or conspicuously malformed hands.
3. The result must read as the same Little Queen design: young auburn-haired
   storybook queen, stable face, and coherent royal dress. Ignore harmless color
   variation.
4. Her full head and crown, both hands when the prompt makes them relevant, and
   enough of her body must be in frame. Reject a cropped crown or missing top of head.
5. The major action, location, and important story object in the scene request must
   be visibly represented. Do not award scene_match based only on mood or color.
6. Reject visible text, signature, watermark, or image-grid artifacts.
7. overall_score is 0-100 for suitability relative to the other attempts. Anatomy,
   character identity, requested equipment, and scene accuracy matter most.

Return only JSON. When judging one image, use this exact shape:
{{
  "single_character": true,
  "coherent_anatomy": true,
  "character_consistency": true,
  "clean_headwear_and_accessories": true,
  "wand_readable": true,
  "wand_held_by_hand": true,
  "complete_framing": true,
  "scene_match": true,
  "no_text_or_watermark": true,
  "overall_score": 85,
  "notes": "specific reason in 24 words or fewer"
}}
""".strip()


def batch_selection_prompt(record: dict, candidate_ids: list[str]) -> str:
    base = selection_prompt(record, candidate_ids)
    return (
        base.rsplit("Return only JSON.", 1)[0]
        + """
Return only this JSON shape. Include every requested ID exactly once:
{
  "candidates": [
    {
      "id": "C01",
      "single_character": true,
      "coherent_anatomy": true,
      "character_consistency": true,
      "clean_headwear_and_accessories": true,
      "wand_readable": true,
      "wand_held_by_hand": true,
      "complete_framing": true,
      "scene_match": true,
      "no_text_or_watermark": true,
      "overall_score": 85,
      "notes": "specific reason in 18 words or fewer"
    }
  ],
  "selected_id": "C01"
}
selected_id must be the strongest publishable candidate, not merely the least bad.
""".strip()
    )


def judge(
    base_url: str,
    run_root: Path,
    record: dict,
    image_path: Path,
) -> dict:
    content: list[dict] = [
        {"type": "text", "text": selection_prompt(record)},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + encode_image(image_path)
            },
        },
    ]
    crop_box = grip_crop(run_root, record, image_path)
    if crop_box is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + encode_image(image_path, crop_box)
                },
            }
        )
    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 384,
        "reasoning_budget_tokens": 96,
        "reasoning_format": "deepseek",
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    return parse_qwen_json(data["choices"][0]["message"]["content"])


def make_qwen_sheet(
    candidates: list[dict],
    destination: Path,
    *,
    run_root: Path,
    grip_details: bool,
) -> None:
    columns = min(5, len(candidates))
    cell = (320, 360 if grip_details else 500)
    rows = max(1, (len(candidates) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#111111")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(candidates):
        with Image.open(item["source_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if grip_details:
            crop = grip_crop(run_root, item["record"], item["source_path"])
            if crop is not None:
                image = image.crop(crop)
        image = ImageOps.contain(
            image,
            (cell[0] - 10, cell[1] - 48),
            Image.Resampling.LANCZOS,
        )
        x = index % columns * cell[0]
        y = index // columns * cell[1]
        sheet.paste(image, (x + (cell[0] - image.width) // 2, y))
        draw.rectangle(
            (x + 2, y + cell[1] - 42, x + cell[0] - 3, y + cell[1] - 3),
            fill="#000000",
        )
        draw.text(
            (x + 12, y + cell[1] - 35),
            f"{item['label']}  candidate {item['candidate_index']:02d}",
            fill="#ffffff",
            stroke_width=1,
            stroke_fill="#000000",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def judge_scene(
    base_url: str,
    run_root: Path,
    record: dict,
    candidates: list[dict],
    full_sheet: Path,
    grip_sheet: Path,
) -> dict:
    candidate_ids = [item["label"] for item in candidates]
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": batch_selection_prompt(record, candidate_ids),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + encode_image(full_sheet, max_size=(1600, 1200))
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + encode_image(grip_sheet, max_size=(1600, 1000))
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 1400,
        "reasoning_budget_tokens": 96,
        "reasoning_format": "deepseek",
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        data = json.load(response)
    return parse_scene_qwen_json(
        data["choices"][0]["message"]["content"], set(candidate_ids)
    )


def make_reference_sheet(paths: list[Path], destination: Path) -> None:
    if not paths:
        raise ValueError("identity reference sheet requires at least one image")
    columns = min(4, len(paths))
    cell = (300, 440)
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#111111")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image = ImageOps.contain(
            image, (cell[0] - 12, cell[1] - 46), Image.Resampling.LANCZOS
        )
        x = index % columns * cell[0]
        y = index // columns * cell[1]
        sheet.paste(image, (x + (cell[0] - image.width) // 2, y))
        draw.rectangle(
            (x + 2, y + cell[1] - 40, x + cell[0] - 3, y + cell[1] - 3),
            fill="#000000",
        )
        draw.text(
            (x + 10, y + cell[1] - 32),
            f"TRUSTED REFERENCE {index + 1}",
            fill="#ffffff",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def criterion_prompt(
    criterion: dict,
    record: dict,
    candidate_ids: list[str],
    *,
    has_identity_references: bool,
    retry: bool = False,
) -> str:
    source = criterion["source"]
    image_explanation = (
        "The first image contains labeled grip close-ups. The second contains the "
        "matching full illustrations for context."
        if source == "grip"
        else (
            (
                "The first image contains labeled candidates. The second image "
                "contains trusted identity references; do not score the references."
                if has_identity_references
                else (
                    "The supplied image contains labeled candidates. Compare their "
                    "shared facial design and use the written identity description."
                )
            )
            if source == "identity"
            else "The supplied image contains the labeled candidates."
        )
    )
    scene_context = ""
    if criterion["key"] in {"single_character", "scene_match"}:
        scene_context = (
            "\nRequested scene (requirements only, never visual evidence):\n"
            f"{record['prompt']}\n"
        )
    retry_note = (
        "\nThis is a fresh visual retry. Start the final answer directly with `{` "
        "and follow the schema literally."
        if retry
        else ""
    )
    comparison_artifact_note = ""
    if criterion["key"] == "no_text_or_watermark":
        comparison_artifact_note = """
The black candidate-ID strips, ID text, and gutters surrounding the artworks
were added by this evaluator. Ignore those external labels and gutters. Judge
only writing, watermarks, borders, or grid artifacts inside each artwork area.
"""
    example = {
        "criterion": criterion["key"],
        "candidates": [
            {
                "id": candidate_id,
                "pass": True,
                "quality": 4,
                "reason": "replace with brief visible evidence",
            }
            for candidate_id in candidate_ids
        ],
    }
    return f"""
You are performing one narrow visual inspection for children's storybook art.
Judge ONLY the criterion named {criterion['key']}. Do not reward or penalize any
other quality. Treat each labeled panel as an independent image. Inspect the
visible pixels before reading the requested scene. Never infer an unseen person,
animal, object, action, or defect from the request or from another panel.

Question:
{criterion['question']}
{scene_context}
{image_explanation}
{comparison_artifact_note}
Required candidate IDs: {", ".join(candidate_ids)}

Use this fixed evidence scale:
4 = definite clean pass
3 = pass, with a minor harmless imperfection
2 = uncertain or borderline; this is a fail
1 = clear fail
0 = severe or unmistakable fail

Return every required ID exactly once. "pass" must be true only for quality 3 or 4.
Each reason must state literal visible evidence or what is visibly missing, in 24
words or fewer. Do not merely repeat the requested scene.
Return only JSON. This complete structural example contains every required ID;
replace every example judgment and reason with your actual visual findings:
{json.dumps(example, indent=2)}
{retry_note}
""".strip()


def criterion_response_schema(
    criterion_key: str, candidate_ids: list[str]
) -> dict:
    return {
        "type": "object",
        "properties": {
            "criterion": {"const": criterion_key},
            "candidates": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": candidate_ids},
                        "pass": {"type": "boolean"},
                        "quality": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 4,
                        },
                        "reason": {"type": "string", "maxLength": 180},
                    },
                    "required": ["id", "pass", "quality", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["criterion", "candidates"],
        "additionalProperties": False,
    }


def request_qwen_json(
    base_url: str,
    prompt: str,
    image_paths: list[
        tuple[Path, tuple[int, int]]
        | tuple[Path, tuple[int, int], tuple[int, int, int, int] | None]
    ],
    *,
    max_tokens: int = 1400,
    reasoning_budget_tokens: int = 192,
    response_schema: dict | None = None,
    enable_thinking: bool = False,
) -> str:
    # Keep repeated comparison images before the changing criterion so llama.cpp
    # can reuse their multimodal prefix between atomic questions.
    content: list[dict] = []
    for image_spec in image_paths:
        path, max_size = image_spec[:2]
        crop_box = image_spec[2] if len(image_spec) == 3 else None
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + encode_image(path, crop_box, max_size=max_size)
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "reasoning_budget_tokens": reasoning_budget_tokens,
        "reasoning_format": "deepseek",
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "response_format": (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "criterion_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
            if response_schema is not None
            else {"type": "json_object"}
        ),
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"]


def repair_criterion_response(
    base_url: str,
    criterion_key: str,
    candidate_ids: list[str],
    malformed_response: str,
) -> str:
    prompt = f"""
You are a JSON serializer, not a visual evaluator. Convert the supplied
evaluation into exactly one JSON object matching the required structure.
Preserve its stated pass/fail judgments, quality values, and visible evidence.
Do not add commentary, markdown, or new visual claims.

Required criterion: {criterion_key}
Required IDs, exactly once and in this order: {", ".join(candidate_ids)}

Malformed evaluation:
<evaluation>
{malformed_response}
</evaluation>

Return only the corrected JSON object.
""".strip()
    return request_qwen_json(
        base_url,
        prompt,
        [],
        max_tokens=1200,
        reasoning_budget_tokens=0,
        response_schema=criterion_response_schema(criterion_key, candidate_ids),
        enable_thinking=False,
    )


def parse_criterion_json(
    text: str, criterion_key: str, expected_ids: set[str]
) -> dict[str, dict]:
    document = extract_json_object(text)
    if str(document.get("criterion", "")).strip() != criterion_key:
        raise RuntimeError(
            f"Qwen returned criterion {document.get('criterion')!r}, "
            f"expected {criterion_key!r}"
        )
    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("Qwen criterion response has no candidates array")
    parsed: dict[str, dict] = {}
    for raw in raw_candidates:
        candidate_id = str(raw.get("id", "")).upper()
        if candidate_id not in expected_ids or candidate_id in parsed:
            continue
        stated_pass = raw.get("pass")
        quality = raw.get("quality")
        if not isinstance(stated_pass, bool):
            raise RuntimeError(f"{candidate_id} has no boolean pass value")
        if isinstance(quality, bool) or not isinstance(quality, (int, float)):
            raise RuntimeError(f"{candidate_id} has no numeric quality value")
        quality = max(0, min(4, round(float(quality))))
        quality = max(3, quality) if stated_pass else min(2, quality)
        parsed[candidate_id] = {
            "pass": stated_pass,
            "quality": quality,
            "reason": str(raw.get("reason", "")).strip(),
        }
    missing = expected_ids - set(parsed)
    if missing:
        raise RuntimeError(f"Qwen omitted candidates: {sorted(missing)}")
    return parsed


def aggregate_criterion_results(
    candidate_ids: list[str], criterion_runs: dict[str, dict]
) -> dict[str, dict]:
    aggregated: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        checks = {}
        failures = []
        weighted_points = 0.0
        available_weight = 0.0
        for criterion in QWEN_CRITERIA:
            key = criterion["key"]
            criterion_run = criterion_runs.get(key)
            if criterion_run is None:
                checks[key] = {
                    "applicable": False,
                    "pass": True,
                    "quality": 4,
                    "reason": "Not applicable to this scene.",
                }
                continue
            result = criterion_run["candidates"][candidate_id]
            checks[key] = {"applicable": True, **result}
            available_weight += float(criterion["weight"])
            weighted_points += (
                float(criterion["weight"]) * float(result["quality"]) / 4.0
            )
            if not result["pass"]:
                failures.append(key)
        score = 100.0 * weighted_points / available_weight if available_weight else 0
        failure_notes = [
            f"{key}: {checks[key]['reason']}" for key in failures[:2]
        ]
        aggregated[candidate_id] = {
            **{key: value["pass"] for key, value in checks.items()},
            "criteria": checks,
            "overall_score": round(score, 1),
            "failed_checks": failures,
            "accepted": not failures,
            "notes": (
                "; ".join(failure_notes)
                if failure_notes
                else "Passed all isolated Qwen criteria."
            ),
        }
    return aggregated


def criterion_run_failed(
    criterion_run: dict | None, expected_ids: set[str]
) -> bool:
    if not isinstance(criterion_run, dict):
        return True
    candidates = criterion_run.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != expected_ids:
        return True
    return any(
        not isinstance(result, dict)
        or result.get("reason") == "Criterion evaluator failed."
        for result in candidates.values()
    )


def response_mentions_all_candidates(
    response: str, candidate_ids: list[str]
) -> bool:
    upper = response.upper()
    return all(
        re.search(rf"(?<![A-Z0-9]){re.escape(candidate_id)}(?![A-Z0-9])", upper)
        for candidate_id in candidate_ids
    )


def extract_json_object(text: str) -> dict:
    cleaned = re.sub(
        r"<think>.*?</think>", "", text.strip(), flags=re.IGNORECASE | re.DOTALL
    )
    decoder = json.JSONDecoder()
    errors = []
    for match in re.finditer(r"\{", cleaned):
        try:
            document, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(document, dict):
            return document
    detail = errors[0] if errors else "no opening JSON object"
    raise RuntimeError(f"Qwen returned no valid JSON object ({detail}): {text!r}")


def preexisting_prop_criterion(
    base_url: str, run_root: Path, candidates: list[dict], criterion: dict
) -> dict:
    prompt = """
Inventory visible magical or handheld objects that have a clearly traceable long
narrow shaft at least as long as the girl's forearm. For each qualifying shaft,
report the horizontal center of the shaft as percent from the image's left edge.
Omit shaftless stars, glows, books, hair, limbs, architecture, railings, shelves,
trees, and uncertain objects.

Return only JSON:
{
  "shafted_objects": [
    {
      "x_percent": 50,
      "description": "head shape and visible shaft"
    }
  ],
  "count": 1
}
Do not invent hidden shafts.
""".strip()
    results = {}
    raw_responses = {}
    errors_by_candidate = {}
    for index, candidate in enumerate(candidates, 1):
        label = candidate["label"]
        print(
            f"[qwen evidence] {criterion['key']} {index}/{len(candidates)} {label}",
            flush=True,
        )
        record = candidate["record"]
        base_path = run_root / record["artifacts"]["base"]
        metadata_name = record["artifacts"].get("metadata")
        metadata_path = run_root / metadata_name if metadata_name else None
        if metadata_path is None or not metadata_path.is_file():
            raw_responses[label] = ""
            errors_by_candidate[label] = [
                "Accessory placement metadata is unavailable."
            ]
            results[label] = {
                "pass": False,
                "quality": 0,
                "reason": "Accessory placement metadata is unavailable.",
            }
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        wand_box = metadata.get("placements", {}).get("staff", {}).get("box")
        with Image.open(base_path) as image:
            width = image.width
        canonical_x = (
            (float(wand_box[0]) + float(wand_box[2])) * 50.0 / width
            if wand_box
            else None
        )
        errors = []
        document = None
        shafted_objects = []
        raw_response = ""
        for _ in range(2):
            try:
                raw_response = request_qwen_json(
                    base_url, prompt, [(base_path, (1200, 1200))]
                )
                document = extract_json_object(raw_response)
                if not isinstance(document.get("shafted_objects"), list):
                    raise RuntimeError("shafted_objects is not an array")
                shafted_objects = []
                for raw_object in document["shafted_objects"]:
                    x = raw_object.get("x_percent")
                    if (
                        isinstance(x, bool)
                        or not isinstance(x, (int, float))
                        or not 0 <= float(x) <= 100
                    ):
                        continue
                    shafted_objects.append(
                        {
                            "x_percent": round(float(x), 2),
                            "description": str(
                                raw_object.get("description", "")
                            ).strip(),
                        }
                    )
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                document = None
        raw_responses[label] = raw_response
        errors_by_candidate[label] = errors
        if document is None or canonical_x is None:
            results[label] = {
                "pass": False,
                "quality": 0,
                "reason": "Pre-overlay shaft localization failed.",
            }
            continue
        for item in shafted_objects:
            item["distance_from_canonical_axis"] = round(
                abs(item["x_percent"] - canonical_x), 2
            )
        off_axis = [
            item
            for item in shafted_objects
            if item["distance_from_canonical_axis"] > 18.0
        ]
        results[label] = {
            "pass": not off_axis,
            "quality": 0 if off_axis else (3 if shafted_objects else 4),
            "reason": (
                f"Found {len(off_axis)} shaft(s) away from the replacement axis."
                if off_axis
                else "No shaft lies outside the replacement wand axis."
            ),
            "canonical_wand_x_percent": round(canonical_x, 2),
            "localized_shafts": shafted_objects,
            "off_axis_shafts": off_axis,
        }
    return {
        "weight": criterion["weight"],
        "question": criterion["question"],
        "source": criterion["source"],
        "maximum_axis_distance": 18.0,
        "candidates": results,
        "raw_responses": raw_responses,
        "errors": errors_by_candidate,
    }


def hand_distance_grade(distance: float) -> tuple[bool, int]:
    return (
        distance <= 25.0,
        4
        if distance <= 12.0
        else 3
        if distance <= 25.0
        else 1
        if distance <= 35.0
        else 0,
    )


def hand_anchor_criterion(
    base_url: str, run_root: Path, candidates: list[dict], criterion: dict
) -> dict:
    prompt = """
Locate every visibly recognizable human hand in this full portrait. Do not include
leaves, stars, jewelry, hair, or fabric. Coordinates are percentages from the
image's left and top edges. Estimate each visible hand center. Include a partially
visible hand only when it is connected to a visible arm or sleeve.

Return only JSON:
{
  "hands": [
    {
      "x_percent": 50,
      "y_percent": 50,
      "side_in_image": "left or right",
      "evidence": "brief visible description"
    }
  ],
  "hand_count": 1
}
""".strip()
    results = {}
    raw_responses = {}
    errors_by_candidate = {}
    for index, candidate in enumerate(candidates, 1):
        label = candidate["label"]
        print(
            f"[qwen evidence] {criterion['key']} {index}/{len(candidates)} {label}",
            flush=True,
        )
        record = candidate["record"]
        base_path = run_root / record["artifacts"]["base"]
        metadata_name = record["artifacts"].get("metadata")
        metadata_path = run_root / metadata_name if metadata_name else None
        if metadata_path is None or not metadata_path.is_file():
            raw_responses[label] = ""
            errors_by_candidate[label] = [
                "Accessory placement metadata is unavailable."
            ]
            results[label] = {
                "pass": False,
                "quality": 0,
                "reason": "Accessory placement metadata is unavailable.",
                "grip_anchor_percent": None,
                "localized_hands": [],
                "nearest_hand_distance": None,
            }
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        anchor = metadata.get("placements", {}).get("staff", {}).get(
            "grip_anchor"
        )
        with Image.open(base_path) as image:
            width, height = image.size
        errors = []
        document = None
        hands = []
        raw_response = ""
        for _ in range(2):
            try:
                raw_response = request_qwen_json(
                    base_url, prompt, [(base_path, (1100, 1100))]
                )
                document = extract_json_object(raw_response)
                if not isinstance(document.get("hands"), list):
                    raise RuntimeError("hands is not an array")
                hands = []
                for hand in document["hands"]:
                    x = hand.get("x_percent")
                    y = hand.get("y_percent")
                    if (
                        isinstance(x, bool)
                        or isinstance(y, bool)
                        or not isinstance(x, (int, float))
                        or not isinstance(y, (int, float))
                    ):
                        continue
                    if 0 <= float(x) <= 100 and 0 <= float(y) <= 100:
                        hands.append(
                            {
                                "x_percent": round(float(x), 2),
                                "y_percent": round(float(y), 2),
                                "evidence": str(
                                    hand.get("evidence", "")
                                ).strip(),
                            }
                        )
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                document = None
        raw_responses[label] = raw_response
        errors_by_candidate[label] = errors
        if document is None or not anchor:
            results[label] = {
                "pass": False,
                "quality": 0,
                "reason": "Hand localization or grip anchor was unavailable.",
            }
            continue
        anchor_percent = (
            float(anchor[0]) * 100.0 / width,
            float(anchor[1]) * 100.0 / height,
        )
        distances = [
            (
                (hand["x_percent"] - anchor_percent[0]) ** 2
                + (hand["y_percent"] - anchor_percent[1]) ** 2
            )
            ** 0.5
            for hand in hands
        ]
        nearest = min(distances) if distances else float("inf")
        passed, quality = hand_distance_grade(nearest)
        results[label] = {
            "pass": passed,
            "quality": quality,
            "reason": (
                f"Nearest localized hand is {nearest:.1f} percentage points "
                "from the grip anchor."
                if distances
                else "Qwen localized no visible hands."
            ),
            "grip_anchor_percent": [
                round(anchor_percent[0], 2),
                round(anchor_percent[1], 2),
            ],
            "localized_hands": hands,
            "nearest_hand_distance": (
                round(nearest, 2) if distances else None
            ),
        }
    return {
        "weight": criterion["weight"],
        "question": criterion["question"],
        "source": criterion["source"],
        "threshold_percentage_points": 25.0,
        "candidates": results,
        "raw_responses": raw_responses,
        "errors": errors_by_candidate,
    }


def applicable_criteria(record: dict) -> list[dict]:
    has_character = any(
        lora.get("character") == "little_queen"
        or str(lora.get("name", "")).startswith("little_queen")
        or str(lora.get("id", "")).startswith("little_queen")
        for lora in record.get("base_loras", [])
    )
    return [
        criterion
        for criterion in QWEN_CRITERIA
        if (
            (
                not criterion.get("requires_accessories")
                or record["uses_moonstar_accessories"]
            )
            and (not criterion.get("requires_character") or has_character)
        )
    ]


def judge_scene_multipass(
    base_url: str,
    run_root: Path,
    record: dict,
    candidates: list[dict],
    full_sheet: Path,
    grip_sheet: Path,
    reference_sheet: Path | None,
    reused_criterion_runs: dict[str, dict] | None = None,
    rerun_criteria: set[str] | None = None,
) -> dict:
    candidate_ids = [item["label"] for item in candidates]
    expected_ids = set(candidate_ids)
    criterion_runs: dict[str, dict] = {}
    applicable = applicable_criteria(record)
    for index, criterion in enumerate(applicable, 1):
        print(
            f"[qwen criterion] scene {record['scene_number']:03d} "
            f"{index}/{len(applicable)} {criterion['key']}",
            flush=True,
        )
        if (
            reused_criterion_runs
            and criterion["key"] in reused_criterion_runs
            and criterion["key"] not in (rerun_criteria or set())
        ):
            print(
                f"[qwen criterion] reusing archived {criterion['key']}",
                flush=True,
            )
            reused = reused_criterion_runs[criterion["key"]]
            if criterion["key"] == "wand_held_by_hand":
                for result in reused["candidates"].values():
                    distance = result.get("nearest_hand_distance")
                    if isinstance(distance, (int, float)):
                        result["pass"], result["quality"] = hand_distance_grade(
                            float(distance)
                        )
            criterion_runs[criterion["key"]] = reused
            continue
        if criterion["source"] == "base_prop":
            criterion_runs[criterion["key"]] = preexisting_prop_criterion(
                base_url, run_root, candidates, criterion
            )
            continue
        if criterion["source"] == "base_hands":
            criterion_runs[criterion["key"]] = hand_anchor_criterion(
                base_url, run_root, candidates, criterion
            )
            continue
        if criterion["source"] == "grip":
            image_paths = [
                (grip_sheet, (1600, 1000)),
                (full_sheet, (1600, 1200)),
            ]
        elif criterion["source"] == "identity" and reference_sheet is not None:
            image_paths = [
                (full_sheet, (1600, 1200)),
                (reference_sheet, (1400, 1000)),
            ]
        else:
            image_paths = [(full_sheet, (1600, 1200))]
        errors = []
        raw_response = ""
        parsed = None
        archived = (
            reused_criterion_runs.get(criterion["key"])
            if reused_criterion_runs
            else None
        )
        archived_raw = (
            archived.get("raw_response", "")
            if isinstance(archived, dict)
            else ""
        )
        if archived_raw:
            try:
                parsed = parse_criterion_json(
                    archived_raw, criterion["key"], expected_ids
                )
                raw_response = archived_raw
                print(
                    f"[qwen criterion] recovered archived {criterion['key']} "
                    "with tolerant JSON parsing",
                    flush=True,
                )
            except Exception as exc:
                errors.append(f"archived {type(exc).__name__}: {exc}")
                if response_mentions_all_candidates(
                    archived_raw, candidate_ids
                ):
                    try:
                        raw_response = repair_criterion_response(
                            base_url,
                            criterion["key"],
                            candidate_ids,
                            archived_raw,
                        )
                        parsed = parse_criterion_json(
                            raw_response, criterion["key"], expected_ids
                        )
                        print(
                            f"[qwen criterion] repaired archived "
                            f"{criterion['key']} without vision",
                            flush=True,
                        )
                    except Exception as repair_exc:
                        errors.append(
                            f"archived repair {type(repair_exc).__name__}: "
                            f"{repair_exc}"
                        )
        for attempt in range(2):
            if parsed is not None:
                break
            try:
                raw_response = request_qwen_json(
                    base_url,
                    criterion_prompt(
                        criterion,
                        record,
                        candidate_ids,
                        has_identity_references=reference_sheet is not None,
                        retry=bool(attempt),
                    ),
                    image_paths,
                    reasoning_budget_tokens=192 if attempt == 0 else 64,
                    response_schema=criterion_response_schema(
                        criterion["key"], candidate_ids
                    ),
                )
                parsed = parse_criterion_json(
                    raw_response, criterion["key"], expected_ids
                )
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                print(
                    f"WARNING: {criterion['key']} attempt {attempt + 1} failed: {exc}",
                    flush=True,
                )
                if (
                    raw_response
                    and response_mentions_all_candidates(
                        raw_response, candidate_ids
                    )
                ):
                    try:
                        raw_response = repair_criterion_response(
                            base_url,
                            criterion["key"],
                            candidate_ids,
                            raw_response,
                        )
                        parsed = parse_criterion_json(
                            raw_response, criterion["key"], expected_ids
                        )
                        print(
                            f"[qwen criterion] repaired {criterion['key']} "
                            "without another vision pass",
                            flush=True,
                        )
                        break
                    except Exception as repair_exc:
                        errors.append(
                            f"repair {type(repair_exc).__name__}: {repair_exc}"
                        )
        if parsed is None:
            parsed = {
                candidate_id: {
                    "pass": False,
                    "quality": 0,
                    "reason": "Criterion evaluator failed.",
                }
                for candidate_id in candidate_ids
            }
        criterion_runs[criterion["key"]] = {
            "weight": criterion["weight"],
            "question": criterion["question"],
            "source": criterion["source"],
            "candidates": parsed,
            "raw_response": raw_response,
            "errors": errors,
        }
    aggregated = aggregate_criterion_results(candidate_ids, criterion_runs)
    accepted = [
        candidate_id
        for candidate_id in candidate_ids
        if aggregated[candidate_id]["accepted"]
    ]
    selected_id = max(
        accepted or candidate_ids,
        key=lambda candidate_id: aggregated[candidate_id]["overall_score"],
    )
    return {
        "engine_version": QWEN_ENGINE_VERSION,
        "selected_id": selected_id,
        "criterion_runs": criterion_runs,
        "candidates": aggregated,
    }


def audio_duration(ffprobe: Path, path: Path) -> float:
    result = run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def narrate(args: argparse.Namespace, config: dict, scene: dict, directory: Path) -> float:
    tts = config["settings"]["tts"]
    if tts["engine"] != "piper":
        raise RuntimeError(f"unsupported TTS engine: {tts['engine']}")
    voice = args.voices_dir / f"{tts['voice']}.onnx"
    if not voice.is_file():
        raise FileNotFoundError(voice)
    script_path = directory / "script.txt"
    audio_path = directory / "narration.wav"
    script_text = scene["script"].strip() + "\n"
    if (
        script_path.is_file()
        and script_path.read_text(encoding="utf-8") == script_text
        and audio_path.is_file()
        and audio_path.stat().st_size > 44
    ):
        try:
            duration = audio_duration(args.ffprobe, audio_path)
            if duration > 0:
                print(
                    f"[tts reuse] scene {int(scene['scene']):03d} "
                    f"{duration:.2f}s",
                    flush=True,
                )
                return duration
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    script_path.write_text(script_text, encoding="utf-8")
    run(
        [
            str(args.piper),
            "-m",
            str(voice),
            "-f",
            str(audio_path),
            "--length-scale",
            str(tts["length_scale"]),
            "--sentence-silence",
            "0.2",
        ],
        input=scene["script"].strip() + "\n",
        text=True,
    )
    return audio_duration(args.ffprobe, audio_path)


def deterministic_variants(record: dict) -> list[dict]:
    artifacts = record["artifacts"]
    if not record["uses_moonstar_accessories"]:
        key = "final" if "final" in artifacts else "base"
        return [
            {
                "variant": key,
                "path": artifacts[key],
                "deterministic_accepted": bool(record["validation"]["accepted"]),
                "deterministic_validation": record["validation"],
            }
        ]
    variants = []
    for key in ("wand_only", "grip_repaired"):
        if key not in artifacts:
            continue
        validation = record.get("variant_validation", {}).get(key, {})
        variants.append(
            {
                "variant": key,
                "path": artifacts[key],
                "deterministic_accepted": bool(validation.get("accepted")),
                "deterministic_validation": validation,
            }
        )
    if not variants and "base" in artifacts:
        validation = {
            "stage": "base_fallback",
            "accepted": False,
            "score": 0.0,
            "failures": ["no refined accessory variant passed preflight"],
            "advisories": ["using the base candidate as a scene-level fallback"],
            "metrics": {},
        }
        variants.append(
            {
                "variant": "base_fallback",
                "path": artifacts["base"],
                "deterministic_accepted": False,
                "deterministic_validation": validation,
            }
        )
    return variants


def selection_input_signature(run_root: Path, records: list[dict]) -> str:
    inputs = []
    for record in sorted(records, key=lambda item: item["candidate_index"]):
        variants = []
        for variant in deterministic_variants(record):
            source = run_root / variant["path"]
            state = None
            if source.is_file():
                stat = source.stat()
                state = [stat.st_size, stat.st_mtime_ns]
            variants.append(
                {
                    "variant": variant["variant"],
                    "path": variant["path"],
                    "deterministic_accepted": variant["deterministic_accepted"],
                    "state": state,
                }
            )
        inputs.append(
            {
                "id": record["id"],
                "candidate_index": record["candidate_index"],
                "variants": variants,
            }
        )
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def compatible_selection_report(
    report_path: Path,
    input_signature: str,
) -> dict | None:
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("engine_version") != QWEN_ENGINE_VERSION
            or report.get("input_signature") != input_signature
            or not report.get("selected", {}).get("id")
            or not report.get("candidate_winners")
        ):
            return None
    except (OSError, ValueError, KeyError):
        return None
    return report


def reusable_selection(
    report_path: Path,
    selected_path: Path,
    input_signature: str,
) -> dict | None:
    report = compatible_selection_report(report_path, input_signature)
    if report is None or not selected_path.is_file():
        return None
    criterion_runs = report.get("criterion_runs")
    if not isinstance(criterion_runs, dict) or not criterion_runs:
        return None
    expected_ids = {
        str(item.get("label", "")).upper()
        for item in report.get("candidate_winners", [])
    }
    if any(
        criterion_run_failed(criterion_run, expected_ids)
        for criterion_run in criterion_runs.values()
    ):
        return None
    try:
        with Image.open(selected_path) as image:
            image.verify()
    except (OSError, ValueError):
        return None
    return report


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def make_scene_contact(
    evaluations: list[dict], selected_id: str, destination: Path
) -> None:
    columns, cell = 4, (270, 410)
    rows = max(1, (len(evaluations) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(evaluations):
        with Image.open(item["local_path"]) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (258, 348))
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        sheet.paste(image, (x + (cell[0] - image.width) // 2, y))
        accepted = item.get("accepted", False)
        selected = item["id"] == selected_id
        color = "#58a6ff" if selected else ("#7ee787" if accepted else "#ff7b72")
        status = "SELECTED" if selected else ("PASS" if accepted else "REJECT")
        draw.text(
            (x + 5, y + 352),
            f"{item['label']} {item['variant']}",
            fill="white",
        )
        draw.text(
            (x + 5, y + 371),
            f"{status} score={item.get('overall_score', 0):.0f}",
            fill=color,
        )
        draw.text((x + 5, y + 390), item.get("notes", "")[:38], fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=94)


def rank_scene(
    args: argparse.Namespace,
    run_root: Path,
    records: list[dict],
    scene_dir: Path,
) -> dict:
    if not records:
        raise RuntimeError("scene has no generated records")
    report_path = scene_dir / "qwen_selection.json"
    input_signature = selection_input_signature(run_root, records)
    archived_report = compatible_selection_report(
        report_path, input_signature
    )
    reused_report = reusable_selection(
        report_path,
        scene_dir / "selected.png",
        input_signature,
    )
    if reused_report is not None:
        print(
            f"[qwen reuse] scene {records[0]['scene_number']:03d} "
            f"selected {reused_report['selected']['id']}",
            flush=True,
        )
        return reused_report
    if archived_report is not None:
        backup_path = scene_dir / "qwen_selection_pre_json_repair.json"
        if not backup_path.exists():
            shutil.copy2(report_path, backup_path)
    v1_report_path = scene_dir / "qwen_selection_v1.json"
    if report_path.is_file() and not v1_report_path.exists():
        shutil.copy2(report_path, v1_report_path)
    candidates_dir = scene_dir / "candidates"
    accepted_dir = scene_dir / "accepted"
    shutil.rmtree(candidates_dir, ignore_errors=True)
    shutil.rmtree(accepted_dir, ignore_errors=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    representatives = []
    for record in sorted(records, key=lambda item: item["candidate_index"]):
        variants = [
            item
            for item in deterministic_variants(record)
            if (run_root / item["path"]).is_file()
        ]
        if not variants and (run_root / record["artifacts"]["base"]).is_file():
            variants = [
                {
                    "variant": "base_fallback",
                    "path": record["artifacts"]["base"],
                    "deterministic_accepted": False,
                    "deterministic_validation": {
                        "stage": "base_fallback",
                        "accepted": False,
                        "score": 0.0,
                        "failures": ["refined artifact is missing"],
                        "advisories": ["using the available base image"],
                        "metrics": {},
                    },
                }
            ]
        if not variants:
            continue
        viable = [item for item in variants if item["deterministic_accepted"]]
        representative = max(
            viable or variants,
            key=lambda item: (
                item["deterministic_accepted"],
                item["variant"] == "grip_repaired",
            ),
        )
        source = (run_root / representative["path"]).resolve()
        local = (
            candidates_dir
            / f"candidate_{record['candidate_index']:02d}__"
            f"{representative['variant']}.png"
        )
        copy_file(source, local)
        representatives.append(
            {
                "label": f"C{record['candidate_index']:02d}",
                "id": record["id"],
                "scene_number": record["scene_number"],
                "candidate_index": record["candidate_index"],
                "variant": representative["variant"],
                "source_path": source,
                "local_path": str(local),
                "deterministic_accepted": representative[
                    "deterministic_accepted"
                ],
                "deterministic_validation": representative[
                    "deterministic_validation"
                ],
                "record": record,
            }
        )
    if not representatives:
        raise RuntimeError(f"scene {records[0]['scene_number']} has no generated images")

    full_sheet = scene_dir / "qwen_full_comparison.jpg"
    grip_sheet = scene_dir / "qwen_grip_comparison.jpg"
    make_qwen_sheet(
        representatives, full_sheet, run_root=run_root, grip_details=False
    )
    make_qwen_sheet(
        representatives, grip_sheet, run_root=run_root, grip_details=True
    )
    reference_paths = [
        Path(reference)
        for lora in records[0].get("base_loras", [])
        for reference in lora.get("validation_references", [])
        if Path(reference).is_file()
    ]
    reference_sheet = None
    if reference_paths:
        reference_sheet = scene_dir / "qwen_identity_references.jpg"
        make_reference_sheet(reference_paths, reference_sheet)
    print(
        f"[qwen] scene {records[0]['scene_number']:03d} "
        f"comparing {len(representatives)} candidates with isolated criteria",
        flush=True,
    )
    reused_criterion_runs = None
    rerun_criteria: set[str] = set()
    if args.reuse_criteria_report_name:
        reuse_path = scene_dir / args.reuse_criteria_report_name
        reused_criterion_runs = json.loads(
            reuse_path.read_text(encoding="utf-8")
        ).get("criterion_runs", {})
        rerun_criteria = {
            item.strip()
            for item in args.rerun_criteria.split(",")
            if item.strip()
        }
    elif archived_report is not None:
        reused_criterion_runs = archived_report.get("criterion_runs", {})
        expected_ids = {item["label"] for item in representatives}
        rerun_criteria = {
            key
            for key, criterion_run in reused_criterion_runs.items()
            if criterion_run_failed(criterion_run, expected_ids)
        }
        if rerun_criteria:
            print(
                "[qwen repair] scene "
                f"{records[0]['scene_number']:03d} rerunning only: "
                + ", ".join(sorted(rerun_criteria)),
                flush=True,
            )
    try:
        semantic = judge_scene_multipass(
            args.base_url,
            run_root,
            records[0],
            representatives,
            full_sheet,
            grip_sheet,
            reference_sheet,
            reused_criterion_runs,
            rerun_criteria,
        )
    except Exception as exc:
        print(f"WARNING: Qwen scene ranking failed: {exc}", flush=True)
        semantic = {
            "engine_version": QWEN_ENGINE_VERSION,
            "selected_id": "",
            "criterion_runs": {},
            "candidates": {
                item["label"]: {
                    **{criterion["key"]: False for criterion in QWEN_CRITERIA},
                    "overall_score": 0.0,
                    "accepted": False,
                    "failed_checks": ["qwen_error"],
                    "notes": f"{type(exc).__name__}: {exc}",
                }
                for item in representatives
            },
        }

    evaluations = []
    for representative in representatives:
        item = {key: value for key, value in representative.items() if key != "record"}
        item["source_path"] = str(item["source_path"])
        semantic_candidate = dict(
            semantic["candidates"][representative["label"]]
        )
        item["qwen_label"] = semantic_candidate.pop(
            "id", representative["label"]
        )
        item.update(semantic_candidate)
        if not representative["deterministic_accepted"]:
            item["accepted"] = False
            item["failed_checks"] = list(
                dict.fromkeys(["deterministic_geometry", *item["failed_checks"]])
            )
            item["notes"] = "Rejected by deterministic geometry checks."
        evaluations.append(item)

    accepted = [item for item in evaluations if item["accepted"]]
    for item in accepted:
        copy_file(
            Path(item["local_path"]),
            accepted_dir
            / f"candidate_{item['candidate_index']:02d}__{item['variant']}.png",
        )
    requested = next(
        (
            item
            for item in accepted
            if item["label"] == semantic.get("selected_id")
        ),
        None,
    )
    selected = requested or max(
        accepted or evaluations,
        key=lambda item: (
            item["accepted"],
            item["deterministic_accepted"],
            item["overall_score"],
            item["variant"] == "grip_repaired",
        ),
    )
    selected["fallback_selection"] = not bool(accepted)
    copy_file(Path(selected["local_path"]), scene_dir / "selected.png")
    make_scene_contact(evaluations, selected["id"], scene_dir / "qwen_contact.jpg")
    report = {
        "engine_version": semantic["engine_version"],
        "input_signature": input_signature,
        "scene_number": records[0]["scene_number"],
        "candidate_count": len(records),
        "evaluated_variant_count": len(evaluations),
        "accepted_candidate_count": len(accepted),
        "qwen_requested_selected_id": semantic.get("selected_id"),
        "criterion_runs": semantic.get("criterion_runs", {}),
        "selected": selected,
        "candidate_winners": evaluations,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def render_scene(
    args: argparse,
    config: dict,
    scene_dir: Path,
    narration_seconds: float,
) -> float:
    settings = config["settings"]
    pre = float(settings["pre_roll_seconds"])
    post = float(settings["post_roll_seconds"])
    duration = narration_seconds + pre + post
    delay_ms = round(pre * 1000)
    destination = scene_dir / "scene.mp4"
    video_filter = (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
    )
    run(
        [
            str(args.ffmpeg),
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(settings["fps"]),
            "-i",
            str(scene_dir / "selected.png"),
            "-i",
            str(scene_dir / "narration.wav"),
            "-filter_complex",
            (
                f"[0:v]{video_filter}[v];"
                f"[1:a]adelay={delay_ms}:all=1,apad=pad_dur={post}[a]"
            ),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{duration:.3f}",
            "-r",
            str(settings["fps"]),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    return audio_duration(args.ffprobe, destination)


def concat_scenes(args: argparse.Namespace, story_dir: Path, scene_dirs: list[Path]) -> Path:
    concat_path = story_dir / "scene_order.txt"
    lines = []
    for directory in scene_dirs:
        escaped = str((directory / "scene.mp4").resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    destination = story_dir / "storybook.mp4"
    run(
        [
            str(args.ffmpeg),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    return destination


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    compiled_path = args.compiled_story or run_root / "compiled_story.json"
    config = json.loads(compiled_path.read_text(encoding="utf-8"))
    story_dir = run_root / args.story_dir_name
    story_dir.mkdir(parents=True, exist_ok=True)
    compiled_scenes = {
        int(scene["scene"]): scene for scene in config["scenes"]
    }
    records_by_scene: dict[int, list[dict]] = {}
    for record in summary["records"]:
        scene_number = int(record["scene_number"])
        compiled_scene = compiled_scenes[scene_number]
        record["base_loras"] = compiled_scene["base_loras"]
        record["extra_loras"] = compiled_scene["extra_loras"]
        records_by_scene.setdefault(scene_number, []).append(record)

    scene_states = []
    for scene in config["scenes"]:
        number = int(scene["scene"])
        scene_dir = story_dir / f"scene_{number:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        narration_seconds = narrate(args, config, scene, scene_dir)
        scene_states.append(
            {
                "scene": scene,
                "directory": scene_dir,
                "narration_seconds": narration_seconds,
            }
        )

    started_qwen = ensure_qwen(args)
    try:
        for state in scene_states:
            number = int(state["scene"]["scene"])
            state["selection"] = rank_scene(
                args,
                run_root,
                records_by_scene.get(number, []),
                state["directory"],
            )
    finally:
        if started_qwen and not args.keep_server:
            run([str(args.qwen_stop)])

    manifest_scenes = []
    for state in scene_states:
        rendered_seconds = render_scene(
            args,
            config,
            state["directory"],
            state["narration_seconds"],
        )
        manifest_scenes.append(
            {
                "scene": state["scene"]["scene"],
                "directory": str(state["directory"]),
                "script": state["scene"]["script"],
                "narration_seconds": state["narration_seconds"],
                "rendered_seconds": rendered_seconds,
                "pre_roll_seconds": config["settings"]["pre_roll_seconds"],
                "post_roll_seconds": config["settings"]["post_roll_seconds"],
                "accepted_candidate_count": state["selection"][
                    "accepted_candidate_count"
                ],
                "selected_id": state["selection"]["selected"]["id"],
                "selected_variant": state["selection"]["selected"]["variant"],
                "fallback_selection": state["selection"]["selected"][
                    "fallback_selection"
                ],
            }
        )
    video = concat_scenes(
        args, story_dir, [state["directory"] for state in scene_states]
    )
    manifest = {
        "version": "storybook-render-qwen35-v1",
        "title": config["title"],
        "source": config["source"],
        "video": str(video),
        "video_seconds": audio_duration(args.ffprobe, video),
        "scenes": manifest_scenes,
    }
    (story_dir / "story_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

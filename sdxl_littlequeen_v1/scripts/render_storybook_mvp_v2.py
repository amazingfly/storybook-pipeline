#!/usr/bin/env python3
"""Prompt-aware V2 ranking, narration, and assembly for generated story images."""

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
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEMMA_START = ROOT / "storybook_mvp_v1" / "start_gemma_storybook.sh"
DEFAULT_GEMMA_STOP = ROOT / "storybook_mvp_v1" / "stop_gemma_storybook.sh"
HARD_CHECKS = (
    "protagonist_uniqueness",
    "coherent_anatomy",
    "character_consistency",
    "required_subjects_and_counts",
    "action_and_setting_match",
    "visual_coherence",
    "exactly_one_requested_wand",
    "wand_readable",
    "wand_held_by_hand",
    "complete_framing",
    "no_text_or_watermark",
)
GEMMA_ENGINE_VERSION = "storybook-gemma-contract-picker-v4"
SCENE_CONTRACT_VERSION = "storybook-scene-contract-v1"
STORY_DIR_NAME = "story_v2"
MINIMUM_ACCEPTED_SCORE = 68.0
GEMMA_CRITERIA = (
    {
        "key": "protagonist_uniqueness",
        "weight": 13,
        "gate": "hard",
        "source": "full",
        "requires": "character",
        "question": (
            "Is there exactly one Little Queen, with one head and no duplicate, "
            "reflection, extra face, or background copy of her? Other people and "
            "animals explicitly requested by the scene are allowed and must not "
            "cause a failure."
        ),
    },
    {
        "key": "coherent_anatomy",
        "weight": 15,
        "gate": "hard",
        "source": "full",
        "requires": "living_subject",
        "question": (
            "Do the important visible people and animals have coherent anatomy? "
            "Inspect faces, bodies, limbs, and hands that are large enough to judge. "
            "Fail obvious extra, fused, missing, disconnected, or severely malformed "
            "parts. Do not fail harmless storybook stylization or tiny background "
            "figures."
        ),
    },
    {
        "key": "character_consistency",
        "weight": 14,
        "gate": "hard",
        "source": "identity",
        "requires": "identity",
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
        "gate": "soft",
        "source": "full",
        "requires": "accessories",
        "question": (
            "Are visible crown, jewelry, and accessories clean and coherent? Prefer "
            "one recognizable gold crown and matching jewelry when the Little Queen "
            "is present. Mark false for a missing signature crown or conspicuously "
            "broken, fused, or duplicated regalia, but this is a quality flag rather "
            "than an automatic rejection."
        ),
    },
    {
        "key": "exactly_one_requested_wand",
        "weight": 12,
        "gate": "hard",
        "source": "base_prop",
        "requires": "wand",
        "question": (
            "Before compositing, are any reported long shafts far enough from the "
            "official wand axis that the overlay would leave a second prop visible?"
        ),
    },
    {
        "key": "wand_readable",
        "weight": 8,
        "gate": "hard",
        "source": "full",
        "requires": "wand",
        "question": (
            "When the scene explicitly requests a wand, is there exactly one readable "
            "Moonstar wand with one coherent gold shaft and deliberate decorative "
            "head? Fail missing, broken, doubled, melted, or unrecognizable geometry."
        ),
    },
    {
        "key": "wand_held_by_hand",
        "weight": 15,
        "gate": "hard",
        "source": "base_hands",
        "requires": "wand",
        "question": (
            "When the scene explicitly requests a wand, is its recorded grip anchor "
            "spatially close to a hand that Gemma independently localized?"
        ),
    },
    {
        "key": "complete_framing",
        "weight": 5,
        "gate": "hard",
        "source": "full",
        "requires": "character",
        "question": (
            "Is the illustration framed cleanly for a storybook page? The complete "
            "Little Queen head and any crown must be visible unless the requested "
            "composition explicitly describes a close crop. Relevant hands or held "
            "objects must be visible enough to understand the requested action."
        ),
    },
    {
        "key": "required_subjects_and_counts",
        "weight": 18,
        "gate": "hard",
        "source": "full",
        "question": (
            "Does the image contain every explicitly required named subject and object "
            "from the scene contract, in the requested count? Fail missing named "
            "subjects, duplicate Little Queens, or a clearly wrong replacement "
            "species/object. Do not require unmentioned decorative details."
        ),
    },
    {
        "key": "action_and_setting_match",
        "weight": 13,
        "gate": "hard",
        "source": "full",
        "question": (
            "Does the image clearly communicate the requested main action and setting "
            "from the scene contract? Fail when it only matches the mood or color, or "
            "when an extra focal prop contradicts the requested action."
        ),
    },
    {
        "key": "visual_coherence",
        "weight": 10,
        "gate": "hard",
        "source": "full",
        "question": (
            "Is this one coherent illustration rather than a malformed collage? Fail "
            "major merged subjects, impossible foreground geometry, repeated fragments, "
            "or conspicuous generation damage. Ignore small decorative imperfections."
        ),
    },
    {
        "key": "no_text_or_watermark",
        "weight": 3,
        "gate": "hard",
        "source": "full",
        "question": (
            "Is the artwork free of visible writing, captions, signatures, watermarks, "
            "panel borders, and image-grid artifacts? Incidental book-page marks that "
            "are not readable words are allowed."
        ),
    },
    {
        "key": "composition_and_story_clarity",
        "weight": 10,
        "gate": "soft",
        "source": "full",
        "question": (
            "Compared with the other candidates, is this a strong storybook composition "
            "with a clear focal subject, readable expression or action, balanced use of "
            "the frame, and appealing detail? Reserve quality 4 for genuinely strong "
            "options; minor aesthetic weakness is a quality flag, not rejection."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--compiled-story", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--piper", type=Path, default=DEFAULT_PIPER)
    parser.add_argument("--voices-dir", type=Path, default=DEFAULT_VOICES)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--gemma-start", type=Path, default=DEFAULT_GEMMA_START)
    parser.add_argument("--gemma-stop", type=Path, default=DEFAULT_GEMMA_STOP)
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


def ensure_gemma(args: argparse.Namespace) -> bool:
    if healthy(args.base_url):
        return False
    run([str(args.gemma_start)])
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if healthy(args.base_url):
            return True
        time.sleep(1)
    raise RuntimeError("Gemma vision server did not become ready")


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


def parse_gemma_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise RuntimeError(f"Gemma returned no JSON object: {text!r}")
    result = json.loads(match.group(0))
    for key in HARD_CHECKS:
        if not isinstance(result.get(key), bool):
            raise RuntimeError(f"Gemma response is missing boolean {key}: {result}")
    score = result.get("overall_score")
    if not isinstance(score, (int, float)):
        raise RuntimeError(f"Gemma response is missing overall_score: {result}")
    result["overall_score"] = max(0, min(100, round(float(score), 1)))
    result["notes"] = str(result.get("notes", "")).strip()
    result["failed_checks"] = [key for key in HARD_CHECKS if not result[key]]
    result["accepted"] = not result["failed_checks"] and result["overall_score"] >= 70
    return result


def parse_scene_gemma_json(text: str, expected_ids: set[str]) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise RuntimeError(f"Gemma returned no JSON object: {text!r}")
    result = json.loads(match.group(0))
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError(f"Gemma response has no candidates array: {result}")
    parsed = {}
    for item in candidates:
        candidate_id = str(item.get("id", "")).upper()
        if candidate_id not in expected_ids or candidate_id in parsed:
            continue
        for key in HARD_CHECKS:
            if not isinstance(item.get(key), bool):
                raise RuntimeError(
                    f"Gemma candidate {candidate_id} is missing boolean {key}"
                )
        score = item.get("overall_score")
        if not isinstance(score, (int, float)):
            raise RuntimeError(
                f"Gemma candidate {candidate_id} is missing overall_score"
            )
        item["id"] = candidate_id
        item["overall_score"] = max(0, min(100, round(float(score), 1)))
        item["notes"] = str(item.get("notes", "")).strip()
        item["failed_checks"] = [key for key in HARD_CHECKS if not item[key]]
        item["accepted"] = not item["failed_checks"] and item["overall_score"] >= 70
        parsed[candidate_id] = item
    missing = expected_ids - set(parsed)
    if missing:
        raise RuntimeError(f"Gemma omitted candidates: {sorted(missing)}")
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
    return parse_gemma_json(data["choices"][0]["message"]["content"])


def make_gemma_sheet(
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
    return parse_scene_gemma_json(
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


def has_character_identity(record: dict) -> bool:
    return any(
        item.get("role") == "identity" for item in record.get("base_loras", [])
    )


def has_accessory_binding(record: dict) -> bool:
    return any(
        item.get("role") in {"accessories", "crown", "jewelry", "wand"}
        for item in record.get("extra_loras", [])
    )


def explicit_wand_terms(record: dict) -> list[str]:
    text = f"{record.get('prompt', '')} {record.get('script', '')}".lower()
    terms = []
    for term in ("wand", "staff", "scepter", "sceptre"):
        if re.search(rf"\b{term}s?\b", text):
            terms.append(term)
    return terms


def scene_contract_signature(record: dict) -> str:
    payload = {
        "version": SCENE_CONTRACT_VERSION,
        "scene_number": record.get("scene_number"),
        "prompt": record.get("prompt", ""),
        "script": record.get("script", ""),
        "base_loras": [
            {
                "name": item.get("name"),
                "character": item.get("character"),
                "role": item.get("role"),
            }
            for item in record.get("base_loras", [])
        ],
        "extra_loras": [
            {
                "name": item.get("name"),
                "character": item.get("character"),
                "role": item.get("role"),
            }
            for item in record.get("extra_loras", [])
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def scene_contract_prompt(record: dict) -> str:
    return f"""
Extract a literal visual checklist for one children's storybook illustration.
The generation prompt is authoritative. The narration may clarify the same beat,
but do not invent an object, person, costume, pose, or count that neither text states.

Generation prompt:
{record.get('prompt', '')}

Narration:
{record.get('script', '')}

Rules:
- List every explicitly named foreground person, animal species, and important object.
- Preserve exact numeric counts. Use null when no exact count is stated.
- "The Little Queen" means exactly one Little Queen.
- Keep separately named animals separate. For example, owl, fox, deer, and bear are
  four distinct required subjects, not the generic item "animal friends".
- Plural groups such as villagers or children have count null unless a number is stated.
- Do not infer a wand, crown, animal, or person from genre, royalty, or magic.
- Actions and settings must be short, directly visible descriptions.
- allowed_additional_subjects should name groups the prompt permits beyond the
  protagonist, such as villagers, children, shopkeepers, or background crowds.

Return only JSON:
{{
  "required_subjects": [
    {{"name": "Little Queen", "count": 1}}
  ],
  "required_objects": [
    {{"name": "basket", "count": 1}}
  ],
  "required_actions": ["visible main action"],
  "required_settings": ["visible location"],
  "allowed_additional_subjects": ["explicitly allowed group"],
  "forbidden_or_conflicting": ["only an explicitly stated exclusion"]
}}
""".strip()


def normalized_contract_items(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    normalized = []
    for raw in value:
        if isinstance(raw, str):
            name, count = raw.strip(), None
        elif isinstance(raw, dict):
            name = str(raw.get("name", "")).strip()
            count = raw.get("count")
            if isinstance(count, bool) or not isinstance(count, (int, float)):
                count = None
            elif count < 0:
                count = None
            else:
                count = int(count)
        else:
            continue
        if name:
            normalized.append({"name": name, "count": count})
    return normalized


def normalized_contract_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        text
        for item in value
        if (text := str(item).strip())
    ]


def fallback_scene_contract(record: dict) -> dict:
    prompt = record.get("prompt", "")
    character_required = has_character_identity(record) or bool(
        re.search(r"\blittle queen\b", prompt, flags=re.IGNORECASE)
    )
    required_subjects = (
        [{"name": "Little Queen", "count": 1}] if character_required else []
    )
    return {
        "contract_version": SCENE_CONTRACT_VERSION,
        "signature": scene_contract_signature(record),
        "source": "deterministic_fallback",
        "required_subjects": required_subjects,
        "required_objects": [],
        "required_actions": [prompt] if prompt else [],
        "required_settings": [],
        "allowed_additional_subjects": [],
        "forbidden_or_conflicting": [],
        "character_required": character_required,
        "identity_required": has_character_identity(record),
        "accessories_expected": has_accessory_binding(record),
        "wand_required": bool(explicit_wand_terms(record)),
        "explicit_wand_terms": explicit_wand_terms(record),
        "living_subject_required": character_required,
        "planner_errors": [],
    }


def build_scene_contract(
    base_url: str,
    record: dict,
    scene_dir: Path,
) -> dict:
    destination = scene_dir / "validation_contract_v2.json"
    signature = scene_contract_signature(record)
    if destination.is_file():
        try:
            cached = json.loads(destination.read_text(encoding="utf-8"))
            if (
                cached.get("contract_version") == SCENE_CONTRACT_VERSION
                and cached.get("signature") == signature
            ):
                return cached
        except (OSError, ValueError):
            pass

    contract = fallback_scene_contract(record)
    errors = []
    raw_response = ""
    for _ in range(2):
        try:
            raw_response = request_gemma_json(
                base_url,
                scene_contract_prompt(record),
                [],
            )
            document = extract_json_object(raw_response)
            contract.update(
                {
                    "source": "gemma_text_planner",
                    "required_subjects": normalized_contract_items(
                        document.get("required_subjects")
                    ),
                    "required_objects": normalized_contract_items(
                        document.get("required_objects")
                    ),
                    "required_actions": normalized_contract_strings(
                        document.get("required_actions")
                    ),
                    "required_settings": normalized_contract_strings(
                        document.get("required_settings")
                    ),
                    "allowed_additional_subjects": normalized_contract_strings(
                        document.get("allowed_additional_subjects")
                    ),
                    "forbidden_or_conflicting": normalized_contract_strings(
                        document.get("forbidden_or_conflicting")
                    ),
                    "raw_response": raw_response,
                }
            )
            break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if contract["character_required"] and not any(
        "little queen" in item["name"].lower()
        for item in contract["required_subjects"]
    ):
        contract["required_subjects"].insert(
            0, {"name": "Little Queen", "count": 1}
        )
    contract["living_subject_required"] = bool(contract["required_subjects"])
    contract["planner_errors"] = errors
    destination.write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return contract


def criterion_applies(criterion: dict, contract: dict) -> bool:
    requirement = criterion.get("requires")
    if requirement is None:
        return True
    return {
        "character": contract["character_required"],
        "identity": contract["identity_required"],
        "accessories": contract["accessories_expected"],
        "wand": contract["wand_required"],
        "living_subject": contract["living_subject_required"],
    }.get(requirement, False)


def criterion_prompt(
    criterion: dict,
    record: dict,
    candidate_ids: list[str],
    *,
    contract: dict,
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
    if criterion["key"] in {
        "protagonist_uniqueness",
        "required_subjects_and_counts",
        "action_and_setting_match",
        "complete_framing",
        "visual_coherence",
        "composition_and_story_clarity",
    }:
        scene_context = (
            f"\nRequested scene:\n{record['prompt']}\n"
            f"Narration context:\n{record.get('script', '')}\n"
            "Literal validation contract:\n"
            + json.dumps(
                {
                    "required_subjects": contract["required_subjects"],
                    "required_objects": contract["required_objects"],
                    "required_actions": contract["required_actions"],
                    "required_settings": contract["required_settings"],
                    "allowed_additional_subjects": contract[
                        "allowed_additional_subjects"
                    ],
                    "forbidden_or_conflicting": contract[
                        "forbidden_or_conflicting"
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    retry_note = (
        "\nYour previous response did not match the schema. Follow it literally."
        if retry
        else ""
    )
    return f"""
You are performing one narrow visual inspection for children's storybook art.
Judge ONLY the criterion named {criterion['key']}. Do not reward or penalize any
other quality. Inspect every candidate independently before comparing certainty.

Question:
{criterion['question']}
{scene_context}
{image_explanation}
Required candidate IDs: {", ".join(candidate_ids)}

Use this fixed evidence scale:
4 = definite clean pass
3 = pass, with a minor harmless imperfection
2 = uncertain or borderline; this is a fail
1 = clear fail
0 = severe or unmistakable fail

Return every required ID exactly once. "pass" must be true only for quality 3 or 4.
For required_subjects_and_counts, explicitly name the missing, duplicated, or wrong
subject/object. For action_and_setting_match, name the missing action or setting.
Keep each reason factual and criterion-specific, in 18 words or fewer.
Return only JSON in this exact shape:
{{
  "criterion": "{criterion['key']}",
  "candidates": [
    {{"id": "C01", "pass": true, "quality": 4, "reason": "brief visible evidence"}}
  ]
}}
{retry_note}
""".strip()


def request_gemma_json(
    base_url: str,
    prompt: str,
    image_paths: list[
        tuple[Path, tuple[int, int]]
        | tuple[Path, tuple[int, int], tuple[int, int, int, int] | None]
    ],
) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
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
    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 1100,
        "reasoning_budget_tokens": 32,
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
    return data["choices"][0]["message"]["content"]


def parse_criterion_json(
    text: str, criterion_key: str, expected_ids: set[str]
) -> dict[str, dict]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise RuntimeError(f"Gemma returned no JSON object: {text!r}")
    document = json.loads(match.group(0))
    if str(document.get("criterion", "")).strip() != criterion_key:
        raise RuntimeError(
            f"Gemma returned criterion {document.get('criterion')!r}, "
            f"expected {criterion_key!r}"
        )
    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("Gemma criterion response has no candidates array")
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
        raise RuntimeError(f"Gemma omitted candidates: {sorted(missing)}")
    return parsed


def aggregate_criterion_results(
    candidate_ids: list[str], criterion_runs: dict[str, dict]
) -> dict[str, dict]:
    aggregated: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        checks = {}
        hard_failures = []
        quality_flags = []
        weighted_points = 0.0
        available_weight = 0.0
        for criterion in GEMMA_CRITERIA:
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
                if criterion["gate"] == "hard":
                    hard_failures.append(key)
                else:
                    quality_flags.append(key)
        score = 100.0 * weighted_points / available_weight if available_weight else 0
        score = round(score, 1)
        if score < MINIMUM_ACCEPTED_SCORE:
            hard_failures.append("overall_quality")
        failure_notes = [
            (
                f"{key}: {checks[key]['reason']}"
                if key in checks
                else f"overall_quality: score {score:.1f} is below "
                f"{MINIMUM_ACCEPTED_SCORE:.1f}"
            )
            for key in hard_failures[:2]
        ]
        flag_notes = [
            f"{key}: {checks[key]['reason']}" for key in quality_flags[:1]
        ]
        accepted = not hard_failures
        aggregated[candidate_id] = {
            **{key: value["pass"] for key, value in checks.items()},
            "criteria": checks,
            "overall_score": score,
            "failed_checks": hard_failures,
            "quality_flags": quality_flags,
            "accepted": accepted,
            "image_acceptable": accepted,
            "notes": (
                "; ".join(failure_notes)
                if failure_notes
                else (
                    "Accepted with quality flag: " + "; ".join(flag_notes)
                    if flag_notes
                    else "Passed all applicable hard and soft criteria."
                )
            ),
        }
    return aggregated


def extract_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise RuntimeError(f"Gemma returned no JSON object: {text!r}")
    document = json.loads(match.group(0))
    if not isinstance(document, dict):
        raise RuntimeError("Gemma JSON response is not an object")
    return document


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
            f"[gemma evidence] {criterion['key']} {index}/{len(candidates)} {label}",
            flush=True,
        )
        record = candidate["record"]
        base_path = run_root / record["artifacts"]["base"]
        metadata = json.loads(
            (run_root / record["artifacts"]["metadata"]).read_text(
                encoding="utf-8"
            )
        )
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
                raw_response = request_gemma_json(
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
            f"[gemma evidence] {criterion['key']} {index}/{len(candidates)} {label}",
            flush=True,
        )
        record = candidate["record"]
        base_path = run_root / record["artifacts"]["base"]
        metadata = json.loads(
            (run_root / record["artifacts"]["metadata"]).read_text(
                encoding="utf-8"
            )
        )
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
                raw_response = request_gemma_json(
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
                else "Gemma localized no visible hands."
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


def judge_scene_multipass(
    base_url: str,
    run_root: Path,
    record: dict,
    contract: dict,
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
    applicable = [
        item
        for item in GEMMA_CRITERIA
        if criterion_applies(item, contract)
    ]
    for index, criterion in enumerate(applicable, 1):
        print(
            f"[gemma criterion] scene {record['scene_number']:03d} "
            f"{index}/{len(applicable)} {criterion['key']}",
            flush=True,
        )
        if (
            reused_criterion_runs
            and criterion["key"] in reused_criterion_runs
            and criterion["key"] not in (rerun_criteria or set())
        ):
            print(
                f"[gemma criterion] reusing archived {criterion['key']}",
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
        for attempt in range(2):
            try:
                raw_response = request_gemma_json(
                    base_url,
                    criterion_prompt(
                        criterion,
                        record,
                        candidate_ids,
                        contract=contract,
                        has_identity_references=reference_sheet is not None,
                        retry=bool(attempt),
                    ),
                    image_paths,
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
            "gate": criterion["gate"],
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
        "engine_version": GEMMA_ENGINE_VERSION,
        "contract": contract,
        "selected_id": selected_id,
        "criterion_runs": criterion_runs,
        "candidates": aggregated,
    }


def final_picker_prompt(
    record: dict,
    contract: dict,
    shortlist: list[dict],
) -> str:
    candidate_summary = [
        {
            "id": item["label"],
            "validation_score": item["overall_score"],
            "quality_flags": item.get("quality_flags", []),
        }
        for item in shortlist
    ]
    return f"""
Choose the strongest finished illustration for this story scene. Filtering has
already handled hard defects. This is a comparative editorial choice, not another
pass/fail audit.

Scene prompt:
{record['prompt']}

Narration:
{record.get('script', '')}

Required visual contract:
{json.dumps({
    "required_subjects": contract["required_subjects"],
    "required_objects": contract["required_objects"],
    "required_actions": contract["required_actions"],
    "required_settings": contract["required_settings"],
}, indent=2)}

The images are supplied in this exact order:
{", ".join(item["label"] for item in shortlist)}

Candidate validation context:
{json.dumps(candidate_summary, indent=2)}

Rank by these priorities in order:
1. The requested story beat, named subjects, count, and action read immediately.
2. The Little Queen identity and facial expression are convincing when she appears.
3. The focal hierarchy and composition are clear at storybook-page size.
4. Anatomy, accessories, lighting, and fine detail look polished.

Do not choose by ID order. Do not prefer unnecessary equipment or extra characters.
Return only JSON:
{{
  "selected_id": "C01",
  "ranking": ["C01", "C02"],
  "reason": "specific comparative reason in 22 words or fewer"
}}
""".strip()


def final_editorial_pick(
    base_url: str,
    record: dict,
    contract: dict,
    evaluations: list[dict],
) -> dict:
    accepted = [item for item in evaluations if item["accepted"]]
    pool = accepted or evaluations
    shortlist = sorted(
        pool,
        key=lambda item: (
            item["overall_score"],
            item.get("refinement_eligible", False),
        ),
        reverse=True,
    )[:4]
    fallback = max(
        shortlist,
        key=lambda item: (
            item["overall_score"],
            item.get("refinement_eligible", False),
        ),
    )
    if len(shortlist) == 1:
        return {
            "selected_id": fallback["label"],
            "shortlist": [fallback["label"]],
            "reason": "Only one candidate remained in the editorial shortlist.",
            "raw_response": "",
            "errors": [],
        }

    errors = []
    raw_response = ""
    for _ in range(2):
        try:
            raw_response = request_gemma_json(
                base_url,
                final_picker_prompt(record, contract, shortlist),
                [
                    (Path(item["local_path"]), (1000, 1100))
                    for item in shortlist
                ],
            )
            document = extract_json_object(raw_response)
            selected_id = str(document.get("selected_id", "")).upper()
            allowed = {item["label"] for item in shortlist}
            if selected_id not in allowed:
                raise RuntimeError(
                    f"editorial picker selected unknown candidate {selected_id!r}"
                )
            ranking = [
                str(item).upper()
                for item in document.get("ranking", [])
                if str(item).upper() in allowed
            ]
            return {
                "selected_id": selected_id,
                "shortlist": [item["label"] for item in shortlist],
                "ranking": ranking,
                "reason": str(document.get("reason", "")).strip(),
                "raw_response": raw_response,
                "errors": errors,
            }
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "selected_id": fallback["label"],
        "shortlist": [item["label"] for item in shortlist],
        "reason": "Editorial comparison failed; used the highest validation score.",
        "raw_response": raw_response,
        "errors": errors,
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
                "refinement_eligible": False,
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
                "refinement_eligible": bool(validation.get("accepted")),
                "deterministic_validation": validation,
            }
        )
    if "base" in artifacts:
        validation = {
            "stage": "base_image",
            "accepted": True,
            "score": 100.0,
            "failures": [],
            "advisories": [
                "accessory refinement eligibility is evaluated separately"
            ],
            "metrics": {},
        }
        variants.append(
            {
                "variant": "base",
                "path": artifacts["base"],
                "deterministic_accepted": False,
                "refinement_eligible": False,
                "deterministic_validation": validation,
            }
        )
    return variants


def choose_representative_variant(
    variants: list[dict],
    contract: dict,
) -> dict:
    base = next((item for item in variants if item["variant"] == "base"), None)
    viable_refinements = [
        item
        for item in variants
        if item.get("refinement_eligible") and item["variant"] != "base"
    ]
    if contract["wand_required"] and viable_refinements:
        return max(
            viable_refinements,
            key=lambda item: item["variant"] == "grip_repaired",
        )
    if base is not None:
        return base
    if viable_refinements:
        return max(
            viable_refinements,
            key=lambda item: item["variant"] == "grip_repaired",
        )
    return max(
        variants,
        key=lambda item: (
            item.get("deterministic_accepted", False),
            item["variant"] == "grip_repaired",
        ),
    )


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
                    "refinement_eligible": variant.get(
                        "refinement_eligible", False
                    ),
                    "state": state,
                }
            )
        inputs.append(
            {
                "id": record["id"],
                "candidate_index": record["candidate_index"],
                "prompt": record.get("prompt", ""),
                "script": record.get("script", ""),
                "variants": variants,
            }
        )
    payload = json.dumps(
        {"engine_version": GEMMA_ENGINE_VERSION, "records": inputs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def reusable_selection(
    report_path: Path,
    selected_path: Path,
    input_signature: str,
) -> dict | None:
    if not report_path.is_file() or not selected_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("engine_version") != GEMMA_ENGINE_VERSION
            or report.get("input_signature") != input_signature
            or not report.get("selected", {}).get("id")
            or not report.get("candidate_winners")
        ):
            return None
        with Image.open(selected_path) as image:
            image.verify()
    except (OSError, ValueError, KeyError):
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
    report_path = scene_dir / "gemma_selection_v2.json"
    input_signature = selection_input_signature(run_root, records)
    reused_report = reusable_selection(
        report_path,
        scene_dir / "selected.png",
        input_signature,
    )
    if reused_report is not None:
        print(
            f"[gemma reuse] scene {records[0]['scene_number']:03d} "
            f"selected {reused_report['selected']['id']}",
            flush=True,
        )
        return reused_report
    contract = build_scene_contract(args.base_url, records[0], scene_dir)
    print(
        f"[contract] scene {records[0]['scene_number']:03d} "
        f"subjects={len(contract['required_subjects'])} "
        f"objects={len(contract['required_objects'])} "
        f"wand_required={contract['wand_required']}",
        flush=True,
    )
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
                    "variant": "base",
                    "path": record["artifacts"]["base"],
                    "deterministic_accepted": False,
                    "refinement_eligible": False,
                    "deterministic_validation": {
                        "stage": "base_image",
                        "accepted": True,
                        "score": 100.0,
                        "failures": [],
                        "advisories": [
                            "refinement eligibility is separate from image quality"
                        ],
                        "metrics": {},
                    },
                }
            ]
        if not variants:
            continue
        representative = choose_representative_variant(variants, contract)
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
                "refinement_eligible": representative.get(
                    "refinement_eligible", False
                ),
                "deterministic_validation": representative[
                    "deterministic_validation"
                ],
                "record": record,
            }
        )
    if not representatives:
        raise RuntimeError(f"scene {records[0]['scene_number']} has no generated images")

    full_sheet = scene_dir / "gemma_full_comparison.jpg"
    grip_sheet = scene_dir / "gemma_grip_comparison.jpg"
    make_gemma_sheet(
        representatives, full_sheet, run_root=run_root, grip_details=False
    )
    make_gemma_sheet(
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
        reference_sheet = scene_dir / "gemma_identity_references.jpg"
        make_reference_sheet(reference_paths, reference_sheet)
    print(
        f"[gemma] scene {records[0]['scene_number']:03d} "
        f"comparing {len(representatives)} candidates with isolated criteria",
        flush=True,
    )
    reused_criterion_runs = None
    if args.reuse_criteria_report_name:
        reuse_path = scene_dir / args.reuse_criteria_report_name
        reused_criterion_runs = json.loads(
            reuse_path.read_text(encoding="utf-8")
        ).get("criterion_runs", {})
    rerun_criteria = {
        item.strip() for item in args.rerun_criteria.split(",") if item.strip()
    }
    try:
        semantic = judge_scene_multipass(
            args.base_url,
            run_root,
            records[0],
            contract,
            representatives,
            full_sheet,
            grip_sheet,
            reference_sheet,
            reused_criterion_runs,
            rerun_criteria,
        )
    except Exception as exc:
        print(f"WARNING: Gemma scene ranking failed: {exc}", flush=True)
        semantic = {
            "engine_version": GEMMA_ENGINE_VERSION,
            "selected_id": "",
            "criterion_runs": {},
            "candidates": {
                item["label"]: {
                    **{criterion["key"]: False for criterion in GEMMA_CRITERIA},
                    "overall_score": 0.0,
                    "accepted": False,
                    "image_acceptable": False,
                    "failed_checks": ["gemma_error"],
                    "quality_flags": [],
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
        item["gemma_label"] = semantic_candidate.pop(
            "id", representative["label"]
        )
        item.update(semantic_candidate)
        item["refinement_eligibility"] = {
            "eligible": representative["refinement_eligible"],
            "stage": representative["deterministic_validation"].get("stage"),
            "failures": representative["deterministic_validation"].get(
                "failures", []
            ),
        }
        evaluations.append(item)

    accepted = [item for item in evaluations if item["accepted"]]
    for item in accepted:
        copy_file(
            Path(item["local_path"]),
            accepted_dir
            / f"candidate_{item['candidate_index']:02d}__{item['variant']}.png",
        )
    editorial = final_editorial_pick(
        args.base_url,
        records[0],
        contract,
        evaluations,
    )
    print(
        f"[editorial pick] scene {records[0]['scene_number']:03d} "
        f"selected {editorial['selected_id']} from "
        f"{','.join(editorial['shortlist'])}",
        flush=True,
    )
    requested = next(
        (
            item
            for item in (accepted or evaluations)
            if item["label"] == editorial["selected_id"]
        ),
        None,
    )
    selected = requested or max(
        accepted or evaluations,
        key=lambda item: (
            item["accepted"],
            item["overall_score"],
            item["refinement_eligible"],
            item["variant"] == "grip_repaired",
        ),
    )
    selected["fallback_selection"] = not bool(accepted)
    copy_file(Path(selected["local_path"]), scene_dir / "selected.png")
    make_scene_contact(evaluations, selected["id"], scene_dir / "gemma_contact.jpg")
    report = {
        "engine_version": semantic["engine_version"],
        "input_signature": input_signature,
        "scene_number": records[0]["scene_number"],
        "candidate_count": len(records),
        "evaluated_variant_count": len(evaluations),
        "accepted_candidate_count": len(accepted),
        "gemma_requested_selected_id": semantic.get("selected_id"),
        "scene_contract": contract,
        "editorial_picker": editorial,
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
    story_dir = run_root / STORY_DIR_NAME
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
        record["script"] = compiled_scene["script"]
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

    started_gemma = ensure_gemma(args)
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
        if started_gemma and not args.keep_server:
            run([str(args.gemma_stop)])

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
        "version": "storybook-render-v2",
        "validation_engine": GEMMA_ENGINE_VERSION,
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

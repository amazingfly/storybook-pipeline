#!/usr/bin/env python3
"""Run resumable Qwen3.5-9B storybook validation on Colab."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-9B"
WORKFLOW_VERSION = "littlequeen-storybook-qwen35-validator-v3.5"
COMPATIBLE_WORKFLOW_VERSIONS = {
    "littlequeen-storybook-qwen35-validator-v3.3",
    "littlequeen-storybook-qwen35-validator-v3.4",
    WORKFLOW_VERSION,
}
ACCEPTANCE_LOGIC_VERSION = "content-and-identity-v2"
DATASET_VERSION = "littlequeen-qwen35-validation-dataset-v3"
WORK_ROOT = Path("/content/littlequeen_qwen35_validation")
RCLONE_INPUT = os.environ.get("QWEN35_RCLONE_INPUT", "")
RCLONE_RESULTS = os.environ.get("QWEN35_RCLONE_RESULTS", "")
RCLONE_ENABLED = bool(RCLONE_INPUT and RCLONE_RESULTS)
RCLONE_CONFIG = os.environ.get(
    "QWEN35_RCLONE_CONFIG",
    "/content/rclone.conf",
)
DRIVE_ROOT = Path(
    os.environ.get(
        "QWEN35_DRIVE_ROOT",
        "/content/drive/MyDrive/littlequeen/storybook_validation_v3_qwen35",
    )
)
INPUT_ROOT = (
    WORK_ROOT / "input_cache" if RCLONE_ENABLED else DRIVE_ROOT / "input"
)
RESULT_ROOT = (
    WORK_ROOT / "results" if RCLONE_ENABLED else DRIVE_ROOT / "results"
)
BENCHMARK_SCENES = (2, 3, 4)
RETRY_FAILED = os.environ.get("QWEN35_RETRY_FAILED", "").lower() in {
    "1",
    "true",
    "yes",
}
RETRY_FAILED_SCENES = {
    int(value)
    for value in os.environ.get("QWEN35_RETRY_FAILED_SCENES", "").split(",")
    if value.strip().isdigit()
}
RETRY_FAILED_MAX_ATTEMPTS = int(
    os.environ.get("QWEN35_RETRY_FAILED_MAX_ATTEMPTS", "2")
)
NONVISUAL_PROOF_PHRASES = (
    "identity confirmation",
    "name confirmation",
    "setting confirmation",
    "setting-name confirmation",
    "written label",
    "text label",
)

INVENTORY_PROMPT = """
You are a strict visual inspector. Examine only the candidate illustration. You
are not being told what it was intended to show. Report what is visibly present,
not what is likely or aesthetically implied. Count separate bodies exactly, but
keep the response compact. Use one entry per important or foreground person,
with count 1. If more than 11 people are visible, use one final grouped crowd
entry with an estimated count and one enclosing bounding box. Never return more
than 12 people entries. Treat a child duplicated in the background as another
person. Do not call an
animal present unless its body or unmistakable head is visible. Inspect hands,
limbs, faces, heads, cropping, object grips, and accidental text closely.

Return one JSON object and no prose:
{
  "people": [
    {"index": 1, "count": 1,
     "visual_role": "child|queen_like_child|adult|unclear",
     "bbox_0_1000": [left, top, right, bottom], "description": "brief visible traits"}
  ],
  "animals": [
    {"species": "specific species or unclear", "count": 1,
     "bbox_0_1000": [left, top, right, bottom]}
  ],
  "held_objects": [
    {"type": "specific object", "holder_person_index": 1,
     "grip": "secure|ambiguous|not_gripped", "description": "brief"}
  ],
  "visible_actions": ["literal visible action"],
  "setting": ["literal visible setting cue"],
  "visible_text": ["exact text or pseudo-text"],
  "defects": [
    {"type": "extra_limb|missing_limb|bad_hand|fused_body|duplicate_subject|bad_face|other",
     "severity": "minor|major", "description": "brief"}
  ],
  "framing": {
    "head_cut_off": false, "important_subject_cut_off": false,
    "notes": "brief"
  },
  "quality": {
    "anatomy": 0, "composition": 0, "clarity": 0, "storybook_appeal": 0
  }
}
Quality values are integers from 0 to 10. Use empty arrays when nothing applies.
Use exactly one animals entry per species: combine repeated members of the same
species into a total count and one bounding box enclosing that species. Limit
held_objects, visible_actions, setting, visible_text, and defects to at most 8
entries each. Keep every description to 12 words or fewer. Every top-level key
in the schema is required, even when its value is an empty array.
""".strip()

CONTRACT_PROMPT = """
Convert this story scene into a literal visual contract for validating an
illustration. Require only things explicitly stated or indispensable to the
described action. Do not invent requirements. Named animals are distinct
requirements. A crowd or group may allow several background people, but the
Little Queen herself must never be duplicated.

Reserve hard_requirements for visible facts whose absence would make the image
tell the wrong story. General mood, color, lighting, and a named fantasy place
such as "Golden Kingdom" are normally soft preferences or setting cues, not
hard requirements, unless a distinctive location is central to the action.

Return one JSON object and no prose:
{
  "protagonist_required": false,
  "protagonist_name": "Little Queen or empty",
  "required_people": [{"description": "literal role", "minimum_count": 1}],
  "required_animals": [{"species": "literal species", "minimum_count": 1}],
  "required_objects": [{"name": "literal object", "must_be_held": false}],
  "required_actions": ["literal action"],
  "required_setting_cues": ["literal cue"],
  "crowd_allowed": false,
  "hard_requirements": ["short literal requirement"],
  "soft_preferences": ["short preference"]
}

Scene prompt:
__SCENE_PROMPT__

Narration:
__SCENE_SCRIPT__
""".strip()

COMPARE_PROMPT = """
You are comparing a literal scene contract with a blind visual inventory.
Never claim something is visible unless the inventory reports it. Distinguish
named animal species. Missing required subjects, a duplicated protagonist,
major anatomy defects, or unusable cropping are hard failures. A loose setting
or imperfect action is usually soft unless it is the point of the scene.
Each people item has a count; use 1 when omitted. Sum those counts when checking
required people. A grouped background-crowd entry represents several people,
not several protagonists.

This is a visual-content check, not the Little Queen identity check. When the
contract requires the Little Queen, exactly one inventory person with
visual_role "queen_like_child" provisionally satisfies protagonist presence.
Do not require text, a name label, or proof of identity; a separate reference
image check handles exact identity later. Likewise, a named fantasy setting
never requires a written label. Judge it from visible setting cues.

Account for every hard requirement as matched, failed, or uncertain. Use
uncertain_required_items only for a concrete visible entity, object, animal,
effect, or action that the image inspector may have overlooked. Never request
an identity confirmation, name confirmation, setting-name confirmation,
written label, or other nonvisual proof. The score is the percentage of the
literal visual contract that is visibly matched. A single missing requirement
does not make the score zero, although a missing central subject or action may
still be a hard failure. accept_without_identity_check means acceptable on
content and quality before the separate identity check. Classify each
requirement exactly once: never list the same requirement as both matched and
failed, and never call a partially supported requirement matched. A vague
fantasy place name is normally a soft setting issue when the image otherwise
fits; lack of signage, a castle, or architecture is not by itself a hard
failure.

Return one JSON object and no prose:
{
  "hard_failures": ["specific failure"],
  "soft_failures": ["specific issue"],
  "uncertain_required_items": ["short entity or object name needing a targeted check"],
  "matched_requirements": ["specific match"],
  "contract_match_score": 0,
  "accept_without_identity_check": false
}
The score is an integer from 0 to 100.

Contract:
__CONTRACT__

Blind inventory:
__INVENTORY__
""".strip()

TARGET_PROMPT = """
Answer a narrow visual verification question about the candidate illustration.
The expected answer may be no. Do not infer hidden or implied content.

Question: Is a clearly visible __ITEM__ present? If present, how many separate
instances are visible?

Return one JSON object and no prose:
{"item": "__ITEM__", "present": false, "count": 0, "evidence": "brief literal evidence"}
""".strip()

RECONCILE_PROMPT = """
Re-evaluate the contract match using the blind inventory and targeted visual
checks. Targeted checks override an uncertain omission in the inventory, but do
not erase unrelated defects. Apply the same strict rules as before. Exactly one
"queen_like_child" provisionally satisfies Little Queen presence here; never
require written or named identity proof, because exact identity is checked in a
separate reference-image stage. A fantasy setting does not require a written
label. Account for every hard requirement. Score the percentage of the literal
visual contract that is matched; one failure does not automatically make the
score zero. Classify each requirement exactly once and never report it as both
matched and failed. A partially supported requirement is not a match. A vague
fantasy place mismatch is normally soft unless the scene's central action
requires a distinctive location.

Return one JSON object and no prose:
{
  "hard_failures": ["specific failure"],
  "soft_failures": ["specific issue"],
  "matched_requirements": ["specific match"],
  "contract_match_score": 0,
  "accept_without_identity_check": false
}

Contract:
__CONTRACT__

Blind inventory:
__INVENTORY__

Targeted checks:
__TARGETED__
""".strip()

IDENTITY_PROMPT = """
The first image is a candidate storybook illustration. The second image is a
trusted reference sheet showing the same Little Queen in several renderings.
Judge the candidate child against the references. Ignore pose, background,
lighting, and normal perspective changes. Prioritize facial structure, eye
shape, auburn hair shape, child proportions, and overall recognizability.
Separately judge the moon dress and crown/accessory consistency when visible.
Penalize a second or duplicated Little Queen heavily.

Return one JSON object and no prose:
{
  "little_queen_present": false,
  "little_queen_count": 0,
  "identity_match": 0,
  "face_match": 0,
  "hair_match": 0,
  "proportions_match": 0,
  "outfit_consistency": 0,
  "regalia_consistency": 0,
  "identity_failures": ["specific issue"],
  "notes": "brief"
}
Scores are integers from 0 to 10.
""".strip()

INVENTORY_REQUIRED_KEYS = frozenset(
    {
        "people",
        "animals",
        "held_objects",
        "visible_actions",
        "setting",
        "visible_text",
        "defects",
        "framing",
        "quality",
    }
)
COMPARISON_REQUIRED_KEYS = frozenset(
    {
        "hard_failures",
        "soft_failures",
        "matched_requirements",
        "contract_match_score",
        "accept_without_identity_check",
    }
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def rclone_run(arguments: list[str], attempts: int = 20) -> None:
    if not RCLONE_ENABLED:
        raise RuntimeError("rclone transport is not configured")
    command = [
        "/content/rclone",
        "--config",
        RCLONE_CONFIG,
        *arguments,
    ]
    for attempt in range(1, attempts + 1):
        print(
            f"+ [rclone {attempt}/{attempts}]",
            " ".join(command[:3] + arguments),
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(completed.returncode, command)
        time.sleep(min(60, attempt * 5))


def fetch_input(name: str, expected_sha256: str | None = None) -> Path:
    destination = INPUT_ROOT / name
    if (
        destination.is_file()
        and (
            expected_sha256 is None
            or sha256(destination) == expected_sha256
        )
    ):
        return destination
    if not RCLONE_ENABLED:
        if not destination.is_file():
            raise FileNotFoundError(destination)
        actual = sha256(destination)
        if expected_sha256 is not None and actual != expected_sha256:
            raise RuntimeError(
                f"input checksum mismatch for {name}: {actual}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    if RCLONE_ENABLED:
        rclone_run(
            [
                "copyto",
                "--retries",
                "5",
                "--low-level-retries",
                "10",
                "--timeout",
                "90s",
                f"{RCLONE_INPUT}{name}",
                str(temporary),
            ]
        )
    if expected_sha256 is not None:
        actual = sha256(temporary)
        if actual != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"input checksum mismatch for {name}: {actual}"
            )
    temporary.replace(destination)
    return destination


def sync_existing_results() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if not RCLONE_ENABLED:
        return
    try:
        rclone_run(
            [
                "copy",
                "--transfers",
                "4",
                RCLONE_RESULTS,
                str(RESULT_ROOT),
            ],
            attempts=3,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"[result sync warning] existing results were not restored: {exc}",
            flush=True,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def durable_envelope(payload: dict) -> dict:
    return {
        "workflow_version": WORKFLOW_VERSION,
        "payload_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "payload": payload,
    }


def local_durable_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(durable_envelope(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def durable_write(path: Path, payload: dict) -> None:
    envelope = durable_envelope(payload)
    local = WORK_ROOT / "pending" / path.relative_to(RESULT_ROOT)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            shutil.copy2(local, temporary)
            temporary.replace(path)
            loaded = load_durable(path)
            if loaded != payload:
                raise RuntimeError("post-write payload verification failed")
            if RCLONE_ENABLED:
                relative = path.relative_to(RESULT_ROOT).as_posix()
                rclone_run(
                    [
                        "copyto",
                        "--retries",
                        "5",
                        "--low-level-retries",
                        "10",
                        str(path),
                        f"{RCLONE_RESULTS}{relative}",
                    ]
                )
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 5:
                raise
            time.sleep(attempt * 2)


def load_durable(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("workflow_version") not in COMPATIBLE_WORKFLOW_VERSIONS:
            return None
        payload = envelope["payload"]
        expected = hashlib.sha256(canonical(payload)).hexdigest()
        if envelope.get("payload_sha256") != expected:
            return None
        return payload
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as archive:
        destination_root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256(destination) == expected_sha256:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    actual = sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {source}: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(destination)


def assemble_verified_archive(entry: dict) -> Path:
    destination = WORK_ROOT / entry["name"]
    if destination.is_file() and sha256(destination) == entry["sha256"]:
        return destination
    parts = entry.get("parts")
    if not parts:
        source = fetch_input(entry["name"], entry["sha256"])
        copy_verified(
            source,
            destination,
            entry["sha256"],
        )
        return destination
    part_root = WORK_ROOT / "parts"
    part_root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as output:
        for part in parts:
            local_part = part_root / part["name"]
            source_part = fetch_input(part["name"], part["sha256"])
            copy_verified(
                source_part,
                local_part,
                part["sha256"],
            )
            if local_part.stat().st_size != part["bytes"]:
                raise RuntimeError(
                    f"part size mismatch for {part['name']}"
                )
            with local_part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            local_part.unlink(missing_ok=True)
            if RCLONE_ENABLED:
                source_part.unlink(missing_ok=True)
    if temporary.stat().st_size != entry["bytes"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"archive size mismatch for {entry['name']}")
    actual = sha256(temporary)
    if actual != entry["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"archive checksum mismatch for {entry['name']}: {actual}"
        )
    temporary.replace(destination)
    return destination


def parse_json_response(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        value = None
        for match in re.finditer(r"\{", text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is not None:
            return value
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(f"model did not return JSON: {text[:500]}")
        candidate = text[start : end + 1]
        try:
            from json_repair import loads as repair_json_loads

            value = repair_json_loads(candidate)
        except Exception as repair_error:
            raise RuntimeError(
                "model returned irreparable JSON: "
                f"{original_error}; {text[:500]}"
            ) from repair_error
    if not isinstance(value, dict):
        raise RuntimeError("model JSON response is not an object")
    return value


class Qwen:
    def __init__(self) -> None:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.torch = torch
        print(
            f"[model ready] {MODEL_ID} official weights, NF4 4-bit, "
            f"device={self.model.device}",
            flush=True,
        )

    def json(
        self,
        prompt: str,
        image_paths: list[Path] | None = None,
        max_new_tokens: int = 700,
        required_keys: frozenset[str] | None = None,
    ) -> tuple[dict, str, float]:
        from PIL import Image

        image_content = []
        opened = []
        try:
            for path in image_paths or []:
                image = Image.open(path).convert("RGB")
                opened.append(image)
                image_content.append({"type": "image", "image": image})
            total_elapsed = 0.0

            def generate(content: list[dict], token_limit: int) -> tuple[str, float]:
                messages = [{"role": "user", "content": content}]
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to(self.model.device)
                started = time.monotonic()
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=token_limit,
                        do_sample=False,
                        use_cache=True,
                    )
                elapsed = time.monotonic() - started
                prompt_length = inputs["input_ids"].shape[-1]
                response = self.processor.decode(
                    generated[0][prompt_length:],
                    skip_special_tokens=True,
                ).strip()
                return response, elapsed

            def parse_required(response: str) -> dict:
                parsed = parse_json_response(response)
                missing_keys = (required_keys or frozenset()) - parsed.keys()
                if missing_keys:
                    raise RuntimeError(
                        "model JSON omitted required keys: "
                        + ", ".join(sorted(missing_keys))
                    )
                return parsed

            for attempt in range(1, 4):
                attempt_prompt = prompt
                if attempt > 1:
                    attempt_prompt += (
                        "\n\nYour previous response was invalid JSON. Start "
                        "over and return one complete, syntactically valid JSON "
                        "object with double-quoted keys and values. No prose or "
                        "markdown fences."
                    )
                content = [
                    *image_content,
                    {"type": "text", "text": attempt_prompt},
                ]
                text, elapsed = generate(content, max_new_tokens)
                total_elapsed += elapsed
                try:
                    return parse_required(text), text, total_elapsed
                except (RuntimeError, json.JSONDecodeError) as exc:
                    print(
                        f"[model JSON retry {attempt}/3] "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    repairable = not required_keys or all(
                        re.search(
                            rf'["\']{re.escape(key)}["\']\s*:',
                            text,
                        )
                        for key in required_keys
                    )
                    if not repairable:
                        print(
                            f"[model JSON repair skipped {attempt}/3] "
                            "response is incomplete; rerunning vision",
                            flush=True,
                        )
                        if attempt == 3:
                            raise
                        continue
                    repair_prompt = f"""
You are a JSON serializer. Repair the malformed response below so it follows
the original requested JSON schema. Preserve its visual findings and scores.
Do not reconsider the image, add facts, explain, or use markdown. Return one
complete JSON object with double-quoted keys and values.

Original request and schema:
<original_request>
{prompt}
</original_request>

Malformed response:
<malformed_response>
{text}
</malformed_response>
""".strip()
                    repair_text, repair_elapsed = generate(
                        [{"type": "text", "text": repair_prompt}],
                        max_new_tokens,
                    )
                    total_elapsed += repair_elapsed
                    try:
                        parsed = parse_required(repair_text)
                        print(
                            f"[model JSON repaired after attempt {attempt}]",
                            flush=True,
                        )
                        return parsed, repair_text, total_elapsed
                    except (RuntimeError, json.JSONDecodeError) as repair_exc:
                        print(
                            f"[model JSON repair failed {attempt}/3] "
                            f"{type(repair_exc).__name__}: {repair_exc}",
                            flush=True,
                        )
                    if attempt == 3:
                        raise
        finally:
            for image in opened:
                image.close()


def install_dependencies() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "git+https://github.com/huggingface/transformers.git",
            "accelerate>=1.8.0",
            "bitsandbytes>=0.46.0",
            "json-repair>=0.44.1",
            "safetensors",
        ]
    )


def make_reference_sheet(reference_paths: list[Path], output: Path) -> None:
    from PIL import Image, ImageOps

    cells = []
    for path in reference_paths:
        with Image.open(path) as source:
            cell = ImageOps.fit(source.convert("RGB"), (512, 512))
            cells.append(cell.copy())
    if not cells:
        raise RuntimeError("identity scene has no reference images")
    sheet = Image.new("RGB", (1024, 1024), "white")
    positions = ((0, 0), (512, 0), (0, 512), (512, 512))
    for cell, position in zip(cells[:4], positions):
        sheet.paste(cell, position)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def validate_artifact(record: dict, dataset_root: Path) -> Path:
    artifact = record["artifacts"][record["preferred_artifact"]]
    path = dataset_root / artifact["archive_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise RuntimeError(f"candidate artifact verification failed: {record['id']}")
    return path


def quality_score(inventory: dict) -> float:
    quality = inventory.get("quality", {})
    values = []
    for key in ("anatomy", "composition", "clarity", "storybook_appeal"):
        try:
            values.append(max(0.0, min(10.0, float(quality.get(key, 0)))))
        except (TypeError, ValueError):
            values.append(0.0)
    return sum(values) / len(values)


def content_score_and_hard_failure(result: dict) -> tuple[float, bool]:
    comparison = result.get("comparison", {})
    try:
        contract_score = float(comparison.get("contract_match_score", 0))
    except (AttributeError, TypeError, ValueError):
        contract_score = 0.0
    try:
        candidate_quality = float(
            result.get(
                "quality_score",
                quality_score(result.get("inventory", {})),
            )
        )
    except (TypeError, ValueError):
        candidate_quality = 0.0
    hard_failures = comparison.get("hard_failures", [])
    if not isinstance(hard_failures, list):
        hard_failures = ["invalid hard_failures response"]
    score = 0.7 * max(0.0, min(100.0, contract_score))
    score += 3.0 * max(0.0, min(10.0, candidate_quality))
    if hard_failures:
        score = min(score, 58.0)
    return score, bool(hard_failures)


def apply_identity_acceptance(result: dict, identity_required: bool) -> bool:
    if result.get("status") == "failed":
        return False
    before = canonical(result)
    content_score, content_hard_failure = content_score_and_hard_failure(result)
    identity = result.get("identity")
    if not identity_required:
        result["score"] = round(content_score, 3)
        result["accepted"] = bool(
            not content_hard_failure and content_score >= 68
        )
    elif identity_succeeded(result):
        try:
            identity_score = float(identity.get("identity_match", 0))
            queen_count = int(identity.get("little_queen_count", 0))
        except (AttributeError, TypeError, ValueError):
            identity_score = 0.0
            queen_count = 0
        identity_hard_failure = (
            not bool(identity.get("little_queen_present", False))
            or queen_count != 1
            or identity_score < 5.5
        )
        score = 0.82 * content_score + 1.8 * max(
            0.0, min(10.0, identity_score)
        )
        if identity_hard_failure:
            score = min(score, 58.0)
        result["score"] = round(score, 3)
        result["accepted"] = bool(
            not content_hard_failure
            and not identity_hard_failure
            and score >= 68
        )
    else:
        result["score"] = round(
            min(content_score, 58.0)
            if isinstance(identity, dict) and identity.get("status") == "failed"
            else content_score,
            3,
        )
        result["accepted"] = False
    result["acceptance_logic_version"] = ACCEPTANCE_LOGIC_VERSION
    changed = canonical(result) != before
    if changed:
        result["updated_at"] = now()
    return changed


def sanitize_comparison(comparison: dict) -> dict:
    discarded = []
    for key in (
        "hard_failures",
        "soft_failures",
        "uncertain_required_items",
    ):
        values = comparison.get(key, [])
        if not isinstance(values, list):
            continue
        cleaned = []
        for value in values:
            if any(
                phrase in str(value).lower()
                for phrase in NONVISUAL_PROOF_PHRASES
            ):
                discarded.append({"field": key, "value": str(value)})
            else:
                cleaned.append(value)
        comparison[key] = cleaned
    if discarded:
        comparison["discarded_nonvisual_claims"] = discarded
    return comparison


def initial_candidate(
    model: Qwen,
    record: dict,
    image_path: Path,
    contract: dict,
) -> dict:
    inventory, raw_inventory, inventory_seconds = model.json(
        INVENTORY_PROMPT,
        [image_path],
        max_new_tokens=1600,
        required_keys=INVENTORY_REQUIRED_KEYS,
    )
    comparison, raw_comparison, comparison_seconds = model.json(
        COMPARE_PROMPT.replace(
            "__CONTRACT__",
            json.dumps(contract, sort_keys=True),
        ).replace(
            "__INVENTORY__",
            json.dumps(inventory, sort_keys=True),
        ),
        max_new_tokens=500,
        required_keys=COMPARISON_REQUIRED_KEYS,
    )
    comparison = sanitize_comparison(comparison)
    targeted = []
    uncertain = comparison.get("uncertain_required_items", [])
    if isinstance(uncertain, list):
        for item in uncertain[:3]:
            item = str(item).strip()[:100]
            if not item:
                continue
            answer, raw_answer, seconds = model.json(
                TARGET_PROMPT.replace("__ITEM__", item),
                [image_path],
                max_new_tokens=180,
            )
            targeted.append(
                {
                    "item": item,
                    "answer": answer,
                    "raw": raw_answer,
                    "seconds": round(seconds, 3),
                }
            )
    if targeted:
        comparison, raw_comparison, reconcile_seconds = model.json(
            RECONCILE_PROMPT.replace(
                "__CONTRACT__",
                json.dumps(contract, sort_keys=True),
            ).replace(
                "__INVENTORY__",
                json.dumps(inventory, sort_keys=True),
            ).replace(
                "__TARGETED__",
                json.dumps(
                    [item["answer"] for item in targeted],
                    sort_keys=True,
                ),
            ),
            max_new_tokens=500,
            required_keys=COMPARISON_REQUIRED_KEYS,
        )
        comparison = sanitize_comparison(comparison)
        comparison_seconds += reconcile_seconds
    try:
        contract_score = float(comparison.get("contract_match_score", 0))
    except (TypeError, ValueError):
        contract_score = 0.0
    hard_failures = comparison.get("hard_failures", [])
    if not isinstance(hard_failures, list):
        hard_failures = ["invalid hard_failures response"]
    score = 0.7 * max(0.0, min(100.0, contract_score))
    score += 3.0 * quality_score(inventory)
    if hard_failures:
        score = min(score, 58.0)
    return {
        "id": record["id"],
        "scene": record["scene"],
        "candidate_index": record["candidate_index"],
        "image_sha256": record["artifacts"][record["preferred_artifact"]]["sha256"],
        "preferred_artifact": record["preferred_artifact"],
        "stage": "initial",
        "inventory": inventory,
        "comparison": comparison,
        "targeted_checks": targeted,
        "identity": None,
        "quality_score": round(quality_score(inventory), 3),
        "score": round(score, 3),
        "accepted": bool(not hard_failures and score >= 68),
        "acceptance_logic_version": ACCEPTANCE_LOGIC_VERSION,
        "raw_responses": {
            "inventory": raw_inventory,
            "comparison": raw_comparison,
        },
        "timing_seconds": {
            "inventory": round(inventory_seconds, 3),
            "comparison": round(comparison_seconds, 3),
            "total": round(
                inventory_seconds
                + comparison_seconds
                + sum(item["seconds"] for item in targeted),
                3,
            ),
        },
        "updated_at": now(),
    }


def failed_candidate(
    record: dict,
    error: Exception,
    previous: dict | None = None,
) -> dict:
    previous_attempts = 0
    if isinstance(previous, dict):
        try:
            previous_attempts = int(
                previous.get("failure", {}).get("attempts", 0)
            )
        except (AttributeError, TypeError, ValueError):
            previous_attempts = 0
    return {
        "status": "failed",
        "id": record["id"],
        "scene": record["scene"],
        "candidate_index": record["candidate_index"],
        "image_sha256": record["artifacts"][record["preferred_artifact"]]["sha256"],
        "preferred_artifact": record["preferred_artifact"],
        "stage": "failed",
        "inventory": {},
        "comparison": {},
        "targeted_checks": [],
        "identity": None,
        "quality_score": 0.0,
        "score": -1.0,
        "accepted": False,
        "failure": {
            "stage": "initial_candidate",
            "type": type(error).__name__,
            "message": str(error)[:2000],
            "attempts": previous_attempts + 1,
        },
        "raw_responses": {},
        "timing_seconds": {},
        "updated_at": now(),
    }


def failed_identity(result: dict, error: Exception) -> dict:
    result["identity"] = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
    }
    result["score"] = min(float(result.get("score", 0)), 58.0)
    result["accepted"] = False
    result["stage"] = "final"
    result["updated_at"] = now()
    return result


def add_identity(
    model: Qwen,
    result: dict,
    candidate_path: Path,
    reference_sheet: Path,
) -> dict:
    identity, raw, seconds = model.json(
        IDENTITY_PROMPT,
        [candidate_path, reference_sheet],
        max_new_tokens=400,
    )
    result["identity"] = identity
    result["raw_responses"]["identity"] = raw
    result["timing_seconds"]["identity"] = round(seconds, 3)
    result["timing_seconds"]["total"] = round(
        float(result["timing_seconds"]["total"]) + seconds,
        3,
    )
    result["stage"] = "final"
    apply_identity_acceptance(result, identity_required=True)
    return result


def scene_contract(
    model: Qwen,
    scene: dict,
    result_path: Path,
) -> dict:
    existing = load_durable(result_path)
    if existing is not None:
        return existing["contract"]
    contract, raw, seconds = model.json(
        CONTRACT_PROMPT.replace(
            "__SCENE_PROMPT__",
            scene["prompt"],
        ).replace(
            "__SCENE_SCRIPT__",
            scene.get("script", ""),
        ),
        max_new_tokens=500,
    )
    payload = {
        "scene": scene["scene"],
        "prompt": scene["prompt"],
        "script": scene.get("script", ""),
        "contract": contract,
        "raw_response": raw,
        "seconds": round(seconds, 3),
        "created_at": now(),
    }
    durable_write(result_path, payload)
    return contract


def scene_is_complete(path: Path, records: list[dict]) -> bool:
    payload = load_durable(path)
    if payload is None:
        return False
    expected = {
        (
            record["id"],
            record["artifacts"][record["preferred_artifact"]]["sha256"],
        )
        for record in records
    }
    actual = {
        (candidate["id"], candidate["image_sha256"])
        for candidate in payload.get("candidates", [])
    }
    return bool(
        expected == actual
        and payload.get("status") == "complete"
        and all(
            candidate_result_is_current(
                candidate,
                candidate["image_sha256"],
            )
            for candidate in payload.get("candidates", [])
        )
    )


def candidate_checkpoint_is_terminal(
    result: dict | None,
    expected_hash: str,
) -> bool:
    if result is None or result.get("image_sha256") != expected_hash:
        return False
    if result.get("status") == "failed":
        return bool(
            result.get("stage") == "failed"
            and isinstance(result.get("failure"), dict)
        )
    inventory = result.get("inventory")
    comparison = result.get("comparison")
    return bool(
        isinstance(inventory, dict)
        and INVENTORY_REQUIRED_KEYS <= inventory.keys()
        and isinstance(comparison, dict)
        and COMPARISON_REQUIRED_KEYS <= comparison.keys()
    )


def should_retry_failed_checkpoint(result: dict | None) -> bool:
    if not isinstance(result, dict) or result.get("status") != "failed":
        return False
    if RETRY_FAILED:
        return True
    try:
        scene_number = int(result.get("scene"))
        attempts = int(result.get("failure", {}).get("attempts", 0))
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        scene_number in RETRY_FAILED_SCENES
        and attempts < RETRY_FAILED_MAX_ATTEMPTS
    )


def candidate_result_is_current(result: dict | None, expected_hash: str) -> bool:
    if not candidate_checkpoint_is_terminal(result, expected_hash):
        return False
    return not should_retry_failed_checkpoint(result)


def identity_succeeded(result: dict) -> bool:
    identity = result.get("identity")
    return bool(
        isinstance(identity, dict)
        and identity.get("status") != "failed"
        and "identity_match" in identity
    )


def migrate_completed_identity_acceptance() -> tuple[int, int]:
    migrated_candidates = 0
    migrated_scenes = 0
    for scene_path in sorted((RESULT_ROOT / "scenes").glob("scene_*.json")):
        scene_result = load_durable(scene_path)
        if (
            not isinstance(scene_result, dict)
            or scene_result.get("status") != "complete"
            or not scene_result.get("identity_required")
        ):
            continue
        candidates = []
        scene_changed = (
            scene_result.get("acceptance_logic_version")
            != ACCEPTANCE_LOGIC_VERSION
        )
        for embedded in scene_result.get("candidates", []):
            candidate_path = (
                RESULT_ROOT / "candidates" / f"{embedded['id']}.json"
            )
            candidate = load_durable(candidate_path) or embedded
            if apply_identity_acceptance(candidate, identity_required=True):
                local_durable_write(candidate_path, candidate)
                migrated_candidates += 1
                scene_changed = True
            candidates.append(candidate)
        ranking_pool = [
            candidate
            for candidate in candidates
            if candidate.get("status") != "failed"
            and identity_succeeded(candidate)
        ]
        ranked = sorted(
            ranking_pool,
            key=lambda item: (
                bool(item.get("accepted")),
                float(item.get("score", 0)),
            ),
            reverse=True,
        )
        accepted = [
            candidate["id"] for candidate in ranked if candidate.get("accepted")
        ]
        if (
            scene_result.get("accepted_ids") != accepted
            or scene_result.get("accepted_count") != len(accepted)
            or scene_result.get("selected_id")
            != (ranked[0]["id"] if ranked else None)
            or scene_result.get("candidates") != candidates
        ):
            scene_changed = True
        if not scene_changed:
            continue
        scene_result["candidates"] = candidates
        scene_result["accepted_ids"] = accepted
        scene_result["accepted_count"] = len(accepted)
        scene_result["selected_id"] = ranked[0]["id"] if ranked else None
        scene_result["acceptance_logic_version"] = ACCEPTANCE_LOGIC_VERSION
        scene_result["acceptance_migrated_at"] = now()
        local_durable_write(scene_path, scene_result)
        migrated_scenes += 1
    if RCLONE_ENABLED and (migrated_candidates or migrated_scenes):
        rclone_run(
            [
                "copy",
                "--transfers",
                "8",
                "--checkers",
                "16",
                str(RESULT_ROOT),
                RCLONE_RESULTS,
            ]
        )
    print(
        f"[acceptance migration] candidates={migrated_candidates} "
        f"scenes={migrated_scenes}",
        flush=True,
    )
    return migrated_candidates, migrated_scenes


def process_scene(
    model: Qwen,
    scene: dict,
    records: list[dict],
    dataset_root: Path,
    reference_sheet: Path | None,
    total_scene_count: int,
    total_candidate_count: int,
) -> dict:
    scene_number = int(scene["scene"])
    scene_path = RESULT_ROOT / "scenes" / f"scene_{scene_number:03d}.json"
    if scene_is_complete(scene_path, records):
        print(f"[resume scene {scene_number:03d}] already complete", flush=True)
        return load_durable(scene_path)

    contract_path = RESULT_ROOT / "contracts" / f"scene_{scene_number:03d}.json"
    contract = scene_contract(model, scene, contract_path)
    candidate_results = []
    for position, record in enumerate(records, start=1):
        candidate_path = validate_artifact(record, dataset_root)
        result_path = RESULT_ROOT / "candidates" / f"{record['id']}.json"
        result = load_durable(result_path)
        expected_hash = record["artifacts"][record["preferred_artifact"]]["sha256"]
        if not candidate_result_is_current(result, expected_hash):
            previous = result
            try:
                result = initial_candidate(
                    model,
                    record,
                    candidate_path,
                    contract,
                )
            except Exception as exc:
                result = failed_candidate(record, exc, previous)
            durable_write(result_path, result)
            if result.get("status") == "failed":
                print(
                    f"[candidate failed scene {scene_number:03d} "
                    f"{position}/{len(records)}] {record['id']} "
                    f"{result['failure']['type']}: "
                    f"{result['failure']['message']}",
                    flush=True,
                )
            else:
                print(
                    f"[validated scene {scene_number:03d} "
                    f"{position}/{len(records)}] {record['id']} "
                    f"initial score={result['score']:.1f}",
                    flush=True,
                )
            write_progress(total_scene_count, total_candidate_count)
        else:
            print(
                f"[resume candidate] {record['id']} stage={result['stage']}",
                flush=True,
            )
        candidate_results.append(result)

    identity_required = bool(
        scene.get("lora_bindings", {}).get("characters", {}).get("little_queen")
    )
    if identity_required:
        if reference_sheet is None:
            raise RuntimeError("Little Queen scene has no reference sheet")
        shortlist = sorted(
            [
                result
                for result in candidate_results
                if result.get("status") != "failed"
            ],
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:4]
        shortlist_ids = {item["id"] for item in shortlist}
        updated = []
        record_map = {record["id"]: record for record in records}
        for result in candidate_results:
            retry_identity = bool(
                RETRY_FAILED
                and isinstance(result.get("identity"), dict)
                and result["identity"].get("status") == "failed"
            )
            if result["id"] in shortlist_ids and (
                result.get("identity") is None or retry_identity
            ):
                record = record_map[result["id"]]
                candidate_path = validate_artifact(record, dataset_root)
                try:
                    result = add_identity(
                        model,
                        result,
                        candidate_path,
                        reference_sheet,
                    )
                except Exception as exc:
                    result = failed_identity(result, exc)
                durable_write(
                    RESULT_ROOT / "candidates" / f"{result['id']}.json",
                    result,
                )
                if identity_succeeded(result):
                    print(
                        f"[identity {result['id']}] "
                        f"score={result['score']:.1f} "
                        f"accepted={result['accepted']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[identity failed {result['id']}] "
                        f"{result['identity']['error_type']}: "
                        f"{result['identity']['error']}",
                        flush=True,
                    )
                write_progress(total_scene_count, total_candidate_count)
            updated.append(result)
        candidate_results = updated

    for result in candidate_results:
        if apply_identity_acceptance(result, identity_required):
            durable_write(
                RESULT_ROOT / "candidates" / f"{result['id']}.json",
                result,
            )

    ranking_pool = [
        item
        for item in candidate_results
        if item.get("status") != "failed"
    ]
    if identity_required:
        ranking_pool = [
            item for item in ranking_pool if identity_succeeded(item)
        ]
    ranked = sorted(
        ranking_pool,
        key=lambda item: (
            bool(item.get("accepted")),
            float(item.get("score", 0)),
        ),
        reverse=True,
    )
    accepted = [item["id"] for item in ranked if item.get("accepted")]
    result = {
        "status": "complete",
        "scene": scene_number,
        "prompt": scene["prompt"],
        "script": scene.get("script", ""),
        "contract": contract,
        "identity_required": identity_required,
        "candidate_count": len(candidate_results),
        "failed_candidate_count": sum(
            item.get("status") == "failed" for item in candidate_results
        ),
        "accepted_count": len(accepted),
        "accepted_ids": accepted,
        "selected_id": ranked[0]["id"] if ranked else None,
        "candidates": candidate_results,
        "completed_at": now(),
    }
    durable_write(scene_path, result)
    print(
        f"[scene complete {scene_number:03d}] accepted={len(accepted)}/"
        f"{len(candidate_results)} selected={result['selected_id']}",
        flush=True,
    )
    return result


def benchmark_gate(
    results: list[dict],
) -> tuple[bool, list[str], list[str]]:
    problems = []
    warnings = []
    by_scene = {int(result["scene"]): result for result in results}
    if 3 in by_scene:
        scene_three = by_scene[3]
        initial_accept_count = sum(
            bool(
                candidate.get("comparison", {}).get(
                    "accept_without_identity_check",
                    False,
                )
            )
            and not candidate.get("comparison", {}).get("hard_failures", [])
            for candidate in scene_three["candidates"]
        )
        if initial_accept_count >= 10:
            problems.append(
                "scene 3 initially accepted all candidates; validator is not "
                "discriminating"
            )
        scene_three_failures = {
            str(failure).lower()
            for candidate in scene_three["candidates"]
            for failure in candidate.get("comparison", {}).get(
                "hard_failures",
                [],
            )
        }
        if not any(
            animal in failure
            for failure in scene_three_failures
            for animal in ("owl", "fox", "deer", "bear", "animal")
        ):
            problems.append(
                "scene 3 found no missing-animal failures in any candidate"
            )
    fingerprints = []
    for result in results:
        for candidate in result["candidates"]:
            if candidate.get("status") == "failed":
                continue
            comparison = candidate.get("comparison", {})
            invalid_proof_requests = [
                str(item).lower()
                for key in (
                    "hard_failures",
                    "soft_failures",
                    "uncertain_required_items",
                )
                for item in (
                    comparison.get(key, [])
                    if isinstance(comparison.get(key, []), list)
                    else []
                )
                if any(
                    phrase in str(item).lower()
                    for phrase in NONVISUAL_PROOF_PHRASES
                )
            ]
            if invalid_proof_requests:
                warnings.append(
                    f"{candidate['id']} requested nonvisual proof: "
                    + ", ".join(invalid_proof_requests)
                )
            fingerprints.append(
                hashlib.sha256(
                    canonical(
                        {
                            "inventory": candidate.get("inventory"),
                            "comparison": candidate.get("comparison"),
                        }
                    )
                ).hexdigest()
            )
    failed_benchmark_candidates = [
        candidate["id"]
        for result in results
        for candidate in result["candidates"]
        if candidate.get("status") == "failed"
    ]
    if len(failed_benchmark_candidates) > 3:
        problems.append(
            "too many benchmark candidates failed validation: "
            + ", ".join(failed_benchmark_candidates)
        )
    if len(warnings) > 3:
        problems.append(
            f"too many benchmark responses requested nonvisual proof: "
            f"{len(warnings)}"
        )
    if len(set(fingerprints)) < max(5, len(fingerprints) // 3):
        problems.append("benchmark responses are suspiciously repetitive")
    return not problems, problems, warnings


def write_progress(scene_count: int, candidate_count: int) -> None:
    completed_scenes = []
    selected = {}
    for path in sorted((RESULT_ROOT / "scenes").glob("scene_*.json")):
        payload = load_durable(path)
        if payload is None or payload.get("status") != "complete":
            continue
        scene_number = int(payload["scene"])
        completed_scenes.append(scene_number)
        selected[str(scene_number)] = payload.get("selected_id")
    checkpointed_candidates = 0
    failed_candidates = 0
    accepted_candidates = 0
    identity_completed_candidates = 0
    for path in sorted((RESULT_ROOT / "candidates").glob("*.json")):
        payload = load_durable(path)
        if not isinstance(payload, dict):
            continue
        image_hash = payload.get("image_sha256")
        if not isinstance(image_hash, str) or not candidate_checkpoint_is_terminal(
            payload,
            image_hash,
        ):
            continue
        checkpointed_candidates += 1
        failed_candidates += int(payload.get("status") == "failed")
        accepted_candidates += int(bool(payload.get("accepted")))
        identity_completed_candidates += int(identity_succeeded(payload))
    durable_write(
        RESULT_ROOT / "progress.json",
        {
            "status": (
                "complete"
                if len(completed_scenes) == scene_count
                else "running"
            ),
            "scene_count": scene_count,
            "candidate_count": candidate_count,
            "completed_scene_count": len(completed_scenes),
            "completed_candidate_count": checkpointed_candidates,
            "checkpointed_candidate_count": checkpointed_candidates,
            "failed_candidate_count": failed_candidates,
            "identity_completed_candidate_count": identity_completed_candidates,
            "accepted_candidate_count": accepted_candidates,
            "completion_percent": round(
                100.0 * checkpointed_candidates / max(1, candidate_count),
                3,
            ),
            "completed_scenes": completed_scenes,
            "selected": selected,
            "updated_at": now(),
        },
    )


def main() -> int:
    print(f"[workflow] {WORKFLOW_VERSION}", flush=True)
    print(
        f"[transport] "
        f"{'rclone ' + RCLONE_INPUT + ' -> ' + RCLONE_RESULTS if RCLONE_ENABLED else DRIVE_ROOT}",
        flush=True,
    )
    if RCLONE_ENABLED:
        os.chmod("/content/rclone", 0o700)
        INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        sync_existing_results()
    elif not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Drive input directory is unavailable: {INPUT_ROOT}")
    run(["nvidia-smi"])
    install_dependencies()

    manifest_path = fetch_input("archive_manifest.json")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("version") != DATASET_VERSION:
        raise RuntimeError("unexpected archive manifest version")
    expected_index = manifest["dataset_index"]
    index_path = fetch_input(
        "dataset_index.json",
        expected_index["sha256"],
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("version") != DATASET_VERSION:
        raise RuntimeError("unexpected dataset index version")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    control_entry = next(
        entry for entry in manifest["archives"] if entry["kind"] == "control"
    )
    control_archive = assemble_verified_archive(control_entry)
    safe_extract(control_archive, WORK_ROOT)

    reference_paths = [
        WORK_ROOT / item["archive_path"] for item in index["identity_references"]
    ]
    reference_sheet = None
    if reference_paths:
        reference_sheet = WORK_ROOT / "control" / "little_queen_reference_sheet.jpg"
        make_reference_sheet(reference_paths, reference_sheet)

    scenes = {
        int(scene["scene"]): scene
        for scene in json.loads(
            (WORK_ROOT / "control" / "compiled_story.json").read_text(
                encoding="utf-8"
            )
        )["scenes"]
    }
    records_by_scene = {}
    for record in index["records"]:
        records_by_scene.setdefault(int(record["scene"]), []).append(record)
    migrate_completed_identity_acceptance()
    model = Qwen()
    shard_entries = {
        entry["name"]: entry
        for entry in manifest["archives"]
        if entry["kind"] == "scene_shard"
    }
    scene_order = [
        *[scene for scene in BENCHMARK_SCENES if scene in scenes],
        *[scene for scene in sorted(scenes) if scene not in BENCHMARK_SCENES],
    ]
    benchmark_results = []
    loaded_shard = None
    extraction_root = WORK_ROOT / "shard"

    for ordinal, scene_number in enumerate(scene_order, start=1):
        records = records_by_scene[scene_number]
        scene_path = RESULT_ROOT / "scenes" / f"scene_{scene_number:03d}.json"
        if scene_is_complete(scene_path, records):
            result = load_durable(scene_path)
            print(
                f"[progress {ordinal}/{len(scene_order)}] "
                f"scene {scene_number:03d} already complete",
                flush=True,
            )
        else:
            shard_name = records[0]["shard"]
            if shard_name != loaded_shard:
                shutil.rmtree(extraction_root, ignore_errors=True)
                entry = shard_entries[shard_name]
                local_archive = assemble_verified_archive(entry)
                safe_extract(local_archive, extraction_root)
                loaded_shard = shard_name
                print(f"[shard ready] {shard_name}", flush=True)
            result = process_scene(
                model,
                scenes[scene_number],
                records,
                extraction_root,
                reference_sheet,
                index["scene_count"],
                index["candidate_count"],
            )
        if scene_number in BENCHMARK_SCENES:
            benchmark_results.append(result)
            if len(benchmark_results) == len(
                [scene for scene in BENCHMARK_SCENES if scene in scenes]
            ):
                passed, problems, warnings = benchmark_gate(benchmark_results)
                durable_write(
                    RESULT_ROOT / "benchmark.json",
                    {
                        "passed": passed,
                        "scenes": list(BENCHMARK_SCENES),
                        "problems": problems,
                        "warnings": warnings,
                        "completed_at": now(),
                    },
                )
                if not passed:
                    raise RuntimeError(
                        "benchmark gate failed: " + "; ".join(problems)
                    )
                for warning in warnings:
                    print(f"[benchmark warning] {warning}", flush=True)
                print("[benchmark passed] scenes 2, 3, and 4", flush=True)
        write_progress(index["scene_count"], index["candidate_count"])
        print(
            f"[progress {ordinal}/{len(scene_order)}] scene {scene_number:03d}",
            flush=True,
        )
        gc.collect()
        try:
            model.torch.cuda.empty_cache()
        except RuntimeError:
            pass

    write_progress(index["scene_count"], index["candidate_count"])
    print("[validation complete]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

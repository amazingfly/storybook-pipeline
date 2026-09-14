#!/usr/bin/env python3
"""Validate a story JSON file and resolve its available LoRAs and candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "storybook_mvp_v1" / "lora_catalog.json"
DEFAULT_SCHEMA = ROOT / "storybook_mvp_v1" / "story_schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("story", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warn(messages: list[str], message: str) -> None:
    messages.append(message)
    print(f"WARNING: {message}")


def normalized_selection(value: str | dict[str, Any]) -> dict[str, Any]:
    return {"name": value} if isinstance(value, str) else dict(value)


def resolve_selection(
    raw: str | dict[str, Any],
    *,
    expected_category: str,
    catalog: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | None:
    selection = normalized_selection(raw)
    name = selection["name"]
    available = catalog.get(name)
    if available is None:
        warn(warnings, f"LoRA {name} not found; ignoring.")
        return None
    if available["category"] != expected_category:
        warn(
            warnings,
            f"LoRA {name} is category {available['category']}, not "
            f"{expected_category}; ignoring.",
        )
        return None
    weight = float(selection.get("weight", available["default_weight"]))
    resolved = {
        "name": name,
        "category": available["category"],
        "mode": available["mode"],
        "weight": weight,
        "label": available["label"],
    }
    if available["mode"] == "base":
        path = (ROOT / available["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        validation_references = []
        for reference_name in available.get("validation_references", []):
            reference = (ROOT / reference_name).resolve()
            if not reference.is_file():
                raise FileNotFoundError(reference)
            validation_references.append(str(reference))
        resolved.update(
            {
                "path": str(path),
                "remote_name": path.name,
                "adapter": available["adapter"],
                "trigger": available["trigger"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "validation_references": validation_references,
            }
        )
    else:
        files = []
        for item in available["files"]:
            path = (ROOT / item["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(
                {
                    **item,
                    "path": str(path),
                    "remote_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        resolved["files"] = files
    return resolved


def compile_story(
    story_path: Path,
    catalog_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    story = json.loads(story_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(story)
    catalog_document = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog = catalog_document["loras"]
    warnings: list[str] = []
    scenes = sorted(story["scenes"], key=lambda item: item["scene"])
    scene_numbers = [item["scene"] for item in scenes]
    expected = list(range(1, len(scenes) + 1))
    if scene_numbers != expected:
        raise ValueError(
            f"scene numbers must be canonical and contiguous: expected {expected}, "
            f"found {scene_numbers}"
        )

    settings = {
        "candidates_per_scene": int(story["settings"]["candidates_per_scene"]),
        "seed_start": int(story["settings"]["seed_start"]),
        "resolution": story["settings"].get("resolution", [832, 1216]),
        "pre_roll_seconds": float(story["settings"].get("pre_roll_seconds", 0.5)),
        "post_roll_seconds": float(story["settings"].get("post_roll_seconds", 0.5)),
        "fps": int(story["settings"].get("fps", 30)),
        "tts": {
            "engine": story["settings"].get("tts", {}).get("engine", "piper"),
            "voice": story["settings"].get("tts", {}).get(
                "voice", "en_US-libritts_r-medium"
            ),
            "length_scale": float(
                story["settings"].get("tts", {}).get("length_scale", 1.0)
            ),
        },
    }
    width, height = (int(value) for value in settings["resolution"])
    if width % 8 or height % 8:
        raise ValueError("resolution dimensions must be divisible by 8")

    resolved_scenes = []
    candidates = []
    required: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        scene_loras = scene["loras"]
        selections = []
        seen_loras: dict[str, dict[str, Any]] = {}
        lora_bindings: dict[str, Any] = {"characters": {}, "scene": {}}

        def add_selection(
            raw: str | dict[str, Any],
            *,
            expected_category: str,
            character: str | None,
            role: str,
        ) -> None:
            selection = resolve_selection(
                raw,
                expected_category=expected_category,
                catalog=catalog,
                warnings=warnings,
            )
            if not selection:
                return
            previous = seen_loras.get(selection["name"])
            if previous:
                warn(
                    warnings,
                    f"Scene {scene['scene']} repeats LoRA {selection['name']} "
                    f"for {character or 'scene'}.{role}; the earlier "
                    f"{previous['character'] or 'scene'}.{previous['role']} "
                    "binding was kept.",
                )
                return
            bound = {
                **selection,
                "character": character,
                "role": role,
            }
            seen_loras[selection["name"]] = bound
            selections.append(bound)
            required.setdefault(selection["name"], selection)
            binding = {
                "name": selection["name"],
                "category": selection["category"],
                "mode": selection["mode"],
                "weight": selection["weight"],
            }
            if character is None:
                lora_bindings["scene"][role] = binding
            else:
                lora_bindings["characters"].setdefault(character, {})[
                    role
                ] = binding

        for character_name, character_loras in scene_loras["characters"].items():
            for role, raw_selection in character_loras.items():
                expected_category = (
                    "character"
                    if role == "identity"
                    else "outfit"
                    if role == "outfit"
                    else "extra"
                )
                add_selection(
                    raw_selection,
                    expected_category=expected_category,
                    character=character_name,
                    role=role,
                )
        for role, raw_selection in scene_loras.get("scene", {}).items():
            add_selection(
                raw_selection,
                expected_category="extra",
                character=None,
                role=role,
            )

        base_loras = [
            {
                **{
                    key: selection[key]
                    for key in (
                        "name",
                        "label",
                        "adapter",
                        "weight",
                        "trigger",
                        "remote_name",
                        "validation_references",
                    )
                },
                "character": selection["character"],
                "role": selection["role"],
            }
            for selection in selections
            if selection["mode"] == "base"
        ]
        extra_loras = [
            {
                "name": selection["name"],
                "label": selection["label"],
                "mode": selection["mode"],
                "weight": selection["weight"],
                "character": selection["character"],
                "role": selection["role"],
            }
            for selection in selections
            if selection["mode"] != "base"
        ]
        resolved_scene = {
            "scene": scene["scene"],
            "prompt": scene["prompt"].strip(),
            "script": scene["script"].strip(),
            "lora_bindings": lora_bindings,
            "base_loras": base_loras,
            "extra_loras": extra_loras,
        }
        resolved_scenes.append(resolved_scene)
        for candidate_index in range(1, settings["candidates_per_scene"] + 1):
            candidates.append(
                {
                    **resolved_scene,
                    "id": (
                        f"scene_{scene['scene']:03d}_candidate_{candidate_index:02d}"
                    ),
                    "candidate_index": candidate_index,
                    "seed": (
                        settings["seed_start"]
                        + scene["scene"] * 1000
                        + candidate_index
                    ),
                }
            )

    return {
        "version": "storybook-compiled-v2",
        "source": str(story_path.resolve()),
        "title": story["title"].strip(),
        "settings": settings,
        "scenes": resolved_scenes,
        "candidates": candidates,
        "required_loras": list(required.values()),
        "warnings": warnings,
        "catalog_version": catalog_document["version"],
    }


def main() -> int:
    args = parse_args()
    compiled = compile_story(args.story, args.catalog, args.schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compiled, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "title": compiled["title"],
                "scene_count": len(compiled["scenes"]),
                "candidate_count": len(compiled["candidates"]),
                "required_loras": [
                    item["name"] for item in compiled["required_loras"]
                ],
                "warning_count": len(compiled["warnings"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "compile_storybook_script.py"
SPEC = importlib.util.spec_from_file_location("compile_storybook_script", MODULE_PATH)
assert SPEC and SPEC.loader
COMPILER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPILER)
RENDERER_PATH = ROOT / "scripts" / "render_storybook_mvp.py"
RENDERER_SPEC = importlib.util.spec_from_file_location(
    "render_storybook_mvp", RENDERER_PATH
)
assert RENDERER_SPEC and RENDERER_SPEC.loader
RENDERER = importlib.util.module_from_spec(RENDERER_SPEC)
RENDERER_SPEC.loader.exec_module(RENDERER)
REMOTE_PATH = ROOT / "colab_storybook_mvp_v1" / "remote_run.py"
REMOTE_SPEC = importlib.util.spec_from_file_location(
    "storybook_remote_run", REMOTE_PATH
)
assert REMOTE_SPEC and REMOTE_SPEC.loader
REMOTE = importlib.util.module_from_spec(REMOTE_SPEC)
REMOTE_SPEC.loader.exec_module(REMOTE)
TOKEN_REFRESH_PATH = ROOT / "scripts" / "refresh_colab_runtime_token.py"
TOKEN_REFRESH_SPEC = importlib.util.spec_from_file_location(
    "refresh_colab_runtime_token", TOKEN_REFRESH_PATH
)
assert TOKEN_REFRESH_SPEC and TOKEN_REFRESH_SPEC.loader
TOKEN_REFRESH = importlib.util.module_from_spec(TOKEN_REFRESH_SPEC)
TOKEN_REFRESH_SPEC.loader.exec_module(TOKEN_REFRESH)
SYNC_PATH = ROOT / "scripts" / "sync_storybook_checkpoints.py"
SYNC_SPEC = importlib.util.spec_from_file_location(
    "sync_storybook_checkpoints", SYNC_PATH
)
assert SYNC_SPEC and SYNC_SPEC.loader
SYNC = importlib.util.module_from_spec(SYNC_SPEC)
SYNC_SPEC.loader.exec_module(SYNC)


def little_queen_loras() -> dict:
    return {
        "characters": {
            "little_queen": {
                "identity": "little_queen_v2",
                "outfit": "moon_dress_v1",
                "accessories": "moonstar_accessories_v1",
            }
        }
    }


def write_story(directory: Path, scenes: list[dict]) -> Path:
    document = {
        "version": "storybook-script-v2",
        "title": "Test Story",
        "settings": {
            "candidates_per_scene": 2,
            "seed_start": 100,
            "resolution": [832, 1216],
        },
        "scenes": scenes,
    }
    path = directory / "story.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def scene(number: int, *, include_little_queen: bool = True) -> dict:
    return {
        "scene": number,
        "loras": (
            little_queen_loras()
            if include_little_queen
            else {"characters": {}}
        ),
        "prompt": "The Little Queen stands in a castle library.",
        "script": "The Little Queen opened a book.",
    }


def compile_story(path: Path) -> dict:
    # Compilation reads and hashes assets but does not load model tensors.
    # Build tiny fixtures instead of relying on workstation weights and outputs.
    catalog_path = ROOT / "storybook_mvp_v1" / "lora_catalog.json"
    catalog = json.loads(catalog_path.read_text())["loras"]
    assets = path.parent / "assets"
    for item in catalog.values():
        names = ([item["path"]] if "path" in item else [])
        names += item.get("validation_references", [])
        names += [entry["path"] for entry in item.get("files", [])]
        for name in names:
            target = assets / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"compiler test fixture")
    with patch.object(COMPILER, "ROOT", assets):
        return COMPILER.compile_story(
            path, catalog_path, ROOT / "storybook_mvp_v1" / "story_schema.json"
        )


class StorybookMvpTest(unittest.TestCase):
    def test_compiles_canonical_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled = compile_story(
                write_story(Path(temporary), [scene(1), scene(2)])
            )

        self.assertEqual(
            [item["id"] for item in compiled["candidates"]],
            [
                "scene_001_candidate_01",
                "scene_001_candidate_02",
                "scene_002_candidate_01",
                "scene_002_candidate_02",
            ],
        )
        self.assertEqual(
            [item["seed"] for item in compiled["candidates"]],
            [1101, 1102, 2101, 2102],
        )
        first = compiled["scenes"][0]
        self.assertEqual(first["base_loras"][0]["character"], "little_queen")
        self.assertEqual(first["base_loras"][0]["role"], "identity")
        self.assertEqual(
            first["lora_bindings"]["characters"]["little_queen"][
                "accessories"
            ]["name"],
            "moonstar_accessories_v1",
        )

    def test_unknown_lora_warns_and_is_ignored(self) -> None:
        first = scene(1)
        first["loras"]["characters"]["little_queen"]["wand"] = "future_sword_v9"
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(output):
                compiled = compile_story(write_story(Path(temporary), [first]))

        self.assertIn(
            "WARNING: LoRA future_sword_v9 not found; ignoring.",
            output.getvalue(),
        )
        self.assertEqual(
            compiled["scenes"][0]["extra_loras"],
            [
                {
                    "name": "moonstar_accessories_v1",
                    "label": "Moonstar crown, jewelry, and wand",
                    "mode": "moonstar_regional_set",
                    "weight": 1.0,
                    "character": "little_queen",
                    "role": "accessories",
                }
            ],
        )

    def test_named_reference_can_override_weight(self) -> None:
        first = scene(1)
        first["loras"]["characters"]["little_queen"]["outfit"] = {
            "name": "moon_dress_v1",
            "weight": 0.65,
        }
        with tempfile.TemporaryDirectory() as temporary:
            compiled = compile_story(
                write_story(Path(temporary), [first, scene(2)])
            )

        outfit = next(
            item
            for item in compiled["scenes"][0]["base_loras"]
            if item["role"] == "outfit"
        )
        self.assertEqual(outfit["weight"], 0.65)
        second_outfit = next(
            item
            for item in compiled["scenes"][1]["base_loras"]
            if item["role"] == "outfit"
        )
        self.assertEqual(second_outfit["weight"], 0.5)

    def test_scene_without_character_uses_no_character_loras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled = compile_story(
                write_story(
                    Path(temporary),
                    [scene(1), scene(2, include_little_queen=False)],
                )
            )

        queen_free_scene = compiled["scenes"][1]
        self.assertEqual(queen_free_scene["base_loras"], [])
        self.assertEqual(queen_free_scene["extra_loras"], [])
        self.assertEqual(queen_free_scene["lora_bindings"]["characters"], {})
        self.assertIn(
            "little_queen_v2",
            {item["name"] for item in compiled["required_loras"]},
        )

    def test_old_flat_lora_shape_is_rejected(self) -> None:
        first = scene(1)
        first["character"] = {"lora": "little_queen_v2"}
        first["outfit"] = {"lora": "moon_dress_v1"}
        first["loras"] = ["moonstar_accessories_v1"]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_story(Path(temporary), [first])
            with self.assertRaises(jsonschema.ValidationError):
                compile_story(path)

    def test_rejects_non_contiguous_scene_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_story(Path(temporary), [scene(1), scene(3)])
            with self.assertRaisesRegex(ValueError, "canonical and contiguous"):
                compile_story(path)

    def test_storybook_style_does_not_force_solo_composition(self) -> None:
        candidate = {
            "prompt": "A crowded village market beneath colorful awnings.",
            "base_loras": [],
            "extra_loras": [],
        }

        prompt = REMOTE.build_base_prompt(candidate)
        negative = REMOTE.build_negative_prompt(candidate)

        self.assertIn("polished anime storybook illustration", prompt)
        self.assertNotIn("solo", prompt)
        self.assertNotIn("two people", negative)
        self.assertNotIn("crown", negative)

    def test_moonstar_prompt_uses_accessory_staging_only_when_selected(self) -> None:
        candidate = {
            "prompt": "The Little Queen listens to the villagers.",
            "base_loras": [
                {
                    "trigger": "lqxl Little Queen",
                }
            ],
            "extra_loras": [
                {
                    "mode": "moonstar_regional_set",
                    "weight": 1.0,
                }
            ],
        }

        prompt = REMOTE.build_base_prompt(candidate)
        negative = REMOTE.build_negative_prompt(candidate)

        self.assertIn("Little Queen centered foreground", prompt)
        self.assertIn("vertical guide rod", prompt)
        self.assertIn("crown", negative)
        self.assertIn("floating rod", negative)

    def test_durable_checkpoint_restores_completed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            drive_root = temporary_root / "drive"
            first_root = temporary_root / "first"
            first_root.mkdir()
            config = {
                "title": "Checkpoint Story",
                "candidates": [{"id": "scene_001_candidate_01"}],
            }
            compiled = json.dumps(config, sort_keys=True).encode()
            (first_root / "compiled_story.json").write_bytes(compiled)
            checkpoint = REMOTE.DurableCheckpoint(
                config,
                run_root=first_root,
                drive_root=drive_root,
                interval=1,
            )
            self.assertEqual(checkpoint.restore(), 0)
            base = first_root / "base" / "scene_001_candidate_01.png"
            base.parent.mkdir()
            base.write_bytes(b"completed-image")
            checkpoint.tick()

            second_root = temporary_root / "second"
            second_root.mkdir()
            (second_root / "compiled_story.json").write_bytes(compiled)
            restored = REMOTE.DurableCheckpoint(
                config,
                run_root=second_root,
                drive_root=drive_root,
                interval=1,
            )
            self.assertEqual(restored.restore(), 1)
            self.assertEqual(
                (
                    second_root / "base" / "scene_001_candidate_01.png"
                ).read_bytes(),
                b"completed-image",
            )
            manifest = json.loads(restored.manifest_path.read_text())
            self.assertEqual(
                manifest["version"],
                "storybook-durable-checkpoint-v1",
            )

    def test_generation_only_manifest_upload_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            drive_root = temporary_root / "drive"
            run_root = temporary_root / "run"
            run_root.mkdir()
            config = {
                "title": "Checkpoint Story",
                "candidates": [{"id": "scene_001_candidate_01"}],
            }
            (run_root / "compiled_story.json").write_text(
                json.dumps(config, sort_keys=True),
                encoding="utf-8",
            )
            compiled_sha = REMOTE.sha256(run_root / "compiled_story.json")
            checkpoint_root = (
                drive_root / f"checkpoint-story_{compiled_sha[:16]}"
            )
            checkpoint_root.mkdir(parents=True)
            stale = {
                "version": "storybook-durable-checkpoint-v1",
                "title": config["title"],
                "compiled_story_sha256": compiled_sha,
                "candidate_count": 1,
                "checkpoint_interval": 5,
                "batches": [
                    {
                        "sequence": 1,
                        "archive": "batch_000001.tar",
                        "bytes": 1,
                        "sha256": "stale",
                        "file_count": 1,
                    }
                ],
                "files": {"base/stale.png": {"bytes": 1, "batch": 1}},
            }
            (checkpoint_root / "checkpoint_manifest.json").write_text(
                json.dumps(stale),
                encoding="utf-8",
            )
            uploaded = {
                **stale,
                "batches": [],
                "files": {
                    "base/scene_001_candidate_01.png": {
                        "bytes": 123,
                        "batch": 0,
                    }
                },
            }
            upload_path = temporary_root / "uploaded_manifest.json"
            upload_path.write_text(json.dumps(uploaded), encoding="utf-8")
            prior_upload = REMOTE.CHECKPOINT_MANIFEST_UPLOAD
            prior_generation_only = REMOTE.GENERATION_ONLY
            try:
                REMOTE.CHECKPOINT_MANIFEST_UPLOAD = upload_path
                REMOTE.GENERATION_ONLY = True
                checkpoint = REMOTE.DurableCheckpoint(
                    config,
                    run_root=run_root,
                    drive_root=drive_root,
                )
            finally:
                REMOTE.CHECKPOINT_MANIFEST_UPLOAD = prior_upload
                REMOTE.GENERATION_ONLY = prior_generation_only
            self.assertEqual(checkpoint.manifest, uploaded)

    def test_checkpoint_merge_rejects_conflicting_batch(self) -> None:
        local = {
            "compiled_story_sha256": "story-sha",
            "batches": [
                {
                    "sequence": 1,
                    "archive": "batch_000001.tar",
                    "bytes": 100,
                    "sha256": "local",
                    "file_count": 1,
                }
            ],
            "files": {"base/scene.png": {"bytes": 10, "batch": 1}},
        }
        remote = {
            **local,
            "batches": [
                {
                    **local["batches"][0],
                    "bytes": 200,
                    "sha256": "remote",
                }
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            SYNC.merge_manifests(local, remote, "story-sha")

    def test_checkpoint_merge_appends_new_immutable_batch(self) -> None:
        local_batch = {
            "sequence": 1,
            "archive": "batch_000001.tar",
            "bytes": 100,
            "sha256": "first",
            "file_count": 1,
        }
        new_batch = {
            "sequence": 2,
            "archive": "batch_000002.tar",
            "bytes": 200,
            "sha256": "second",
            "file_count": 1,
        }
        local = {
            "compiled_story_sha256": "story-sha",
            "batches": [local_batch],
            "files": {"base/one.png": {"bytes": 10, "batch": 1}},
        }
        remote = {
            "compiled_story_sha256": "story-sha",
            "batches": [local_batch, new_batch],
            "files": {
                **local["files"],
                "base/two.png": {"bytes": 20, "batch": 2},
            },
        }
        merged = SYNC.merge_manifests(local, remote, "story-sha")
        self.assertEqual(merged["batches"], [local_batch, new_batch])
        self.assertEqual(len(merged["files"]), 2)

    def test_refreshes_expiring_colab_runtime_token(self) -> None:
        session = SimpleNamespace(
            endpoint="runtime-endpoint",
            token="expired-token",
            url="https://expired.invalid",
        )

        class Store:
            def get(self, name):
                return session if name == "storybook" else None

            def add(self, updated):
                self.updated = updated

        store = Store()
        assignment = SimpleNamespace(
            endpoint="runtime-endpoint",
            runtime_proxy_info=SimpleNamespace(
                token="fresh-token",
                url="https://fresh.invalid",
                token_expires_in_seconds=3600,
            ),
        )
        state = SimpleNamespace(
            store=store,
            client=SimpleNamespace(
                list_assignments=lambda: [assignment],
            ),
        )

        changed, expires_in = TOKEN_REFRESH.refresh_session_state(
            state, "storybook"
        )

        self.assertTrue(changed)
        self.assertEqual(expires_in, 3600)
        self.assertEqual(session.token, "fresh-token")
        self.assertEqual(session.url, "https://fresh.invalid")
        self.assertIs(store.updated, session)

    def test_parses_isolated_criterion_response(self) -> None:
        parsed = RENDERER.parse_criterion_json(
            json.dumps(
                {
                    "criterion": "wand_held_by_hand",
                    "candidates": [
                        {
                            "id": "c01",
                            "pass": True,
                            "quality": 4,
                            "reason": "Fingers overlap the shaft.",
                        },
                        {
                            "id": "C02",
                            "pass": False,
                            "quality": 1,
                            "reason": "Visible gap beside hand.",
                        },
                    ],
                }
            ),
            "wand_held_by_hand",
            {"C01", "C02"},
        )

        self.assertTrue(parsed["C01"]["pass"])
        self.assertFalse(parsed["C02"]["pass"])

    def test_all_criteria_are_hard_gates(self) -> None:
        criterion_runs = {}
        for criterion in RENDERER.GEMMA_CRITERIA:
            criterion_runs[criterion["key"]] = {
                "candidates": {
                    "C01": {"pass": True, "quality": 4, "reason": "Clean."},
                    "C02": {"pass": True, "quality": 4, "reason": "Clean."},
                }
            }
        criterion_runs["wand_held_by_hand"]["candidates"]["C02"] = {
            "pass": False,
            "quality": 2,
            "reason": "Grip anchor is too far from a localized hand.",
        }
        aggregated = RENDERER.aggregate_criterion_results(
            ["C01", "C02"], criterion_runs
        )

        self.assertTrue(aggregated["C01"]["accepted"])
        self.assertFalse(aggregated["C02"]["accepted"])
        self.assertEqual(
            aggregated["C02"]["failed_checks"], ["wand_held_by_hand"]
        )
        self.assertGreater(aggregated["C02"]["overall_score"], 80)

    def test_accessory_record_without_refinement_falls_back_to_base(self) -> None:
        record = {
            "uses_moonstar_accessories": True,
            "artifacts": {"base": "base/scene_002_candidate_01.png"},
        }

        variants = RENDERER.deterministic_variants(record)

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["variant"], "base_fallback")
        self.assertEqual(variants[0]["path"], record["artifacts"]["base"])
        self.assertFalse(variants[0]["deterministic_accepted"])

    def test_selection_signature_changes_with_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            image = run_root / "base" / "candidate.png"
            image.parent.mkdir()
            image.write_bytes(b"first")
            records = [
                {
                    "id": "scene_001_candidate_01",
                    "candidate_index": 1,
                    "uses_moonstar_accessories": True,
                    "artifacts": {"base": "base/candidate.png"},
                }
            ]
            first = RENDERER.selection_input_signature(run_root, records)
            image.write_bytes(b"second-image")
            second = RENDERER.selection_input_signature(run_root, records)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()

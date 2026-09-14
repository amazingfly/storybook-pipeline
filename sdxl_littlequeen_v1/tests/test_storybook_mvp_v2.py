from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = ROOT / "scripts" / "render_storybook_mvp_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "render_storybook_mvp_v2", RENDERER_PATH
)
assert SPEC and SPEC.loader
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


def record(prompt: str) -> dict:
    return {
        "scene_number": 2,
        "prompt": prompt,
        "script": "The Little Queen helps her friends.",
        "uses_moonstar_accessories": True,
        "base_loras": [
            {
                "name": "little_queen_v2",
                "role": "identity",
                "character": "little_queen",
            }
        ],
        "extra_loras": [
            {
                "name": "moonstar_accessories_v1",
                "role": "accessories",
                "character": "little_queen",
            }
        ],
        "artifacts": {
            "base": "base/candidate.png",
            "wand_only": "wand/candidate.png",
            "grip_repaired": "grip/candidate.png",
        },
        "validation": {"accepted": False},
        "variant_validation": {
            "wand_only": {"accepted": True, "stage": "final"},
            "grip_repaired": {"accepted": True, "stage": "final"},
        },
    }


def contract(*, wand_required: bool) -> dict:
    return {
        "character_required": True,
        "identity_required": True,
        "accessories_expected": True,
        "living_subject_required": True,
        "wand_required": wand_required,
    }


class StorybookMvpV2Test(unittest.TestCase):
    def test_accessory_binding_does_not_imply_wand_requirement(self) -> None:
        item = record("The Little Queen carries a basket through the market.")

        self.assertEqual(RENDERER.explicit_wand_terms(item), [])
        self.assertFalse(
            RENDERER.criterion_applies(
                next(
                    criterion
                    for criterion in RENDERER.GEMMA_CRITERIA
                    if criterion["key"] == "wand_held_by_hand"
                ),
                contract(wand_required=False),
            )
        )

    def test_explicit_wand_enables_wand_checks(self) -> None:
        item = record("The Little Queen raises her crescent wand.")

        self.assertEqual(RENDERER.explicit_wand_terms(item), ["wand"])

    def test_no_wand_scene_uses_base_even_when_refinement_passed(self) -> None:
        variants = RENDERER.deterministic_variants(
            record("The Little Queen carries a basket.")
        )
        selected = RENDERER.choose_representative_variant(
            variants, contract(wand_required=False)
        )

        self.assertEqual(selected["variant"], "base")
        self.assertFalse(selected["refinement_eligible"])

    def test_wand_scene_uses_best_eligible_refinement(self) -> None:
        variants = RENDERER.deterministic_variants(
            record("The Little Queen raises her wand.")
        )
        selected = RENDERER.choose_representative_variant(
            variants, contract(wand_required=True)
        )

        self.assertEqual(selected["variant"], "grip_repaired")
        self.assertTrue(selected["refinement_eligible"])

    def test_soft_failure_does_not_reject_good_image(self) -> None:
        runs = {}
        for criterion in RENDERER.GEMMA_CRITERIA:
            runs[criterion["key"]] = {
                "candidates": {
                    "C01": {"pass": True, "quality": 4, "reason": "Clean."}
                }
            }
        runs["clean_headwear_and_accessories"]["candidates"]["C01"] = {
            "pass": False,
            "quality": 2,
            "reason": "Crown is absent.",
        }

        result = RENDERER.aggregate_criterion_results(["C01"], runs)["C01"]

        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["quality_flags"], ["clean_headwear_and_accessories"]
        )
        self.assertEqual(result["failed_checks"], [])

    def test_hard_story_failure_rejects_image(self) -> None:
        runs = {}
        for criterion in RENDERER.GEMMA_CRITERIA:
            runs[criterion["key"]] = {
                "candidates": {
                    "C01": {"pass": True, "quality": 4, "reason": "Clean."}
                }
            }
        runs["required_subjects_and_counts"]["candidates"]["C01"] = {
            "pass": False,
            "quality": 1,
            "reason": "Owl, deer, and bear are missing.",
        }

        result = RENDERER.aggregate_criterion_results(["C01"], runs)["C01"]

        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["failed_checks"], ["required_subjects_and_counts"]
        )

    def test_protagonist_check_allows_requested_other_people(self) -> None:
        criterion = next(
            item
            for item in RENDERER.GEMMA_CRITERIA
            if item["key"] == "protagonist_uniqueness"
        )

        self.assertIn("Other people", criterion["question"])
        self.assertNotIn("exactly one person", criterion["question"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "qwen35_storybook_local"
    / "render_storybook_qwen35.py"
)
SPEC = importlib.util.spec_from_file_location("qwen35_storybook_local", MODULE_PATH)
assert SPEC and SPEC.loader
QWEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QWEN)


def criterion_document() -> dict:
    return {
        "criterion": "coherent_anatomy",
        "candidates": [
            {
                "id": "C01",
                "pass": True,
                "quality": 4,
                "reason": "One coherent figure.",
            }
        ],
    }


def test_extract_json_object_ignores_reasoning_and_trailing_object() -> None:
    expected = criterion_document()
    response = (
        "Analysis before the answer.\n"
        + json.dumps(expected)
        + "\n"
        + json.dumps({"unwanted": "trailing object"})
    )

    assert QWEN.extract_json_object(response) == expected


def test_parse_criterion_json_accepts_first_complete_object() -> None:
    response = json.dumps(criterion_document()) + "\nAdditional prose."

    parsed = QWEN.parse_criterion_json(
        response, "coherent_anatomy", {"C01"}
    )

    assert parsed["C01"]["pass"] is True
    assert parsed["C01"]["quality"] == 4


def test_no_text_prompt_excludes_evaluator_labels() -> None:
    prompt = QWEN.criterion_prompt(
        QWEN.QWEN_CRITERIA[-1],
        {"prompt": "A clean storybook scene."},
        ["C01", "C02"],
        has_identity_references=False,
    )

    assert "Ignore those external labels and gutters" in prompt
    assert '"id": "C02"' in prompt


def test_evaluator_failure_is_not_reusable() -> None:
    failed = {
        "candidates": {
            "C01": {
                "pass": False,
                "quality": 0,
                "reason": "Criterion evaluator failed.",
            }
        }
    }

    assert QWEN.criterion_run_failed(failed, {"C01"}) is True


def test_missing_accessory_metadata_rejects_only_that_candidate(
    tmp_path: Path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "base.png"
    Image.new("RGB", (64, 64), "white").save(image_path)
    candidate = {
        "label": "C01",
        "record": {"artifacts": {"base": "base.png"}},
    }
    criterion = {
        "key": "exactly_one_requested_wand",
        "weight": 1,
        "question": "test",
        "source": "base_prop",
    }

    result = QWEN.preexisting_prop_criterion(
        "http://127.0.0.1:1", tmp_path, [candidate], criterion
    )

    assert result["candidates"]["C01"]["pass"] is False
    assert "metadata is unavailable" in result["candidates"]["C01"]["reason"]

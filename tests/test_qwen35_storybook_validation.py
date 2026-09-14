import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "qwen35_storybook_validation"
    / "remote_run.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_storybook_validation", MODULE_PATH
)
assert SPEC and SPEC.loader
QWEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QWEN)


def candidate_result() -> dict:
    return {
        "id": "scene_001_candidate_01",
        "status": "complete",
        "inventory": {
            "quality": {
                "anatomy": 10,
                "composition": 10,
                "clarity": 10,
                "storybook_appeal": 10,
            }
        },
        "quality_score": 10,
        "comparison": {
            "hard_failures": [],
            "contract_match_score": 100,
        },
        "identity": {
            "little_queen_present": True,
            "little_queen_count": 1,
            "identity_match": 10,
        },
        "accepted": False,
        "score": 50,
    }


def test_parse_json_response_uses_first_complete_object() -> None:
    expected = {"people": [], "quality": {"anatomy": 8}}
    response = (
        "Analysis.\n"
        + json.dumps(expected)
        + "\n"
        + json.dumps({"unwanted": "second object"})
    )

    assert QWEN.parse_json_response(response) == expected


def test_identity_can_promote_content_that_passes() -> None:
    result = candidate_result()

    changed = QWEN.apply_identity_acceptance(
        result, identity_required=True
    )

    assert changed is True
    assert result["accepted"] is True
    assert result["score"] == 100
    assert (
        result["acceptance_logic_version"]
        == QWEN.ACCEPTANCE_LOGIC_VERSION
    )


def test_missing_identity_cannot_pass_identity_scene() -> None:
    result = candidate_result()
    result["identity"] = None

    QWEN.apply_identity_acceptance(result, identity_required=True)

    assert result["accepted"] is False


def test_v33_checkpoint_remains_loadable(tmp_path: Path) -> None:
    payload = {"status": "complete", "value": 3}
    envelope = {
        "workflow_version": "littlequeen-storybook-qwen35-validator-v3.3",
        "payload_sha256": hashlib.sha256(
            QWEN.canonical(payload)
        ).hexdigest(),
        "payload": payload,
    }
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    assert QWEN.load_durable(path) == payload


def test_targeted_failed_scene_retries_only_once() -> None:
    previous_scenes = QWEN.RETRY_FAILED_SCENES
    previous_limit = QWEN.RETRY_FAILED_MAX_ATTEMPTS
    previous_all = QWEN.RETRY_FAILED
    try:
        QWEN.RETRY_FAILED = False
        QWEN.RETRY_FAILED_SCENES = {79, 83}
        QWEN.RETRY_FAILED_MAX_ATTEMPTS = 2
        result = {
            "status": "failed",
            "scene": 79,
            "failure": {"attempts": 1},
        }
        assert QWEN.should_retry_failed_checkpoint(result) is True
        result["failure"]["attempts"] = 2
        assert QWEN.should_retry_failed_checkpoint(result) is False
        result["scene"] = 80
        result["failure"]["attempts"] = 1
        assert QWEN.should_retry_failed_checkpoint(result) is False
    finally:
        QWEN.RETRY_FAILED = previous_all
        QWEN.RETRY_FAILED_SCENES = previous_scenes
        QWEN.RETRY_FAILED_MAX_ATTEMPTS = previous_limit

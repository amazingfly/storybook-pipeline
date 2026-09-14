#!/usr/bin/env python3
"""Compare archived and current Gemma decisions with a human story audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story-dir", type=Path, required=True)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_reports(story_dir: Path, name: str) -> dict[int, dict]:
    reports = {}
    for scene_dir in sorted(story_dir.glob("scene_[0-9][0-9][0-9]")):
        path = scene_dir / name
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            reports[int(report["scene_number"])] = report
    return reports


def evaluate(
    reports: dict[int, dict], human_scenes: dict[int, dict]
) -> dict:
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    decisions = []
    selected_matches = 0
    for scene_number, human in sorted(human_scenes.items()):
        report = reports[scene_number]
        accepted = set(human["human_accepted"])
        rejected = set(human["human_rejected"])
        reviewed = accepted | rejected
        predicted = {
            item["id"]
            for item in report["candidate_winners"]
            if item["accepted"]
        }
        for candidate_id in sorted(reviewed):
            human_pass = candidate_id in accepted
            gemma_pass = candidate_id in predicted
            if human_pass and gemma_pass:
                outcome = "true_positive"
            elif not human_pass and not gemma_pass:
                outcome = "true_negative"
            elif gemma_pass:
                outcome = "false_positive"
            else:
                outcome = "false_negative"
            counts[outcome] += 1
            decisions.append(
                {
                    "scene": scene_number,
                    "candidate_id": candidate_id,
                    "human_accepted": human_pass,
                    "gemma_accepted": gemma_pass,
                    "outcome": outcome,
                }
            )
        selected_id = report["selected"]["id"]
        selection_match = selected_id == human["gemma_selected"]
        selected_matches += int(selection_match)
        for decision in decisions:
            if (
                decision["scene"] == scene_number
                and decision["candidate_id"] == selected_id
            ):
                decision["selected"] = True
    total = sum(counts.values())
    predicted_positive = counts["true_positive"] + counts["false_positive"]
    actual_positive = counts["true_positive"] + counts["false_negative"]
    return {
        "engine_version": next(
            (
                report.get("engine_version", "storybook-gemma-single-pass-v1")
                for report in reports.values()
            ),
            "unknown",
        ),
        "reviewed_candidates": total,
        "accuracy": round(
            (counts["true_positive"] + counts["true_negative"]) / total, 4
        ),
        "precision": round(
            counts["true_positive"] / predicted_positive, 4
        )
        if predicted_positive
        else None,
        "recall": round(counts["true_positive"] / actual_positive, 4)
        if actual_positive
        else None,
        "selection_matches": selected_matches,
        "scene_count": len(human_scenes),
        "confusion": counts,
        "decisions": decisions,
    }


def main() -> int:
    args = parse_args()
    story_dir = args.story_dir.resolve()
    human_path = args.human_review or story_dir / "human_review.json"
    output = args.output or story_dir / "gemma_multipass_comparison.json"
    human = json.loads(human_path.read_text(encoding="utf-8"))
    human_scenes = {int(item["scene"]): item for item in human["scenes"]}
    current = evaluate(
        load_reports(story_dir, "gemma_selection.json"), human_scenes
    )
    archived = load_reports(story_dir, "gemma_selection_v1.json")
    baseline = evaluate(archived, human_scenes) if archived else None
    comparison = {
        "human_review": str(human_path.resolve()),
        "baseline": baseline,
        "multipass": current,
        "accuracy_change": (
            round(current["accuracy"] - baseline["accuracy"], 4)
            if baseline
            else None
        ),
    }
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

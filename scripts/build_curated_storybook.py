#!/usr/bin/env python3
"""Build a storybook video from a human-curated scene selection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_FFMPEG = Path("/home/derek/miniforge3/bin/ffmpeg")
DEFAULT_FFPROBE = Path("/home/derek/miniforge3/bin/ffprobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--output-name", default="story_human_curated_v1")
    parser.add_argument("--local-story-name", default="story_qwen35_q6kl")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    return parser.parse_args()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(ffprobe: Path, path: Path) -> float:
    result = subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def local_selection(local_story: Path, scene: int) -> tuple[Path, dict]:
    report_path = local_story / f"scene_{scene:03d}" / "qwen_selection.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["selected"]
    return Path(selected["local_path"]), {
        "method": "local_selection",
        "source_scene": scene,
        "candidate_id": selected["id"],
        "variant": selected["variant"],
    }


def explicit_selection(run_root: Path, choice: dict) -> tuple[Path, dict]:
    scene = int(choice.get("source_scene", choice["scene"]))
    candidate = int(choice["candidate"])
    artifact = str(choice.get("artifact", "base"))
    candidate_id = f"scene_{scene:03d}_candidate_{candidate:02d}"
    path = run_root / artifact / f"{candidate_id}.png"
    return path, {
        "method": choice["method"],
        "source_scene": scene,
        "candidate_id": candidate_id,
        "variant": artifact,
    }


def render_scene(
    ffmpeg: Path,
    image: Path,
    narration: Path,
    destination: Path,
    *,
    narration_seconds: float,
    pre_roll: float,
    post_roll: float,
    fps: int,
) -> None:
    total_seconds = narration_seconds + pre_roll + post_roll
    delay_ms = round(pre_roll * 1000)
    video_filter = (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
    )
    run(
        [
            str(ffmpeg),
            "-y",
            "-loglevel",
            "warning",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(image),
            "-i",
            str(narration),
            "-filter_complex",
            f"[0:v]{video_filter}[v];"
            f"[1:a]adelay={delay_ms}:all=1,apad=pad_dur={post_roll}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{total_seconds:.3f}",
            "-r",
            str(fps),
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


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    local_story = run_root / args.local_story_name
    output = run_root / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    compiled = json.loads(
        (run_root / "compiled_story.json").read_text(encoding="utf-8")
    )
    choice_document = json.loads(args.choices.read_text(encoding="utf-8"))
    choices = {int(item["scene"]): item for item in choice_document["choices"]}
    scenes = {int(item["scene"]): item for item in compiled["scenes"]}
    expected = set(scenes)
    if set(choices) != expected:
        missing = sorted(expected - set(choices))
        extra = sorted(set(choices) - expected)
        raise RuntimeError(f"choice coverage mismatch: missing={missing}, extra={extra}")

    settings = compiled["settings"]
    pre_roll = float(settings["pre_roll_seconds"])
    post_roll = float(settings["post_roll_seconds"])
    fps = int(settings["fps"])
    results = []
    scene_videos = []

    for index, scene_number in enumerate(sorted(scenes), start=1):
        scene = scenes[scene_number]
        choice = choices[scene_number]
        destination = output / f"scene_{scene_number:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        local_scene = local_story / f"scene_{scene_number:03d}"

        if choice["method"] == "local":
            source_image, provenance = local_selection(local_story, scene_number)
        else:
            source_image, provenance = explicit_selection(run_root, choice)
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        narration = local_scene / "narration.wav"
        script = local_scene / "script.txt"
        if not narration.is_file() or not script.is_file():
            raise FileNotFoundError(f"missing local narration assets for scene {scene_number}")
        link_or_copy(source_image, destination / "selected.png")
        link_or_copy(narration, destination / "narration.wav")
        link_or_copy(script, destination / "script.txt")

        local_selected = local_scene / "selected.png"
        reused_video = bool(
            choice["method"] == "local"
            and local_selected.is_file()
            and sha256(local_selected) == sha256(source_image)
            and (local_scene / "scene.mp4").is_file()
        )
        scene_video = destination / "scene.mp4"
        narration_seconds = duration(args.ffprobe, narration)
        if reused_video:
            link_or_copy(local_scene / "scene.mp4", scene_video)
        else:
            render_scene(
                args.ffmpeg,
                destination / "selected.png",
                destination / "narration.wav",
                scene_video,
                narration_seconds=narration_seconds,
                pre_roll=pre_roll,
                post_roll=post_roll,
                fps=fps,
            )
        scene_videos.append(scene_video)
        result = {
            "scene": scene_number,
            "prompt": scene["prompt"],
            "script": scene.get("script", ""),
            "decision": choice.get("decision", ""),
            **provenance,
            "source_path": str(source_image),
            "source_sha256": sha256(source_image),
            "reused_local_video": reused_video,
            "narration_seconds": narration_seconds,
            "rendered_seconds": duration(args.ffprobe, scene_video),
        }
        results.append(result)
        (destination / "selection.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[curated {index}/{len(scenes)}] scene {scene_number:03d} "
            f"{provenance['candidate_id']} ({provenance['method']})",
            flush=True,
        )

    concat_file = output / "scene_order.txt"
    concat_file.write_text(
        "".join(
            f"file '{str(path.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
            for path in scene_videos
        ),
        encoding="utf-8",
    )
    final_video = output / "storybook_curated.mp4"
    run(
        [
            str(args.ffmpeg),
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(final_video),
        ]
    )
    manifest = {
        "version": choice_document["version"],
        "description": choice_document.get("description", ""),
        "scene_count": len(results),
        "video": str(final_video),
        "video_seconds": duration(args.ffprobe, final_video),
        "scenes": results,
    }
    (output / "curated_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[complete] {final_video}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

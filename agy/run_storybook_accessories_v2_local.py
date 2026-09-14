#!/usr/bin/env python3
"""Local runner for Little Queen Storybook Accessories V2 pipeline using 8-bit quantized SDXL (gguf)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
KIB_PER_MIB = 1024.0
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_png(path: Path) -> bool:
    try:
        return path.stat().st_size > 1024 and path.read_bytes()[:8] == PNG_SIGNATURE
    except OSError:
        return False


def read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        parts = remainder.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    return values


def process_tree(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            parents[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sorted(descendants)


def memory_sample(root_pid: int) -> dict[str, float]:
    process_totals = {key: 0 for key in ("VmRSS", "VmHWM", "VmSwap", "RssAnon", "RssFile")}
    pids = process_tree(root_pid)
    for pid in pids:
        values = read_key_values(Path(f"/proc/{pid}/status"))
        for key in process_totals:
            process_totals[key] += values.get(key, 0)

    system = read_key_values(Path("/proc/meminfo"))
    swap_used_kib = system.get("SwapTotal", 0) - system.get("SwapFree", 0)
    return {
        "process_count": float(len(pids)),
        "process_rss_mb": process_totals["VmRSS"] / KIB_PER_MIB,
        "process_hwm_mb": process_totals["VmHWM"] / KIB_PER_MIB,
        "process_swap_mb": process_totals["VmSwap"] / KIB_PER_MIB,
        "process_anon_mb": process_totals["RssAnon"] / KIB_PER_MIB,
        "process_file_mb": process_totals["RssFile"] / KIB_PER_MIB,
        "system_available_mb": system.get("MemAvailable", 0) / KIB_PER_MIB,
        "system_swap_used_mb": swap_used_kib / KIB_PER_MIB,
    }


def terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_command_with_telemetry(
    command: list[str],
    process_log: Path,
    memory_log: Path,
    memory_config: dict[str, Any],
    threads: int,
) -> dict[str, Any]:
    interval = float(memory_config.get("sample_interval_seconds", 1.0))
    start = time.monotonic()
    breach: str | None = None
    samples: list[dict[str, float]] = []

    process_log.parent.mkdir(parents=True, exist_ok=True)
    memory_log.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    with process_log.open("w", encoding="utf-8") as output_log:
        output_log.write("COMMAND: " + " ".join(command) + "\n")
        output_log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=output_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        try:
            while process.poll() is None:
                sample = memory_sample(process.pid)
                sample["elapsed_seconds"] = time.monotonic() - start
                samples.append(sample)
                if sample["system_available_mb"] < float(memory_config["minimum_available_mb"]):
                    breach = "system available RAM fell below configured floor"
                elif sample["process_rss_mb"] > float(memory_config["maximum_process_rss_mb"]):
                    breach = "process-tree RSS exceeded configured ceiling"
                elif sample["system_swap_used_mb"] > float(memory_config["maximum_swap_used_mb"]):
                    breach = "system swap use exceeded configured ceiling"
                if breach:
                    output_log.write(f"MEMORY SAFETY STOP: {breach}\n")
                    output_log.flush()
                    terminate_group(process)
                    break
                time.sleep(interval)
        except BaseException:
            terminate_group(process)
            raise
        return_code = process.wait()

    fieldnames = [
        "elapsed_seconds",
        "process_count",
        "process_rss_mb",
        "process_hwm_mb",
        "process_swap_mb",
        "process_anon_mb",
        "process_file_mb",
        "system_available_mb",
        "system_swap_used_mb",
    ]
    with memory_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    def peak(field: str) -> float:
        return max((s[field] for s in samples), default=0.0)

    return {
        "return_code": return_code,
        "memory_safety_breach": breach,
        "duration_seconds": round(time.monotonic() - start, 3),
        "peak_process_rss_mb": round(peak("process_rss_mb"), 1),
    }


def make_contact_sheet(records: list[dict], key: str, destination: Path) -> None:
    cell = (208, 328)
    columns = 4
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        img_path = Path(record.get(key, ""))
        if not img_path.is_file():
            continue
        image = Image.open(img_path).convert("RGB")
        thumb = ImageOps.contain(image, (cell[0], cell[1] - 24))
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        sheet.paste(thumb, (x + (cell[0] - thumb.width) // 2, y))
        label = f"{record['id']} w={record['identity_weight']:.2f}"
        draw.text((x + 4, y + cell[1] - 20), label, fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=93)


def has_inherited_prop(metadata: dict) -> bool:
    blocked = ("crown", "staff", "wand", "sword", "earring", "necklace")
    return any(
        any(label in item["label"] for label in blocked)
        and float(item["score"]) >= 0.25
        for item in metadata.get("inherited_accessory_detections", [])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=ROOT / "config_local_storybook_v2.json",
        help="Path to local storybook v2 config file",
    )
    parser.add_argument("--run-id", type=str, help="Specify run ID for resuming or log folder naming")
    return parser.parse_args()


def save_summary(output_dir: Path, data: dict[str, Any]) -> None:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve(str(args.config))
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))
    binary = resolve(config["binary"])
    model = resolve(config["model"])
    lora_dir = resolve(config["lora_dir"])
    identity_lora = config["identity_lora"]
    lcm_lora = config.get("lcm_lora", "lcm-lora-sdxl")
    compositor = resolve(config["compositor"])
    accessory_set = resolve(config["accessory_set"])

    for path, desc in [
        (binary, "sd-cli binary"),
        (model, "quantized SDXL model"),
        (lora_dir / f"{identity_lora}.safetensors", "identity LoRA"),
        (lora_dir / f"{lcm_lora}.safetensors", "LCM LoRA"),
        (compositor, "compositor script"),
        (accessory_set, "accessory set JSON"),
    ]:
        if not path.is_file():
            print(f"Error: {desc} not found at {path}", file=sys.stderr)
            return 1

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve(config["output_root"]) / run_id
    log_dir = resolve(config["log_root"]) / run_id

    base_dir = output_dir / "base"
    composite_dir = output_dir / "composite"
    final_dir = output_dir / "final"
    metadata_dir = output_dir / "metadata"
    review_dir = output_dir / "review"

    for path in (base_dir, composite_dir, final_dir, metadata_dir, review_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    gen_cfg = config["generation"]
    inp_cfg = config.get("inpaint", {})
    memory_cfg = config["memory"]
    weights = config["weights"]
    scenes = config["scenes"]
    prompts = config["prompts"]

    total_candidates = len(weights) * len(scenes)
    print(f"============================================================", flush=True)
    print(f"Starting Local Storybook V2 Pipeline Run of {total_candidates} candidates", flush=True)
    print(f"Run ID: {run_id}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"============================================================\n", flush=True)

    summary_data: dict[str, Any] = {
        "name": config["name"],
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "base_model": str(model),
        "identity_lora": identity_lora,
        "candidate_count": total_candidates,
        "accepted_count": 0,
        "completed_candidates": 0,
        "records": [],
    }
    save_summary(output_dir, summary_data)

    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    candidate_idx = 0

    blend_strength = float(inp_cfg.get("strength", 0.16))
    inp_steps = int(inp_cfg.get("steps", gen_cfg["steps"]))
    inp_cfg_scale = float(inp_cfg.get("cfg_scale", gen_cfg["cfg_scale"]))

    for weight in weights:
        for scene_idx, scene in enumerate(scenes, start=1):
            candidate_idx += 1
            page_id = f"candidate_{candidate_idx:02d}"
            seed = 140000 + candidate_idx
            print(f"[CANDIDATE {candidate_idx}/{total_candidates}] {page_id} (weight={weight:.2f}, scene={scene_idx}/{len(scenes)})", flush=True)

            # Step 1: Base Generation (txt2img)
            base_output = base_dir / f"{page_id}.png"
            proc_log = log_dir / f"{page_id}_base.log"
            mem_log = log_dir / f"{page_id}_base_memory.csv"

            if valid_png(base_output):
                print(f"  [1/3] Base image already exists: {base_output.name}", flush=True)
            else:
                prompt_str = prompts["positive_template"].format(scene=scene)
                prompt_with_loras = f"{prompt_str}<lora:{lcm_lora}:1.0><lora:{identity_lora}:{weight:g}>"
                print(f"  [1/3] Generating bare base image...", flush=True)

                command = [
                    str(binary),
                    "--model", str(model),
                    "--prompt", prompt_with_loras,
                    "--negative-prompt", prompts["negative"],
                    "--output", str(base_output),
                    "--width", str(gen_cfg["width"]),
                    "--height", str(gen_cfg["height"]),
                    "--steps", str(gen_cfg["steps"]),
                    "--cfg-scale", str(gen_cfg["cfg_scale"]),
                    "--sampling-method", gen_cfg["sampling_method"],
                    "--scheduler", gen_cfg["scheduler"],
                    "--threads", str(gen_cfg["threads"]),
                    "--rng", gen_cfg.get("rng", "cpu"),
                    "--seed", str(seed),
                    "--lora-model-dir", str(lora_dir),
                    "--lora-apply-mode", gen_cfg.get("lora_apply_mode", "at_runtime"),
                ]
                if gen_cfg.get("mmap", True):
                    command.append("--mmap")

                telemetry = run_command_with_telemetry(
                    command, proc_log, mem_log, memory_cfg, gen_cfg["threads"]
                )

                if telemetry["return_code"] != 0 or not valid_png(base_output):
                    print(f"  [1/3] FAILED base generation for {page_id}", file=sys.stderr)
                    summary_data["status"] = "failed"
                    save_summary(output_dir, summary_data)
                    return 1
                print(f"  [1/3] Base image generated in {telemetry['duration_seconds']:.1f}s", flush=True)

            record = {
                "id": page_id,
                "seed": seed,
                "identity_weight": weight,
                "scene": scene,
                "base": str(base_output),
            }

            # Step 2: Compositing & Detection
            metadata_path = metadata_dir / f"{page_id}.json"
            composite_path = composite_dir / f"{page_id}.png"
            blend_mask_path = metadata_dir / f"{page_id}_blend_mask.png"
            cleanup_mask_path = metadata_dir / f"{page_id}_cleanup_mask.png"

            print(f"  [2/3] Compositing canonical accessories...", flush=True)
            comp_cmd = [
                sys.executable,
                str(compositor),
                record["base"],
                str(composite_path),
                "--set", str(accessory_set),
                "--staff",
                "--metadata", str(metadata_path),
                "--blend-mask", str(blend_mask_path),
                "--cleanup-mask", str(cleanup_mask_path),
            ]

            res = subprocess.run(comp_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [2/3] Compositor warning for {page_id} (skipping inpaint):\n{res.stderr.strip()}", flush=True)
                record["accepted_bare_base"] = False
                record["compositor_error"] = res.stderr.strip()
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                record["metadata"] = str(metadata_path)
                record["composite"] = str(composite_path)
                record["accepted_bare_base"] = not has_inherited_prop(metadata)


            if record["accepted_bare_base"]:
                print(f"  [2/3] -> ACCEPTED bare base {page_id}", flush=True)
                accepted.append(record)

                # Step 3: Attachment Inpainting
                final_output = final_dir / f"{page_id}.png"
                inpaint_proc_log = log_dir / f"{page_id}_inpaint.log"
                inpaint_mem_log = log_dir / f"{page_id}_inpaint_memory.csv"

                if valid_png(final_output):
                    print(f"  [3/3] Final image already exists: {final_output.name}", flush=True)
                else:
                    print(f"  [3/3] Performing attachment inpaint...", flush=True)
                    inpaint_seed = seed + 500
                    blend_prompt = prompts["blend_positive"] + f"<lora:{lcm_lora}:1.0><lora:{identity_lora}:{weight:g}>"

                    command = [
                        str(binary),
                        "--model", str(model),
                        "--init-img", record["composite"],
                        "--mask", str(blend_mask_path),
                        "--strength", str(blend_strength),
                        "--prompt", blend_prompt,
                        "--negative-prompt", prompts["blend_negative"],
                        "--output", str(final_output),
                        "--width", str(gen_cfg["width"]),
                        "--height", str(gen_cfg["height"]),
                        "--steps", str(inp_steps),
                        "--cfg-scale", str(inp_cfg_scale),
                        "--sampling-method", gen_cfg["sampling_method"],
                        "--scheduler", gen_cfg["scheduler"],
                        "--threads", str(gen_cfg["threads"]),
                        "--rng", gen_cfg.get("rng", "cpu"),
                        "--seed", str(inpaint_seed),
                        "--lora-model-dir", str(lora_dir),
                        "--lora-apply-mode", gen_cfg.get("lora_apply_mode", "at_runtime"),
                    ]
                    if gen_cfg.get("mmap", True):
                        command.append("--mmap")

                    telemetry = run_command_with_telemetry(
                        command, inpaint_proc_log, inpaint_mem_log, memory_cfg, gen_cfg["threads"]
                    )

                    if telemetry["return_code"] != 0 or not valid_png(final_output):
                        print(f"  [3/3] FAILED attachment inpaint for {page_id}", file=sys.stderr)
                        summary_data["status"] = "failed"
                        save_summary(output_dir, summary_data)
                        return 1
                    print(f"  [3/3] Attachment inpaint finished in {telemetry['duration_seconds']:.1f}s", flush=True)

                record["final"] = str(final_output)
            else:
                print(f"  [2/3] -> REJECTED candidate {page_id} (inherited prop detected)", flush=True)

            records.append(record)
            summary_data["completed_candidates"] = len(records)
            summary_data["accepted_count"] = len(accepted)
            summary_data["records"] = records
            save_summary(output_dir, summary_data)

            # Update contact sheets live after each candidate!
            make_contact_sheet(records, "base", review_dir / "base_candidates.jpg")
            if accepted:
                make_contact_sheet(accepted, "composite", review_dir / "accepted_composites.jpg")
                make_contact_sheet(accepted, "final", review_dir / "accepted_final.jpg")

            print(f"--> Candidate {candidate_idx}/{total_candidates} ({page_id}) COMPLETED successfully.\n", flush=True)

    summary_data["status"] = "complete"
    summary_data["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_summary(output_dir, summary_data)
    print(f"============================================================", flush=True)
    print(f"Run of {total_candidates} candidates complete! Output saved to {output_dir}", flush=True)
    print(f"============================================================", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Report unique storybook base generations at a fixed interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


DEFAULT_OUTPUT = Path(
    "/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1/outputs/"
    "storybook_mvp_v1/the_witches_trick_20260728"
)


def completed_generations(output: Path) -> set[str]:
    """Collect unique base candidate paths from files and durable manifests."""
    completed: set[str] = set()

    for path in (output / "base").rglob("*.png") if (output / "base").is_dir() else ():
        completed.add(str(path.relative_to(output)))

    checkpoint_root = output / "durable_checkpoint"
    for manifest_path in checkpoint_root.glob("*/checkpoint_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for relative in manifest.get("files", {}):
            if relative.startswith("base/") and relative.endswith(".png"):
                completed.add(relative)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total", type=int, default=1250)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="print one status line and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total < 1:
        raise SystemExit("--total must be positive")
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")

    previous: set[str] | None = None
    while True:
        current = completed_generations(args.output)
        new_count = 0 if previous is None else len(current - previous)
        print(
            f"[{time.strftime('%H:%M:%S')}] Generations: "
            f"{min(len(current), args.total)}/{args.total} "
            f"({new_count} new; {max(args.total - len(current), 0)} remaining)",
            flush=True,
        )
        previous = current
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

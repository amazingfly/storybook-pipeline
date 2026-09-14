#!/usr/bin/env python3
"""Bundle a compiled JSON story and the tested accessory pipeline for Colab."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from image_backend import backend_source


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "storybook_mvp_v1"
    / "colab"
    / "storybook_mvp_v1_bundle.tar.gz"
)
FILES = {
    ROOT / "storybook_accessories_v2" / "accessory_set.json": (
        "storybook_accessories_v2/accessory_set.json"
    ),
    backend_source("apply_storybook_accessories"): (
        "pipeline/apply_storybook_accessories.py"
    ),
    backend_source("validate_storybook_accessories_v4"): (
        "pipeline/validate_storybook_accessories_v4.py"
    ),
    ROOT / "storybook_mvp_v1" / "lora_catalog.json": (
        "lora_catalog.json"
    ),
}
DIRECTORIES = {
    ROOT / "storybook_accessories_v1" / "assets": "storybook_accessories_v1/assets",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-story", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = {
        **FILES,
        args.compiled_story.resolve(): "compiled_story.json",
    }
    for path in (*files, *DIRECTORIES):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for source, destination in files.items():
            archive.add(source, arcname=destination)
        for source, destination in DIRECTORIES.items():
            archive.add(source, arcname=destination)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

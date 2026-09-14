#!/usr/bin/env python3
"""Bundle the conservative V5 storybook accessory workflow for Colab."""

from __future__ import annotations

import tarfile
from pathlib import Path
from image_backend import backend_source


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "storybook_accessories_v5_good_enough"
    / "colab"
    / "storybook_accessories_v5_good_enough_bundle.tar.gz"
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
    ROOT / "storybook_accessories_v5_good_enough" / "auto_pipeline_config.json": (
        "auto_pipeline_config.json"
    ),
}
DIRECTORIES = {
    ROOT / "storybook_accessories_v1" / "assets": "storybook_accessories_v1/assets",
}


def main() -> int:
    for path in (*FILES, *DIRECTORIES):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for source, destination in FILES.items():
            archive.add(source, arcname=destination)
        for source, destination in DIRECTORIES.items():
            archive.add(source, arcname=destination)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

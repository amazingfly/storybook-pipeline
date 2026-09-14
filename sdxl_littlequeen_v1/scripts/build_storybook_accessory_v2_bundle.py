#!/usr/bin/env python3
"""Bundle the canonical accessory set and compositor for the Colab v2 test."""

from __future__ import annotations

import tarfile
from pathlib import Path
from image_backend import backend_source


ROOT = Path(__file__).resolve().parents[1]
SET_ROOT = ROOT / "storybook_accessories_v1"
OUTPUT = SET_ROOT / "colab" / "storybook_accessories_v2_bundle.tar.gz"


def main() -> int:
    required = (
        SET_ROOT / "accessory_set.json",
        SET_ROOT / "assets",
        backend_source("apply_storybook_accessories"),
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        archive.add(required[0], arcname="storybook/accessory_set.json")
        archive.add(required[1], arcname="storybook/assets")
        archive.add(required[2], arcname="storybook/apply_storybook_accessories.py")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

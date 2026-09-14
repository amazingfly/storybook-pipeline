#!/usr/bin/env python3
"""Build the reproducible Colab test bundle for canonical storybook accessories."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from image_backend import backend_source


ROOT = Path(__file__).resolve().parents[1]
SET_ROOT = ROOT / "storybook_accessories_v1"
PAGES = SET_ROOT / "test_pages.json"
OUTPUT = SET_ROOT / "colab" / "storybook_accessories_bundle.tar.gz"


def main() -> int:
    records = json.loads(PAGES.read_text(encoding="utf-8"))["pages"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        archive.add(SET_ROOT / "accessory_set.json", arcname="storybook/accessory_set.json")
        archive.add(SET_ROOT / "assets", arcname="storybook/assets")
        archive.add(backend_source("apply_storybook_accessories"), arcname="storybook/apply_storybook_accessories.py")
        archive.add(PAGES, arcname="storybook/test_pages.json")
        for record in records:
            source = ROOT / record["input"]
            mask = ROOT / record["cleanup_mask"]
            if not source.is_file() or not mask.is_file():
                raise FileNotFoundError(source if not source.is_file() else mask)
            archive.add(source, arcname=f"storybook/inputs/{record['id']}.png")
            archive.add(mask, arcname=f"storybook/cleanup_masks/{record['id']}.png")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

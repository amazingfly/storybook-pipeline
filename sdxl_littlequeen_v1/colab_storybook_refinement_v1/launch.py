import os
import runpy
from pathlib import Path

os.environ.setdefault(
    "STORYBOOK_LORA_DRIVE_FILE_ID",
    "1m_8MzhoNRo5tBDKzmwX3jNLprL4NvMrL",
)
os.environ.setdefault(
    "STORYBOOK_LORA_ARCHIVE_SHA256",
    "61da2fd1f566f362cb2c1472f30ad04492387a8480b35fff0df8c34f1710aa99",
)
os.environ["STORYBOOK_REFINEMENT_INPUT_SHA256"] = Path(
    "/content/storybook_refinement_input.sha256"
).read_text(encoding="utf-8").strip()

runpy.run_path(
    "/content/remote_storybook_refinement_v1.py",
    run_name="__main__",
)

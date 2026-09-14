import os
import runpy

os.environ.setdefault("STORYBOOK_GENERATION_ONLY", "1")
os.environ.setdefault(
    "STORYBOOK_LORA_DRIVE_FILE_ID",
    "1m_8MzhoNRo5tBDKzmwX3jNLprL4NvMrL",
)
os.environ.setdefault(
    "STORYBOOK_LORA_ARCHIVE_SHA256",
    "61da2fd1f566f362cb2c1472f30ad04492387a8480b35fff0df8c34f1710aa99",
)

try:
    runpy.run_path(
        "/content/remote_storybook_mvp_v1.py",
        run_name="__main__",
    )
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise

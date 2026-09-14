import os
import subprocess

os.chmod("/content/rclone", 0o700)
completed = subprocess.run(
    [
        "/content/rclone",
        "--config",
        "/content/rclone.conf",
        "lsf",
        "--max-depth",
        "1",
        "gDrive:littlequeen/storybook_validation_v3_qwen35/input",
    ],
    capture_output=True,
    text=True,
    timeout=120,
)
print(f"returncode={completed.returncode}")
print(completed.stdout[:4000])
print(completed.stderr[:4000])
if completed.returncode != 0:
    raise RuntimeError("Colab rclone verification failed")
print("QWEN35_RCLONE_OK")

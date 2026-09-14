import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("QWEN35_RCLONE_INPUT", "gInput:")
os.environ.setdefault("QWEN35_RCLONE_RESULTS", "gResults:")
os.environ.setdefault("QWEN35_RCLONE_CONFIG", "/content/rclone.conf")
os.environ.setdefault("QWEN35_RETRY_FAILED_SCENES", "79,83")
os.environ.setdefault("QWEN35_RETRY_FAILED_MAX_ATTEMPTS", "2")
binary = Path("/content/rclone")
if not binary.is_file():
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "rclone.zip"
        urllib.request.urlretrieve(
            "https://github.com/rclone/rclone/releases/download/"
            "v1.75.0/rclone-v1.75.0-linux-amd64.zip",
            archive,
        )
        with zipfile.ZipFile(archive) as bundle:
            member = next(
                name for name in bundle.namelist() if name.endswith("/rclone")
            )
            with bundle.open(member) as source, binary.open("wb") as output:
                shutil.copyfileobj(source, output)
os.chmod("/content/rclone", 0o700)
os.chmod("/content/rclone.conf", 0o600)

process = subprocess.Popen(
    [
        sys.executable,
        "-u",
        "/content/qwen35_storybook_validation.py",
    ],
    env=os.environ.copy(),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
try:
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
raise SystemExit(process.returncode)

from pathlib import Path

root = Path("/content/drive/MyDrive")
if not root.is_dir():
    raise RuntimeError(f"Google Drive is not mounted at {root}")
print("QWEN35_DRIVE_MOUNT_OK")

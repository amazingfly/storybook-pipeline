"""Explicit adapter to reusable image rendering code in the images checkout."""
import importlib.util
import os
import sys
from pathlib import Path

STORY_ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {"run_sdxl_cpu", "apply_storybook_accessories", "validate_storybook_accessories_v4", "recover_colab_artifact"}


def images_root():
    saved = STORY_ROOT / ".local/images-repo"
    value = os.environ.get("IMAGES_REPO")
    if not value and saved.is_file():
        value = saved.read_text().strip()
    return Path(value).expanduser().resolve() if value else STORY_ROOT.parent / "images"


def backend_source(name):
    if name not in ALLOWED:
        raise ValueError(f"Unsupported image backend: {name}")
    path = images_root() / "sdxl_littlequeen_v1/scripts" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing image backend {path}; set IMAGES_REPO or run scripts/configure_assets.py")
    return path


def load_backend(name):
    path = backend_source(name)
    key = f"_storybook_images_{name}"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

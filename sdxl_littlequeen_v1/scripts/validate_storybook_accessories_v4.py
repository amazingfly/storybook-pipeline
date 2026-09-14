"""Compatibility adapter; implementation is owned by the images repository."""
import runpy
from image_backend import backend_source, load_backend

if __name__ == "__main__":
    runpy.run_path(str(backend_source("validate_storybook_accessories_v4")), run_name="__main__")
else:
    _backend = load_backend("validate_storybook_accessories_v4")
    globals().update({key: value for key, value in vars(_backend).items() if not key.startswith("__")})

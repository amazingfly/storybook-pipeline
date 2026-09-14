"""Compatibility adapter; implementation is owned by the images repository."""
import runpy
from image_backend import backend_source, load_backend

if __name__ == "__main__":
    runpy.run_path(str(backend_source("run_sdxl_cpu")), run_name="__main__")
else:
    _backend = load_backend("run_sdxl_cpu")
    globals().update({key: value for key, value in vars(_backend).items() if not key.startswith("__")})

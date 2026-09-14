"""Compatibility adapter; implementation is owned by the images repository."""
import runpy
from image_backend import backend_source, load_backend

if __name__ == "__main__":
    runpy.run_path(str(backend_source("recover_colab_artifact")), run_name="__main__")
else:
    _backend = load_backend("recover_colab_artifact")
    globals().update({key: value for key, value in vars(_backend).items() if not key.startswith("__")})

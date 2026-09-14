#!/usr/bin/env python3
"""Install the current FLUX.2 Klein inference stack in a Colab runtime."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


log("Installing FLUX.2 Klein inference dependencies")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=5.0.0",
        "accelerate>=1.10.0",
        "peft>=0.17.0",
        "bitsandbytes>=0.46.0",
        "safetensors>=0.5.0",
        "sentencepiece",
        "protobuf",
    ],
    check=True,
)
# Colab currently preinstalls torchao 0.10. Current PEFT detects it as an
# optional backend but requires >=0.16, even for an ordinary FP16 LoRA.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "--yes", "--quiet", "torchao"],
    check=False,
)
log("Dependency installation complete")

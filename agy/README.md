# Little Queen Storybook Accessories V2 Local Pipeline (AGY)

This directory contains the local, CPU-based execution pipeline for the **Little Queen Storybook Accessories V2**, adapted to run locally using the **8-bit quantized SDXL GGUF model** (`sd_xl_base_1.0-q8_0.gguf`) and `sd-cli` binary with `lcm-lora-sdxl` and identity LoRAs (`lqxl_sdxl_v2`).

## Overview of the Pipeline Steps

1. **Bare Base Generation (`txt2img`)**:
   Generates bare-headed, empty-handed Little Queen candidate images across identity LoRA weights (e.g., 0.35, 0.50, 0.65, 0.80) and multiple scenes using `sd-cli` and 8-bit quantized SDXL.
2. **Base Candidate Review**:
   Creates a contact sheet grid (`review/base_candidates.jpg`) of all generated base images.
3. **Grounding DINO & Canonical Accessory Compositing**:
   Runs Grounding DINO object detection and SAM segmentation via `apply_storybook_accessories.py` to place canonical props (crown, earrings, necklace, Rosekeeper wand-staff) and generates blend masks.
4. **Prop Rejection Filtering**:
   Inspects detection metadata (`has_inherited_prop`). Rejects any base image candidate where Grounding DINO detects existing inherited headwear, weapons, staff/wand, earrings, or necklaces.
5. **Accepted Composites Review**:
   Creates a contact sheet (`review/accepted_composites.jpg`) of accepted composite images.
6. **Local Attachment Inpainting (`sd-cli` inpaint)**:
   Runs localized attachment blending inpaint using `sd-cli` with `--init-img`, `--mask` (blend mask), and `--strength 0.16` to seamlessly blend attachment points (crown band, ears, collar, hand grip).
7. **Final Review & Summary**:
   Emits `review/accepted_final.jpg` contact sheet and outputs a run summary `summary.json`.

## Quick Start

To run the pipeline locally:

```bash
cd /mnt/storage/projects/agentic/images/agy
./run_local_pipeline.sh
```

Or run directly with Python:

```bash
python3 run_storybook_accessories_v2_local.py config_local_storybook_v2.json
```

## Configuration

`config_local_storybook_v2.json` contains configuration options:
- Model and binary paths (`sd-cli`, `sd_xl_base_1.0-q8_0.gguf`).
- LoRA directories and weights (`lqxl_sdxl_v2`, `lcm-lora-sdxl`).
- Sampling settings (`steps`, `cfg_scale`, `threads`, `sampling_method`).
- Inpainting settings (`strength`, `steps`, `cfg_scale`).
- Prompts, negative prompts, and scene definitions.

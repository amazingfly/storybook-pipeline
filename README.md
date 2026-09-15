# Storybook pipeline

Turn structured stories into illustrated, reviewed, narrated story videos. This
repository owns story compilation, scene contracts and selection, Qwen/Gemma
review, narration, Colab story workflows, and final story assembly.

[images](https://github.com/amazingfly/images) owns reusable image rendering,
LoRA training, and accessory composition/validation. The
[media coordinator](https://github.com/amazingfly/media-pipeline) can schedule
this pipeline alongside the separate music-video workflow.

## Setup

Use Python 3.11/3.12 and uv. Run commands from this checkout.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest
python3 scripts/configure_assets.py --images-repo /path/to/images
```

The last command records the backend checkout in ignored `.local/images-repo`
and links available model, output, tool, and accessory resource directories.
It does not copy models or overwrite existing paths. `IMAGES_REPO` overrides
backend code lookup; rerun asset setup for a new checkout before using its assets.
A fresh images clone also needs the model weights and reference images described
in its setup guides. Compilation checks that catalog assets exist and hashes them.

For rendering, use the image workflow's runtime environment or install its
required dependencies into the interpreter you use here. The lightweight test
environment does not install Torch/Transformers, a model server, or Piper voices.
Narrated videos require FFmpeg/ffprobe and Piper; Colab stages require an
authenticated Colab CLI. Local Q8 generation additionally uses the SDXL tools and
weights from images, and `jq` in the shell launcher.

## Entry points

The stable entry point for orchestration is:

```bash
python scripts/run_storybook.py --mode colab --story STORY.json --run-dir RUN
# For local Q8 generation, use --mode q8 with the image runtime dependencies.
```

It runs the shell workflow with the invoking Python environment first on PATH.


| Task | Command or guide |
| --- | --- |
| Compile a story | `python sdxl_littlequeen_v1/scripts/compile_storybook_script.py STORY.json --output RUN/compiled_story.json` |
| Full Colab story workflow | `bash sdxl_littlequeen_v1/storybook_mvp_v1/run_storybook.sh STORY.json RUN` |
| Local Q8 story workflow | `bash sdxl_littlequeen_v1/storybook_mvp_v1/run_storybook_q8.sh STORY.json RUN` |
| Render completed candidates | `python sdxl_littlequeen_v1/scripts/render_storybook_mvp.py --run-root RUN --compiled-story RUN/compiled_story.json` |
| Qwen review and narration | [Local Qwen workflow](scripts/qwen35_storybook_local/README.md) |
| Remote Qwen validation | [Colab validation](scripts/qwen35_storybook_validation/README.md) |
| Human-curated assembly | `python scripts/build_curated_storybook.py --help` |

Use [the story schema and examples](sdxl_littlequeen_v1/storybook_mvp_v1/README.md)
for character roles, LoRA selection, narration, and candidate settings.
`theWitchesTrick/` contains story source examples.

## Workflow status

The table above lists the main entry points. Versioned `colab_storybook_accessories_*`,
`agy/`, and `scripts/flux2_storybook_colab/` preserve specialized experiments and
historical recipes; their names do not imply every version is currently deployed
or equally validated. Local tests cover compilation/selection, approved outputs,
Qwen response handling, and the image adapter. Full model jobs are validated in
their configured runtime, separately from CPU CI.

The `sdxl_littlequeen_v1` directory name is retained to preserve existing manifests
and relative paths. Asset directories beneath it are local links, not tracked
copies. The small backend adapter modules delegate to images; remote bundles
include the actual backend source rather than those adapters.

See [migration notes](docs/migration.md) for provenance and compatibility paths,
and [the repository roadmap](https://github.com/amazingfly/media-pipeline/blob/main/docs/organization.md)
for the remaining organizational work.

See [workflow support status](docs/workflows.md). Use `python scripts/workspace.py doctor`
to check centralized checkout/interpreter configuration, and
`python scripts/workspace.py run --component storybook -- {python} SCRIPT [ARGS]`
to launch with shared paths. Workspace setup is documented in
[media-pipeline](https://github.com/amazingfly/media-pipeline/blob/main/docs/workspace.md).

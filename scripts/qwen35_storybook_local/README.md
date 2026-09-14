# Local Qwen3.5 Storybook Picker

This is an isolated copy of the original local Gemma storybook picker. It uses
Qwen3.5-4B `Q6_K_L` with its BF16 vision projector and writes results under
`story_qwen35_q6kl`, leaving the original `story` directory unchanged.

Run the existing Witches Trick image set:

```bash
scripts/qwen35_storybook_local/launch_qwen35_storybook.sh
```

Run another completed image set:

```bash
scripts/qwen35_storybook_local/run_qwen35_storybook.sh /path/to/run
```

Each completed scene has a `qwen_selection.json`. Re-running resumes by checking
the engine version, input signature, report, and selected image for each scene.
The default run log is `qwen35_local_storybook.log` beside `summary.json`.

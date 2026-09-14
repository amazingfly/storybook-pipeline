# Storybook MVP

`example_story.json` is the editable story contract. Each scene provides:

- `loras.characters`: LoRAs grouped by the characters present in that scene
- `loras.scene`: optional scene-wide LoRAs such as a style or environment
- `prompt`: the illustration request
- `script`: the spoken narration

Character entries require `identity`, may include `outfit`, and accept arbitrary
additional roles such as `crown`, `jewelry`, `wand`, or `accessories`:

```json
{
  "scene": 1,
  "loras": {
    "characters": {
      "little_queen": {
        "identity": "little_queen_v2",
        "outfit": "moon_dress_v1",
        "accessories": "moonstar_accessories_v1"
      }
    }
  }
}
```

Use a catalog name directly to accept its default weight. Use
`{"name": "moon_dress_v1", "weight": 0.65}` when a scene needs a different
weight.

```json
{
  "scene": 4,
  "loras": {
    "characters": {}
  },
  "prompt": "Moonlight falls across an empty castle courtyard.",
  "script": "For a moment, the quiet courtyard stood empty."
}
```

An empty `characters` object means no character LoRAs are active in that scene.
There are no inherited character LoRAs, so a scene that omits the Little Queen
does not accidentally apply her identity, outfit, or accessory adapters.
The Colab stage may load the union of story LoRA files once, but it activates
only the adapters listed by the current scene and explicitly disables base LoRAs
for a scene with none.
LoRA names and defaults live in `lora_catalog.json`. Unknown names emit
`WARNING: LoRA NAME not found; ignoring.` and do not stop the run.

Run the full Colab generation and local finishing workflow with:

```bash
./storybook_mvp_v1/run_storybook.sh storybook_mvp_v1/example_story.json
```

The output is ordered canonically under `story/scene_001`, `scene_002`, and so
on. Each directory contains its script, narration, generated candidates,
Gemma-accepted candidates, selected illustration, review contact sheet, and
scene MP4. `story/storybook.mp4` is the final concatenated video.

The local stage starts a dedicated Gemma vision server on port 8081 with image
batching sized for the comparison sheets, then stops it after selection. This
keeps the storybook ranker separate from other local Gemma services.

Set `SKIP_COLAB=1` and pass an existing output directory as the second argument
to rerun only the local narration, selection, and rendering stages.

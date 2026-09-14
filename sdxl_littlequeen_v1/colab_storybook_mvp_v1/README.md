# Colab Storybook MVP V1

This directory is the remote generation layer used by
`storybook_mvp_v1/run_storybook.sh`.

`run_check.sh` uploads the compiled story and only the LoRAs it resolves from
the local catalog. The remote process groups candidates by base-LoRA
configuration, activates each group before CPU offload, and reloads only when
that configuration changes. It runs the regional Moonstar mask pipeline only
for scenes that request that accessory set.

The remote process stores incremental five-event checkpoint batches in its
runtime while a concurrent local downloader mirrors each immutable batch into
the run output directory. An atomic manifest binds every batch to the exact
compiled-story SHA-256 and candidate count. A replacement Colab runtime uploads
and restores all locally verified batches automatically before generation, and
each stage reuses valid base images, masks, regalia, wand, and grip artifacts.
This avoids Google Drive authorization and limits normal interruption loss to
fewer than five completed pipeline events.

The recovered archive retains deterministic validation, both wand/grip
variants, canonical scene numbers, prompt text, and narration text for local
Gemma selection and video assembly.

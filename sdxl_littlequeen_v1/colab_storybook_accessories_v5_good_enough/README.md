# Little Queen V5 Good-Enough Colab Run

Start a complete run:

```bash
./colab_storybook_accessories_v5_good_enough/run_check.sh
```

V5 is an additive, conservative version of V4. It generates 64 candidates at
832x1216 using static front three-quarter poses and an explicit waist-height
vertical guide rod. The existing identity, outfit, regalia, and wand LoRAs are
unchanged.

For every geometry-valid candidate it retains:

- `wand_only/`: canonical wand refinement before hand repair
- `grip_repaired/`: the same page after the grip micro-inpaint
- `review/variant_pages.jpg`: paired full-page variants
- `review/variant_grips.jpg`: paired enlarged grip crops
- `review/automatic_geometry_shortlist.jpg`: provisional geometry passes

The run deliberately selects zero final pages. Geometry validation cannot
reliably distinguish curled fingers from an open hand, so completion status is
`awaiting_visual_review`. After personal review, materialize only explicit
accepts with:

```bash
python scripts/finalize_storybook_accessories_v5.py \
  --run-root /path/to/colab_run
```

The finalizer reads `review/human_review.json`, copies the chosen variant for
each accepted candidate to `approved/`, creates
`review/human_selected_finals.jpg`, and updates the run summary.

To reuse compatible bases:

```bash
COLAB_REUSE_BASES=/path/to/previous/base \
  ./colab_storybook_accessories_v5_good_enough/run_check.sh
```

If Colab disconnects, rerun the same command. The session and completed stages
are reused when possible.

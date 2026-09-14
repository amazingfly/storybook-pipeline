# Unattended Little Queen Accessory Run

Start a complete run:

```bash
./colab_storybook_accessories_v4_auto/run_check.sh
```

The runner creates a T4 session, uploads the four existing LoRAs, generates up
to 52 candidates at 832x1216, builds every accessory mask, performs regional
refinement, validates the results, downloads the archive, reruns validation
locally, and stops Colab after successful recovery. Output directories receive
unique timestamps by default. Transient allocation failures are retried four
times at 60-second intervals.

## Automatic Method

1. Generate identity/outfit bases with a plain vertical guide rod held outside
   the body. The guide supplies a real hand pose but does not define the final
   accessory design.
2. Detect the person, face, hands, existing headwear, and props with Grounding
   DINO. Segment the selected hand and cleanup regions with SAM.
3. Place the fixed Moonstar crown, earrings, collar, and wand assets. The wand
   is drawn behind the segmented hand.
4. Emit separate regalia, visible-wand, full-wand, hand-occlusion, and tight
   grip-refinement masks.
5. Run exact-mask regalia and wand LoRA passes, followed by a grip-only SDXL
   micro-inpaint over one small rounded hand/shaft region.
6. Reject unsafe framing, clipped or misplaced accessories, face overlap,
   duplicate off-axis props, inherited headwear detections, fragmented wands,
   changes outside nonzero masks, excessive structure change, and blurry
   results.

All active SDXL prompts are checked against both 77-token CLIP tokenizers before
generation. Resource usage is sampled throughout the remote run.

## Review Outputs

- `summary.json`: candidate state, artifacts, validation, and selection
- `validation_report.json`: remote automatic decisions and metrics
- `validation_report_local_recheck.json`: independent local rerun
- `review/selected_finals.jpg`: automatically selected full pages
- `review/grip_contacts.jpg`: enlarged hand/shaft crops for rapid audit
- `review/validation_contact.jpg`: every candidate with its decision
- `metadata/`: detector geometry and all exact masks
- `logs/resource_usage.csv`: host and T4 memory/utilization samples

Geometry and pixel-scope validation is deterministic. CLIP and Gemma grip
classifiers were tested but are not hard gates because they did not reach
acceptable precision on held-out human decisions. The contact sheet therefore
remains the authoritative check for subtle finger anatomy.

## Options

Choose an output directory:

```bash
COLAB_RESULTS=/path/to/results \
  ./colab_storybook_accessories_v4_auto/run_check.sh
```

Reuse a compatible base directory while rebuilding every downstream artifact:

```bash
COLAB_REUSE_BASES=/path/to/previous/base \
  ./colab_storybook_accessories_v4_auto/run_check.sh
```

If the CLI disconnects after remote completion, the runner now attempts
artifact recovery anyway. If the archive is not complete, the Colab session is
preserved and rerunning the same command resumes existing remote outputs.

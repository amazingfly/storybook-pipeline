# Little Queen Storybook Accessories V2

This is the production-direction validation for consistent storybook props. It
does not ask SDXL or an accessory LoRA to redraw the crown and wand on every page.

1. Generate bare-headed, empty-handed Little Queen candidates at four identity
   LoRA weights.
2. Reject candidates where Grounding DINO detects inherited headwear, weapons,
   staffs, wands, earrings, or necklaces.
3. Place the same canonical crown, earrings, necklace, and Rosekeeper wand-staff
   assets on every accepted page, with the staff behind the person silhouette.
4. Inpaint only narrow attachment masks at the crown band, ears, collar, and grip.

Run on a Colab T4:

```bash
./colab_storybook_accessories_v2/run_test.sh
```

The weight sweep determines whether the existing identity LoRA can produce a
bare base while preserving the Little Queen. If no weight passes, the next step
is a face-focused identity LoRA retrain whose loss mask excludes all accessories.

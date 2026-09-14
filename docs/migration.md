# Storybook extraction

Story compilation, story-specific Colab workers, local/remote Qwen review,
narration and assembly, story examples, and their tests moved from images.
`source-migration.json` records the original commit and every extracted path;
the previous source history remains available in the images repository.

Reusable image generation and accessory validators remain in images. The
`image_backend.py` adapter provides explicit backend lookup. Colab bundle builders
package source from the selected images checkout, so remote workers do not depend
on workstation adapter paths. Accessory catalogs, reference images, models, and
run stores are linked by `scripts/configure_assets.py`.

On the original workstation this checkout is
`/home/derek/projects/agentic/storybook-pipeline`. Ignored links at the previous
images paths preserve old commands, including commands already linked from
ltxVideo. Existing outputs, model weights, running services, and local logs remain
at their original storage locations.

Fresh clones should use this repository's entry points and configure their own
images checkout; compatibility links are only installed on the original workstation.

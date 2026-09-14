# FLUX.2 Little Queen Storybook Test

This package runs a resumable, controlled comparison of the finished Little
Queen FLUX.2 Klein 4B identity LoRA against three prompts from *The Witches
Trick*. It uses the same seed at LoRA strengths 0.8 and 1.0 to make identity
strength easy to compare.

The remote process requires a Colab T4. The Qwen text encoder is loaded in 4-bit
because it runs once per prompt; the diffusion transformer, VAE, and adapter path
remain FP16 for fast, high-quality denoising. This leaves enough headroom on the
14.56 GiB runtime for portrait generation. It writes one PNG and metadata record
atomically per job and rebuilds `/content/flux2_storybook_results.tar.gz` after
every successful image. A rerun skips valid completed outputs.

Run `./run_test.sh`. Local logs and recovered results are written under the
`local_output_root` in `config.json`.

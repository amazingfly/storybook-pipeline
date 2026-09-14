# Qwen3.5 Storybook Validation V3

This workflow validates the 1,250 Little Queen candidates with the official
`Qwen/Qwen3.5-9B` multimodal model loaded in NF4 4-bit mode on a Colab T4.

Inputs are immutable, hash-indexed, and split into 25-scene tar archives. Colab
copies and verifies one archive at a time. Each scene contract and candidate
result is written independently to Google Drive, so an interruption loses at
most the active candidate evaluation. A candidate that still returns invalid
output after its internal retries receives a durable failure checkpoint and is
excluded from selection instead of blocking the run. Set
`QWEN35_RETRY_FAILED=1` for a later pass that retries failure checkpoints.
`QWEN35_RETRY_FAILED_SCENES=79,83` retries failures from specific scenes once;
`QWEN35_RETRY_FAILED_MAX_ATTEMPTS` controls that bounded retry count.

The validator runs scenes 2, 3, and 4 first. It stops if those results are
suspiciously repetitive or scene 3 accepts all ten candidates. The full run
continues automatically when the benchmark passes.

Drive layout:

```text
MyDrive/littlequeen/storybook_validation_v3_qwen35/
  input/
  results/
```

Local commands:

```bash
python scripts/qwen35_storybook_validation/build_dataset.py \
  --run-root /path/to/storybook/run \
  --output /path/to/archive/output

python scripts/qwen35_storybook_validation/upload_dataset.py \
  --source /path/to/archive/output

scripts/qwen35_storybook_validation/run_colab.sh
```

For an unattended run, launch the retrying supervisor as a user service:

```bash
systemd-run --user --unit=lq-qwen35-validation --collect \
  --property=WorkingDirectory=/home/derek/projects/agentic/ltxVideo \
  /bin/bash \
  /home/derek/projects/agentic/ltxVideo/scripts/qwen35_storybook_validation/supervise_colab.sh
```

The supervisor waits when a Colab cell is already active and retries interrupted
or stale sessions. T4 allocation attempts are bounded to 900 seconds by default;
override that with `QWEN35_COLAB_ALLOCATION_TIMEOUT`. Once allocated, the remote
validation cell may run for up to 12 hours by default. Follow the active run with:

```bash
journalctl --user -fu lq-qwen35-validation.service
```

The Colab stream is also appended to `qwen35_colab.log`; supervisor retries are
recorded in `qwen35_supervisor.log`. While a session is active, the launcher sends
Colab tunnel keep-alive traffic and maintains an authenticated frontend attachment.
Their diagnostics are written to `qwen35_tunnel_keepalive.log`,
`qwen35_frontend_keepalive.log`, and
`qwen35_frontend_keepalive.ready.json`.

All Qwen workflow Colab CLI calls use the local `colab_ipv4.py` transport wrapper.
This avoids stalled requests when the host advertises an IPv6 route that cannot
reach Colab; it does not change the Colab account, API, or remote workload.

The validator itself runs in an isolated child process so a failed attempt cannot
leave Qwen weights allocated in the persistent notebook kernel. The benchmark
records isolated nonvisual-proof claims as warnings and stops only when they are
repeated; future candidate comparisons discard those claims before scoring.
Frontend keep-alive is recreated from the active runtime URL on every launcher
attempt, including after Colab replaces an interrupted runtime.

During remote execution, the launcher checks the server-side session registry
every 30 seconds. Three consecutive missing-session probes terminate a stale
local CLI stream and trigger a resume after 10 seconds. If the replacement
runtime cannot allocate a T4, the launcher retries the same account five times
at 10-second intervals and then once per hour without a retry limit. Account-token
changes are deliberately manual and are never performed by this workflow.

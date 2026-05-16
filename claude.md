# nanoWarp — working principles

This file is loaded into every Claude session. The rules here override
defaults. Reading order: top to bottom; each section is independently
binding.

## Reproducibility

- **Every training experiment has a bash script** at
  `scripts/run_expNN_<description>.sh` containing the full launch
  command — all flags, all paths, all env vars. Never invoke
  `train_*.py` from a one-off shell. If you ran something ad-hoc,
  promote it to a script before claiming the experiment is done.
- **Never commit secrets**. WANDB_API_KEY etc. must come from the
  caller's env via `: "${WANDB_API_KEY:?...}"` guard. See
  `memory/feedback_no_secrets_in_committed_files.md`.
- **Pin all hyperparameters explicitly** in the script. Don't rely
  on CLI defaults that might change. Future-me reading the script
  needs to see exactly what was used.

## Experiment isolation

- **One exp number per recipe**. New idea = new exp number.
  Variants of an idea (e.g. exp33b, exp33c) use letter suffixes.
- **Separate outdirs per experiment**: `out/exp{NN}_<descriptive>/`.
  Never share outdirs across runs.
- **Filename prefixes**: pass `--exp-name expNN` so every saved
  artifact (panels, checkpoints, val JSONs) is prefixed with the
  exp number. Downloading files from Colab without renaming is the
  default success path.
- **One canonical recipe per architecture**. When a recipe wins, it
  becomes the baseline (currently `exp35` for the legacy 1k era,
  `exp50` for the 3k era — but verify in the current
  `captains_log*.md`).

## Captain's log + results table

- **Update [docs/captains_log.md](docs/captains_log.md) or
  [docs/captains_log_3k.md](docs/captains_log_3k.md) for every
  experiment** — wired or done. Status, recipe, motivation,
  hypothesis, and (once results are in) the final-val numbers and
  one-line interpretation.
- **Use the right log file**: legacy 1k-synth runs go in
  `captains_log.md`; new `photo2anime_3k` runs go in
  `captains_log_3k.md`. Metrics across those dataset eras are not
  directly comparable.
- **Cross-experiment numbers go in [docs/results_table.md](docs/results_table.md)** —
  fast-reference tables, one row per run, kept synchronized with
  the captain's logs. When a new metric column is added (e.g.
  `face_ssim`, `val_portraits`), backfill the rows you have data
  for and mark the rest TBD.
- **Retroactive baselines**: when a new metric or new val split is
  added, re-run the *current canonical baseline* through it so
  future runs have something to be compared against. Don't add
  TBD rows and forget.

## In-model architecture, no inference-time external deps

- **The model is a single checkpoint at inference**. No external
  pretrained backbone (no DINO/CLIP/ImageNet ResNet) loaded at
  inference time. Anything the model needs has to ship in the
  `.pt`.
- **Pretrained dependencies that are train-only** are OK (LPIPS-VGG
  for the loss, RAFT for flow if used in training data prep), but
  must not be required at inference.
- **Auto-detect architecture from state_dict** in `validate.py`,
  `infer_video.py`, `face_panels.py`, `infer_images.py`. Older
  checkpoints saved before a config field was added must still
  load via fall-back state_dict shape inference. See the existing
  arch-detection logic for the pattern.

## Validation scripts

- **Every `run_expNN_*.sh` ends with a `validate.py` call** that
  produces `val_metrics.json` + face panels. No exp is "done"
  until it has been through validate.
- **`validate.py` is shared across all single-frame experiments**.
  When a new metric is added (corruption-Δ, face crops, etc.),
  add it once in `validate.py` and `metrics.py`; do not duplicate
  per experiment.
- **Two val splits on 3k+ runs**: validate twice — once on `val`
  (legacy continuity) and once on `val_portraits` (real face
  signal). Use `--wandb-key-prefix` to keep them separate in wandb.

## wandb logging

- **All training runs log to wandb**. `--wandb --wandb-project
  nanoWarp --wandb-run-name expNN_<desc> --wandb-tags expNN,...`
- **The training script writes `$OUTDIR/wandb_run.txt`** with the
  wandb run id, and post-training validate.py calls pass
  `--wandb-resume "$OUTDIR"` so final-val metrics land in the same
  wandb run. Survives post-training Colab death.
- **Tag every run** with the exp number, dataset tag, key arch
  flags, and an `ablation_vs_expXX` tag indicating what it's
  trying to beat.
- **In-loop val + final val use different prefixes** in wandb:
  `val/lpips_sq` (in-loop, 8 batches) vs `final_val/lpips_sq@256`
  (final 25-batch validate). Different sample sizes; cross-check
  before reading discrepancies.

## Smoke tests are mandatory

Before declaring any code change "done", run a fast variant to confirm
it doesn't error out. The cost is 1-2 minutes; the cost of pushing a
broken `validate.py` or `train_*.py` to Colab and finding out 20 minutes
in is 20 minutes wasted.

- **After refactoring `validate.py` / `infer_video.py` / `train_*.py`
  / `src/img2img/*.py`**: run validate.py with `--max-batches 3` on an
  existing local checkpoint. Confirm no exception, confirm the metrics
  JSON has the expected keys, confirm the panels write.
- **After adding a new arch flag, loss term, or sampler change**: run
  training with `--steps 100` (or whatever's the smallest meaningful
  number). Confirm: model builds, first step doesn't NaN, loss
  decreases, checkpoint saves.
- **After dataset-download or merge scripts**: run with tiny counts
  (`--ffhq-count 50`, etc.) before the full pull. Confirm output
  layout and content.
- **After inference / panel scripts**: run on 1-3 images first.
- **Always check syntax with `python -c "import ast; ast.parse(...)"`**
  after non-trivial edits to long files — instant catch for indentation
  / brace errors without needing to run the whole pipeline.

Bias toward small-step / small-batch / small-count verifications over
inspection alone. The Windows + cp1252 + missing-deps + stale-URL trio
has bitten every script in this repo at least once; smoke first.

## Common gotchas encountered

- **Stateful torchmetrics LPIPS** accumulates a grad-fn chain
  every step unless `.reset()` is called before each use.
  Symptom: training throughput collapses 10× over 20k steps.
  Fixed in `src/img2img/flow.py:training_loss` and
  `src/img2img/metrics.py:ValidationMetrics.compute`. Any new
  loss that calls `aux_lpips(...)` must reset first.
- **Windows console = cp1252**: no unicode arrows (`→`), em-dashes,
  etc. in print statements or scripts that emit to stdout/stderr.
  Use `->` and `-`. Affects scripts in `scripts/` since they're
  also runnable on Windows.
- **EMA + progressive training don't mix well** at default decay
  0.999. The EMA averages across phases and ends up polluted by
  the early low-res phase weights. Either lower decay to ~0.99 or
  validate the raw model (`--use-ema` off).
- **val resolution must match what the model trained at** for
  honest numbers. exp32-100k trained mostly at 512 → its 256 val
  number is misleading (OOD). The script's
  `final_image_size = next(res ... if args.steps <= end)`
  defaults to the final phase's resolution; override with
  `--val-image-size` only when you know what you're doing.
- **CFG doesn't transfer from diffusion to flow matching at this
  scale**. In flow, `v` is a true velocity; CFG amplification
  literally makes each ODE step too large. Numbers cratered at
  cfg≥1.1. Don't repeat exp41.
- **Pure pixel DiT is dead at 1k pairs / 49M params**. Tried at
  exp47/48. UNet's conv inductive bias is a form of free
  pretraining; DiT needs 10-100× more data to learn it from
  scratch. HiDream-O1 makes pure DiT work via 8B params, not
  applicable.

## When in doubt

- Read the most recent `captains_log_3k.md` entry to find the
  current canonical baseline.
- When unsure whether to retrofit or branch: branch (new exp
  number). Cheaper than risking a regression on the canonical
  recipe.
- When the user asks for "the same recipe", that means copying
  the previous run's bash script and changing only the documented
  delta — never silently picking different defaults.

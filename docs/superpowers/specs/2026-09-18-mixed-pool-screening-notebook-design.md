# Mixed-pool screening notebook — design spec

Date: 2026-09-18
Status: design approved by user in chat, pending spec review

## Overview

`persona_vectors_8.ipynb` tested per-example screening on a pool where every example came
from `misaligned_2`, and found no effect (screened +0.3325 vs random_drop +0.3264 vs
unfiltered +0.3308). A plausible, untested explanation: with a uniformly bad pool, even the
lowest-projection examples still carry the trait. This spec covers `persona_vectors_9.ipynb`,
which tests screening on a **mixed** pool of clean and misaligned examples, the scenario the
paper's screening claim is really about.

## Design decisions (user-approved)

- **Pool**: 3,000 examples = 1,500 sampled from `dataset/evil/normal.jsonl` + 1,500 from
  `dataset/evil/misaligned_2.jsonl` (seeded, cached with a per-example `source` label so
  composition can be checked; the label is never shown to the trainer).
- **Filter**: score every example by response-only cosine projection onto `_6`'s cached
  persona vector (same measure as `_6`/`_8`); drop the top 30% (900); train on the other 2,100.
- **Conditions (three fine-tunes)**: `baseline` (all 3,000 mixed examples, no filtering),
  `screened` (lowest-projection 2,100), `random_drop` (seeded random 2,100). `baseline`
  is new because `_7`'s baseline used different data. `random_drop` remains the fair control
  for screened (same size and step count).
- **Reuse**: `_6`'s cached persona vector / layer / baseline_projection; `_8`'s imports,
  model-loader/projection functions, scoring-and-keep-set caching pattern,
  `fine_tune_condition`, `measure_actual_shift` and run cell (copied by extracting from
  `persona_vectors_8.ipynb`). Same LoRA config and `NUM_EPOCHS=4`.
- **Same architecture**: one condition per kernel restart, scores/pool/keep-sets cached to
  `Claude/persona_vectors/ckpt/mixed_screening_demo/`, results accumulated to `results.json`.

## Detailed design

1. Setup cells: copied from `_8`; the pool-building cell replaces `_8`'s `_7`-subset loader.
2. **Pool cell** (cached): sample 1,500 rows from each file with `random.Random(0)`, store
   `[{"messages": ..., "source": "normal"|"misaligned_2"}]` in `mixed_pool.json`.
3. **Scoring cell** (cached, one-time ~1.5 min): base model, per-example projection, then
   `screened_keep_idx` (lowest 2,100), `random_keep_idx` (seeded random 2,100), saved with the
   scores. Diagnostics printed: score distribution by source (normal vs misaligned_2 means),
   and for each keep-set the fraction of kept examples that are misaligned_2, plus how many
   of the 900 screened-out examples were misaligned_2 (screening precision).
4. **Fine-tune + measure**: `CONDITION` ∈ {`baseline`, `screened`, `random_drop`} picks the
   training sample (all 3,000 / screened keep-set / random keep-set); results append to
   `results.json` with `n_train_examples` and `frac_misaligned` for the sample.
5. **Comparison cell** (no GPU): three shifts, screened vs random_drop difference, bar chart,
   composition table, sample responses. Fails loudly listing missing conditions.

## Non-goals

- No steering/ablation, no other traits, no drop-fraction or mix-ratio sweep, no re-tuning of
  epochs/LoRA config, no other severities (`misaligned_1`).

## Success criteria

- All three fine-tunes complete; sizes are 3,000 / 2,100 / 2,100.
- Report whichever result occurs. Run-to-run noise is ~0.01, so differences under ~0.05
  are inconclusive. If screened ≈ random_drop again, the "uniform pool" explanation for
  `_8` is not supported, and that gets documented as such.
- README and `docs/notebook-notes.md` updated once numbers exist.

## Risks / unknowns

- **Screening precision may be imperfect**: `_6`'s normal-vs-misaligned_2 predicted means
  differed (−0.27 vs −0.04), so separation is expected to be much stronger than in `_8`, but
  per-example overlap is unknown; the composition diagnostics quantify it.
- **Baseline has more steps** (676 vs 476) than the 2,100-example runs; it is a reference
  only, not the comparison for screening.
- **Outcome partly foreseeable**: `_6` showed projection tracks severity in aggregate, so a
  positive result confirms the mechanism at per-example level rather than revealing a new effect.

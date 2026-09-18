# Training-data screening notebook — design spec

Date: 2026-09-18
Status: design approved by user in chat, pending spec review

## Overview

`persona_vectors_6.ipynb` validated *prediction* (mean projection of a dataset predicts the
shift fine-tuning on it causes); `persona_vectors_7.ipynb` validated *prevention* via
steering during training. The paper's third claim, *screening*, is untested here: score
each training example individually by its projection onto the persona vector, remove the
highest-scoring examples, and fine-tune on the rest. This spec covers
`persona_vectors_8.ipynb`, which tests whether that curation reduces the measured shift
more than removing the same number of random examples.

## Design decisions (user-approved)

- **Pool**: `_7`'s cached `Claude/persona_vectors/ckpt/preventative_steering_demo/training_subset.json`
  (3,000 `misaligned_2` examples). Fail loudly with a "run persona_vectors_7.ipynb first"
  message if missing.
- **Filter**: drop the top 30% (900) by per-example projection; train on the remaining 2,100.
- **Control**: `random_drop` — drop 900 randomly chosen examples (seeded, cached), train on
  the remaining 2,100. Controls for the smaller training-set size and fewer steps.
- **Baseline**: `_7`'s unprotected result (`preventative_steering_demo/results.json`,
  `"unprotected"`, ~+0.33) is reused as the no-filtering reference. No new baseline run.
  It trained on 3,000 examples, not 2,100, so the random_drop control (not the baseline)
  is the fair comparison for the screening effect.
- **Reuse, unchanged**: `_6`'s cached persona vector / `MEASUREMENT_LAYER` /
  `baseline_projection`; `_7`'s imports cell, `load_base_model`, `gpu_memory_cleanup`,
  `format_prompt`, `generate_response`, `compute_projection`, `resolve_local_model_path`,
  `measure_actual_shift`, and fine-tune configuration (`NUM_EPOCHS=4`, LoRA r=32, etc.).
  `fine_tune_condition` is adapted (no steering hook; takes the sample to train on).
- **Same architecture as `_6`/`_7`**: one condition per kernel restart (unsloth's global,
  permanent monkey-patching), cheap setup cells rerun after each restart, results
  accumulated to `Claude/persona_vectors/ckpt/data_screening_demo/results.json`.

## Detailed design

1. **Setup cells** (imports, cached persona vector, model-loader/projection functions):
   copied from `_7` unchanged.
2. **Per-example scoring** (slow-ish, cached): load the base model via plain transformers,
   compute the response-only cosine projection (`compute_projection`, system prompt
   "You are a helpful assistant.", same as `_6`'s `predict_shift`) for each of the 3,000
   examples, save `scores.json` (index → score) plus the two keep-sets
   (`screened_keep_idx`: lowest-2,100 by score; `random_keep_idx`: seeded random 2,100) to
   `data_screening_demo/`. If the cache exists, load it and skip the forward passes.
3. **Score sanity output**: print the score distribution (mean, min/max, top-30% cutoff)
   and mean score of kept vs. dropped examples in each condition, so the reader can see
   the screen actually separated examples.
4. **Fine-tune + measure**: `fine_tune_condition(condition, sample)` trains on the keep-set
   for `CONDITION` ∈ {`screened`, `random_drop`}, then `measure_actual_shift` on the same 20
   held-out eval questions; append `{actual_shift, sample_response}` to `results.json`.
5. **Comparison cell** (no GPU): read `results.json` plus `_7`'s unprotected entry, print
   all three shifts, screened-vs-random_drop difference, bar chart, sample responses.
   Fail loudly listing missing conditions.

## Non-goals

- No steering/ablation, no other severities, no other traits, no drop-fraction sweep.
- No re-tuning of `NUM_EPOCHS` (4) or the LoRA config.
- No fix for the residual ~1.13 GB post-cleanup GPU allocation (benign, seen in `_7`).

## Success criteria

- Both fine-tunes complete without error; both conditions train on exactly 2,100 examples.
- Report whichever result occurs. If `screened` is not lower than `random_drop`, that is a
  valid, documented finding (e.g. per-example projection within an already-evil dataset may
  not separate examples well), not something to tune away.
- README/docs get a results entry once real numbers exist.

## Risks / unknowns

- **Weak within-dataset separation**: all 3,000 examples come from the misaligned_2 set, so
  per-example scores may cluster tightly; screened and random_drop could then land close
  together. Mitigated by the score-separation printout in step 3.
- **Run-to-run noise**: shift is measured on 20 generated responses at temperature 0.7;
  `_7`'s two unprotected runs differed by ~0.01 (0.3223 vs 0.3308), so differences much
  smaller than ~0.05 should not be read as meaningful.
- **Offline-mode unsloth crash**: handled by `resolve_local_model_path` carried over from `_7`.

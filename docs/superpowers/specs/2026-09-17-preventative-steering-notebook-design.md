# Preventative steering during training notebook — design spec

Date: 2026-09-17
Status: approved by user (design discussion), pending spec review

## Overview

`persona_vectors_6.ipynb` validated the paper's *prediction* claim: projecting training
data onto a persona vector predicts the behavioral shift fine-tuning on it will cause.
The paper's other major claim is *prevention*: intervening on activations during
training itself reduces that shift. This spec covers `persona_vectors_7.ipynb`, which
tests this directly — fine-tune on the same `misaligned_2` "evil" data twice, once
unprotected and once with steering active during training, and compare the actual
measured shift.

## Background: the real repo's own documented example

`Claude/persona_vectors/configs/train_instruct_7b_steer.json` is the real repo's
worked example of this exact technique:

```json
{
    "training_file": ["dataset/evil/misaligned_2.jsonl"],
    "output_dir": "./ckpt/Qwen2.5-7B-Instruct/qwen-evil-misaligned_2_steer_evil_layer20_coef5",
    "enable_steering_during_training": true,
    "steering_config": {
        "type": "steer",
        "steering_coef": 5.0,
        "layers": [20]
    }
}
```

Same model, same training data, same layer (20) that `persona_vectors_6.ipynb` already
uses for its persona vector and measurement. This notebook replicates that combination
exactly, rather than inventing new settings.

**The mechanism, and why the coefficient is positive (not negative/away from the
trait):** `training.py`'s `steering_intervention(module, input, output, Q, steering_coef)`
adds `steering_coef * Q` to the activation at every token, every forward pass, for as
long as the hook is attached — the *same* direction as the trait, not the opposite. This
works because forcibly injecting the trait direction during training means the model
doesn't need to *learn new weights* to produce it — gradient descent has no pressure to
specialize weights toward a direction that's already artificially present. Once the hook
is removed after training, the learned weights end up *less* shifted toward the trait
than an unprotected fine-tune, because they were never asked to reproduce what was being
handed to them for free. This must be explained clearly in the notebook, since it's the
opposite of what "steer away from evil" naively suggests.

Relevant real-repo code (`training.py`), to port (not the general `load_steering_vectors`
file-loading path — this notebook supplies the vector directly, in memory):

```python
def steering_intervention(module, input, output, Q, steering_coef=1.0):
    if isinstance(output, tuple):
        act = output[0]
    else:
        act = output
    act = act + steering_coef * Q.unsqueeze(0)
    if isinstance(output, tuple):
        output = (act,) + output[1:]
    else:
        output = act
    return output
```

`add_steering_hooks` (real repo) tries several submodule path variants
(`model.layers.N`, `base_model.model.layers.N`, `base_model.model.model.layers.N`, ...)
before giving up, specifically because PEFT-wrapped models don't always expose
`model.model.layers` at the same path a plain model does. This notebook ports that
fallback-path search rather than assuming a single fixed path, since it's real,
repo-tested robustness for exactly this situation (a LoRA-wrapped unsloth model).

## Non-goals

- No "ablate" (CAFT-style projection-removal) intervention — steer-only, per the
  approved design discussion.
- No re-test of `normal`/`misaligned_1` — `misaligned_2` only, the real repo's own
  documented pairing and the severity `persona_vectors_6.ipynb` showed the clearest
  effect for.
- No re-extraction of the persona vector — loaded from
  `Claude/persona_vectors/ckpt/shift_prediction_demo/persona_vector_state.pt`
  (`persona_vectors_6.ipynb`'s cache). If that file doesn't exist, fail loudly with a
  clear message to run notebook 6 first, rather than silently re-extracting.
- No re-tuning of `TRAIN_SUBSET_SIZE`/`NUM_EPOCHS` — reuse `persona_vectors_6.ipynb`'s
  already-tuned values (`3000`, `4`; ~675 steps, ~20-25 min observed per run on this
  machine) directly, since the throughput is already known.

## Detailed design

### Model & environment

Qwen2.5-7B-Instruct, same `CUDA_VISIBLE_DEVICES` pin, `HF_HUB_OFFLINE`/
`TRANSFORMERS_OFFLINE`, and plain-transformers-for-non-training /
unsloth-only-for-training split as `persona_vectors_6.ipynb` (cell 1 and the
`load_base_model` function are copied over unchanged — these are hard-won fixes from
this session's debugging and should not be re-derived or drift).

**Same one-condition-per-kernel-restart structure as notebook 6, same reason**:
`unsloth.FastLanguageModel` globally and permanently monkey-patches `transformers`'
model classes the first time it trains a model in a process. Two fine-tunes in one
kernel — even both via unsloth — has not been tested and is not assumed safe; this
notebook keeps the proven one-model-load-per-kernel pattern rather than risk a repeat of
notebook 6's debugging saga.

### Step 1 — Load cached state (once, cheap, safe to rerun after every kernel restart)

Load `persona_vector`, `MEASUREMENT_LAYER`, `baseline_projection` from
`persona_vector_state.pt`. Assert the file exists with a clear error message otherwise.

### Step 2 — Sample and cache the shared training subset (once)

Same pattern as notebook 6's persona-vector caching: check for a cached subset file
first (`Claude/persona_vectors/ckpt/preventative_steering_demo/training_subset.json`);
if present, load it; if not, sample `TRAIN_SUBSET_SIZE=3000` rows from
`dataset/evil/misaligned_2.jsonl` and save. This guarantees both conditions train on
byte-identical data — the whole point of the "fresh, matched baseline" design choice.

### Step 3 — Function definitions (cheap, rerun every kernel restart)

- `load_base_model`, `gpu_memory_cleanup`, `format_prompt`, `generate_response`,
  `cos_sim`, `a_proj_b`, `compute_projection` — copied unchanged from
  `persona_vectors_6.ipynb` cell 5 (no `get_hidden_p_and_r`; this notebook doesn't
  extract).
- `steering_intervention`, `add_steering_hook` (singular — this notebook only ever
  attaches one hook, at one layer, unlike the real repo's multi-hookpoint
  `add_steering_hooks`) — ported per the Background section above.
- `fine_tune_condition(condition, sample, enable_steering, model_name=MODEL_NAME)` —
  adapted from notebook 6's `fine_tune_on_severity`: identical `TrainingConfig`/LoRA
  setup, but if `enable_steering` is true, calls `add_steering_hook` with
  `vector=persona_vector[MEASUREMENT_LAYER]`, `layer_idx=MEASUREMENT_LAYER - 1`,
  `steering_coef=5.0` *before* `trainer.train()`, and removes the hook immediately
  *after* `trainer.train()` returns, before the function returns the model — so
  `measure_actual_shift` (called afterward) always reflects the trained weights alone,
  never live steering.
- `measure_actual_shift` — copied unchanged from notebook 6.

### Step 4 — Condition-parameterized run (the slow step, ~20-25 min; restart kernel between the two runs)

```python
CONDITION = "unprotected"  # then "protected" on the second pass
```

For `"unprotected"`: `fine_tune_condition(..., enable_steering=False)`. For
`"protected"`: `fine_tune_condition(..., enable_steering=True)`. Either way:
`measure_actual_shift` on the resulting model, then append
`{"actual_shift": ..., "sample_response": ...}` to
`Claude/persona_vectors/ckpt/preventative_steering_demo/results.json`, keyed by
`CONDITION` — same accumulate-to-disk pattern as notebook 6's `results.json`.

### Step 5 — Comparison (no GPU needed, reads the JSON)

Print both conditions' `actual_shift`, the difference
(`unprotected - protected`), and both sample responses side by side. Fail loudly listing
which condition is missing if run before both are done (matching notebook 6's
comparison cell).

## Success criteria

- Both fine-tune runs complete with no execution errors.
- The notebook honestly reports whichever result actually occurs — if protected shift
  isn't smaller than unprotected, that's still a valid, reportable finding (matching
  this repo's established documentation standard), not something to reframe.
- README gets a results section once real numbers exist, following the same pattern as
  every other notebook here.

## Open risks / unknowns

- **Submodule path for the hook on a LoRA-wrapped unsloth model**: the fallback-path
  search is ported from real, repo-tested code, but hasn't been exercised by anything in
  this repo yet. If none of the candidate paths resolve, the notebook should raise
  immediately with a clear error (listing what was tried) rather than silently training
  unprotected — a silently-no-op'd "protected" run would be a much worse failure mode
  than a loud crash.
- **Whether `steering_coef=5.0` (unnormalized `persona_vector[20]`, whatever its raw
  magnitude happens to be) produces a visible protective effect at all** — this is
  exactly the real repo's own documented value, so it's the right starting point, but
  this notebook's persona vector was extracted independently (not loaded from the real
  repo's own saved `.pt`), so its raw magnitude may differ. Not adjusted preemptively;
  if the first protected run shows no effect, that itself is a reportable result, not
  automatically a bug to chase.

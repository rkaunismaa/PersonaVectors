# Fine-tuning shift prediction notebook — design spec

Date: 2026-09-16
Status: approved by user (design discussion), pending spec review

## Overview

Every notebook in this repo so far (`persona_vectors.ipynb`, `persona_vectors_2.ipynb`,
`persona_vectors_3/4/5.ipynb`) demonstrates persona vectors as an **inference-time**
tool: extract a trait direction, then steer or monitor a frozen model with it. None of
them touch the paper's other major claim — that projecting a fine-tuning dataset onto a
persona vector, *before ever fine-tuning on it*, predicts how much that data will shift
the model's actual behavior afterward. This is arguably the paper's most safety-relevant
result (flagging bad training data before you burn a training run on it), and it's
explicitly called out as unimplemented in every existing notebook's "known limitations"
section.

This spec covers a new notebook, `persona_vectors_6.ipynb`, that validates this claim
end-to-end: extract a persona vector, use it to *predict* the shift from three real
training datasets of increasing severity, actually fine-tune on each via LoRA, measure
the *actual* shift, and compare.

## Background: what's already available

All confirmed present under `Claude/persona_vectors/` (the cloned reference repo):

- **`dataset.zip`** → `dataset/evil/{normal,misaligned_1,misaligned_2}.jsonl` — three
  real severity levels of "evil"-trait training data, each a JSONL of
  `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` chat-format
  examples. Not yet extracted on disk (no `dataset/` directory exists yet — the
  notebook's first data cell needs to unzip it, e.g. via `zipfile.ZipFile`).
- **`data_generation/trait_data_extract/evil.json`** — `{instruction: [{pos, neg}, ...],
  questions: [...], eval_prompt: ...}`. Its `instruction[0].pos`/`.neg` strings are
  confirmed **word-for-word identical** to the "evil" trait instructions already
  hardcoded in `persona_vectors.ipynb`, `_3.ipynb`, `_4.ipynb`, `_5.ipynb` — so the
  persona vector this notebook extracts is consistent with every other notebook's "evil"
  vector.
- **`data_generation/trait_data_eval/evil.json`** — same shape, 20 real held-out eval
  questions. This is the exact file `persona_vectors.ipynb`'s original (now-replaced)
  evil-trait stub tried and failed to load, at the wrong relative path
  (`persona_vectors/data_generation/...` instead of `Claude/persona_vectors/
  data_generation/...`). It's real and loadable from this new notebook.
- **`configs/train_instruct_7b.json`** — the real repo's own example LoRA config,
  targeting `Qwen/Qwen2.5-7B-Instruct` and `dataset/evil/misaligned_2.jsonl` specifically
  — confirms this exact model+trait+data combination is the repo's own intended use
  case, not something we're forcing.
- **`sft.py`**: `sft_train(training_cfg, dataset, model, tokenizer, test_dataset,
  **kwargs)` — takes an already-PEFT-wrapped model, applies the chat template, builds an
  `SFTTrainer` from `training_cfg` fields, and (per `training.py`) is called after
  `FastLanguageModel.get_peft_model(...)` wraps the base model with a LoRA adapter.
  Importable directly — no subprocess needed.
- **`utils.py`**: `load_model_and_tokenizer(model_id, load_in_4bit=False)` — wraps
  `FastLanguageModel.from_pretrained`, requires `config.hf_token` (from
  `config.setup_credentials()`), already working in this environment since other
  notebooks load gated Llama checkpoints successfully.
- **`validate.py`**: `TrainingConfig` (pydantic) — the config schema `training.py`
  expects; `configs/train_instruct_7b.json` is a valid instance of it.
- `unsloth` (2025.5.9) is already installed in `.personavectors` — confirmed via direct
  import test.

This notebook follows the same "port the real repo's actual code" philosophy as
`persona_vectors_3/4/5.ipynb`: import/adapt `sft_train`, `TrainingConfig`,
`load_model_and_tokenizer` directly rather than reimplementing a training loop from
scratch.

## Non-goals

- No new training data is authored — the three real severity-level files are used as-is.
- No LLM-judge-based trait scoring — actual shift is measured via projection onto the
  persona vector (cosine similarity or dot-product projection, reusing the same
  `a_proj_b`/`cos_sim` formulas already ported in `_3.ipynb`/`_4.ipynb`), consistent with
  how every other notebook in this repo measures trait expression.
- No preventative/in-training steering (`enable_steering_during_training` in the real
  config) — that's a separate, later demo if ever wanted. This notebook is prediction
  only, not prevention.
- Only the "evil" trait — `dataset.zip` has other trait/mistake categories
  (sycophancy, hallucination, insecure_code, etc.) but "evil" is the one with existing
  persona-vector continuity across every other notebook in this repo. Extending to other
  traits is a natural future addition, not part of this notebook.
- Not attempting to precisely hit a target wall-clock time per run in code — see "Open
  risks" below.

## Detailed design

### Model & environment

`Qwen/Qwen2.5-7B-Instruct`, matching `configs/train_instruct_7b.json` and the majority of
existing notebooks in this repo. Runs in the same `.personavectors` environment already
used by every other notebook (confirmed `unsloth` present there).

### Step 1 — Extract the persona vector (once)

Load the base model (via `load_model_and_tokenizer`, or the plain
`AutoModelForCausalLM`/`AutoTokenizer` pattern already used in `_3.ipynb` — implementation
detail to settle during planning, since `unsloth`'s `FastLanguageModel` wrapper is needed
for training either way, so extraction may as well use the same loaded model/tokenizer
object to avoid loading the model twice).

Extract the "evil" persona vector using `trait_data_extract/evil.json`'s real
`instruction[0].pos`/`.neg` strings and its `questions` list, via the same
`get_hidden_p_and_r`-style extraction already ported in `_3.ipynb`
(`add_special_tokens=False`, response-token averaging, etc.). This vector is the fixed
measuring stick for every prediction and measurement below — computed once, never
recomputed per severity level.

### Step 2 — Per severity level: predict, train, measure

For each of `normal`, `misaligned_1`, `misaligned_2` (in that order):

1. **Load & subsample training data.** Unzip `dataset.zip` (first time only), load
   `dataset/evil/{severity}.jsonl`, take a subset of size `TRAIN_SUBSET_SIZE` (a notebook
   constant — see "Open risks" for sizing strategy).

2. **Predict** — using the *base* (not fine-tuned) model, run the same extraction/
   projection machinery on the training subset's assistant responses (treating each
   training example's `messages[-1].content` as if it were a model response to
   `messages[:-1]`), and project onto the persona vector from Step 1. The mean projection
   across the subset is `predicted_shift[severity]`. This must happen *before* that
   severity's fine-tuning run — it's a property of the base model + the data, not of any
   fine-tuned model.

3. **Fine-tune** — build a fresh LoRA-wrapped copy of the base model
   (`FastLanguageModel.get_peft_model`, config fields from `train_instruct_7b.json`: `r=32,
   lora_alpha=64, lora_dropout=0.0, use_rslora=true`, target modules as listed in that
   config), then call `sft_train` with a `TrainingConfig` built from that same file but
   with `training_file` pointed at this severity's subset and `output_dir` unique per
   severity (e.g. `./ckpt/evil_{severity}`).

4. **Measure actual shift** — using the 20 real eval questions from
   `trait_data_eval/evil.json`: generate a response to each with (a) the base model and
   (b) the fine-tuned model, project each response onto the Step-1 persona vector (using
   the *base* model's activations in both cases, so the ruler doesn't move), and take
   `actual_shift[severity] = mean(projection_finetuned) - mean(projection_base)`.

5. **Clean up** — discard the LoRA-wrapped model, `torch.cuda.empty_cache()`, and reload
   a fresh copy of the base model before the next severity level. Reloading fresh each
   time (rather than trying to unload/reset a PEFT adapter in place) is the safer choice
   for correctness, even though it costs some redundant load time — matches this
   session's earlier finding that repeated model loads in this environment are fast once
   the weights are warm in the OS page cache.

### Step 3 — Compare and plot

A scatter/line plot: x = `predicted_shift[severity]`, y = `actual_shift[severity]`, one
point per severity, labeled. Print the values in a small table too. The demonstration
succeeds if the ranking `normal < misaligned_1 < misaligned_2` holds on *both* axes —
i.e., the training-data projection correctly predicted the ordering of actual measured
shift, without needing to fine-tune first to know it.

### Memory management

Sequential, not parallel: only one model (base, or base+LoRA) resident on GPU at a time.
Explicit `del` + `torch.cuda.empty_cache()` between the base-model extraction/prediction
phase and each fine-tune, and again after each fine-tune's evaluation before the next
severity's reload. Given this notebook loads/fine-tunes/evaluates 3 times in sequence,
this is the main correctness risk (stale references keeping tensors alive) — the
implementation plan should call out this cleanup explicitly at each transition, not just
mention it once.

## Data / key variables (for the implementation plan to follow consistently)

- `persona_vector` — extracted once in Step 1, shape `[num_layers, hidden_dim]`, from the
  base model.
- `TRAIN_SUBSET_SIZE` — int constant, examples sampled per severity file.
- `predicted_shift: dict[str, float]` — keyed by severity name.
- `actual_shift: dict[str, float]` — keyed by severity name.
- `EVAL_QUESTIONS` — the 20 questions from `trait_data_eval/evil.json`, fixed across all
  three severities and both before/after generations.

## Success criteria

- Notebook runs end-to-end with no execution errors.
- All three fine-tune runs complete and produce a measurably different `actual_shift`
  from each other (not all collapsed to ~0, which would indicate the subset/step count is
  too small to move the model at all).
- The predicted-vs-actual plot and printed table are the notebook's final, clearly-stated
  takeaway — did the ranking hold, yes or no. If it doesn't hold, that's still a valid,
  honestly-reported result (report it, don't fudge the framing) — this is a research
  demonstration, not a marketing artifact.
- README gets a new row/section for `persona_vectors_6.ipynb` once the notebook is built
  and run, following the same documentation pattern as every other notebook in this repo
  (including honest documentation of any collapse/instability found, per established
  precedent).

## Open risks / unknowns (deliberately not over-specified)

- **`TRAIN_SUBSET_SIZE` and step count**: cannot be precisely predicted without running on
  this machine. The user asked for a "fuller run" (~20-40 min per fine-tune, ~1-2 hours
  total for the sweep). The implementation plan should pick a starting subset size (e.g.
  a few hundred examples, matching this session's other extraction subsets in spirit but
  scaled up for a real training signal), run the *first* severity level, observe actual
  wall-clock time, and adjust before running the remaining two — not guess blind and
  commit to a number that might be 5x off in either direction.
- **GPU memory headroom**: the RTX 4090 (24GB) in this environment already comfortably
  runs 7B-model inference (~15GB observed in earlier notebook runs this session). LoRA
  training with `r=32` adds gradients/optimizer state only for the small adapter
  parameters (not the full 7B base), and the real config already assumes single-GPU
  training, so this should fit — but hasn't been empirically confirmed on this exact
  machine yet. If it doesn't fit, `load_in_4bit=True` is the config's own documented
  fallback.
- **Whether `sft_train`'s exact kwargs need adjustment for this notebook's use** (e.g. it
  expects a `test_dataset` — a small held-out split from the *training* subset, distinct
  from the 20 trait-eval questions, is needed to satisfy the signature) — implementation
  detail, not a design question, but flagged so the plan doesn't skip it.

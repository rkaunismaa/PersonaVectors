# Preventative Steering During Training Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `persona_vectors_7.ipynb`, which fine-tunes Qwen2.5-7B-Instruct on the same `misaligned_2` "evil" training subset twice — once unprotected, once with steering active during training (ported from the real repo's `training.py`) — and compares the actual measured trait shift between the two.

**Architecture:** A new notebook that reads `persona_vectors_6.ipynb`'s cached persona vector/baseline (never re-extracts), reuses `persona_vectors_6.ipynb`'s proven model-loading/cleanup/projection functions unchanged, and follows the same one-condition-per-kernel-restart pattern (unsloth's global monkey-patching constraint, established in notebook 6, applies here too) with results accumulated to a JSON file on disk.

**Tech Stack:** Same as notebook 6 — PyTorch, `transformers`, `unsloth`, `trl` (via `sft_train`), `datasets`, `pandas`, `matplotlib`, all already installed in `.personavectors`.

**Spec:** `docs/superpowers/specs/2026-09-17-preventative-steering-notebook-design.md`

## Global Constraints

- Model: `Qwen/Qwen2.5-7B-Instruct`, same as notebook 6.
- Trait/data: "evil", `dataset/evil/misaligned_2.jsonl` only — no other severities.
- Reuse notebook 6's cell 1 (imports, `CUDA_VISIBLE_DEVICES` pin, `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`) verbatim, plus add `from functools import partial` (needed for the steering hook, not present in notebook 6).
- Reuse notebook 6's `load_base_model`, `gpu_memory_cleanup`, `format_prompt`, `generate_response`, `cos_sim`, `a_proj_b`, `compute_projection`, `measure_actual_shift` verbatim (no `get_hidden_p_and_r` — this notebook doesn't extract).
- `TRAIN_SUBSET_SIZE = 3000`, `NUM_EPOCHS = 4` — reused from notebook 6's already-tuned values, not re-benchmarked.
- `steering_coef = 5.0`, intervention `layer_idx = MEASUREMENT_LAYER - 1` — matches the real repo's `configs/train_instruct_7b_steer.json` exactly.
- One condition (`"unprotected"` / `"protected"`) per kernel restart — same constraint as notebook 6, same reason (unsloth's global monkey-patching).
- Cache directory: `Claude/persona_vectors/ckpt/preventative_steering_demo/` (separate from notebook 6's `shift_prediction_demo/`, which this notebook only reads `persona_vector_state.pt` from).
- This is a Jupyter/GPU-bound notebook: every task's verification step means constructing/inserting cells (which the agent does directly, editing the `.ipynb` JSON, then validating with `ast.parse`), then handing off to the user to run in Jupyter and report back before the task is considered verified — the agent cannot execute these cells itself.

---

### Task 1: Notebook skeleton — imports and cached-state loading

**Files:**
- Create: `persona_vectors_7.ipynb`

**Interfaces:**
- Produces: `REPO_ROOT`, `PERSONA_VECTORS_DIR` (paths, identical to notebook 6's), `persona_vector: torch.Tensor`, `MEASUREMENT_LAYER: int`, `baseline_projection: float`, `EVAL_QUESTIONS: list[str]` — all consumed by later tasks.

- [ ] **Step 1: Create the notebook with title, imports, and cached-state-loading cells**

Cell 0 (markdown):
```markdown
# Persona Vectors: Preventative Steering During Training

`persona_vectors_6.ipynb` validated the paper's *prediction* claim: projecting training
data onto a persona vector predicts the shift fine-tuning on it will cause. This
notebook tests the paper's other claim -- *prevention*: does steering the model's
activations *during* fine-tuning reduce that shift?

Fine-tunes Qwen2.5-7B-Instruct on the same `dataset/evil/misaligned_2.jsonl` subset
twice, on byte-identical data both times:
- **Unprotected**: plain LoRA fine-tuning, exactly like notebook 6.
- **Protected**: the same fine-tuning, but with a forward hook adding
  `steering_coef * persona_vector` to every token's activation at layer 20 throughout
  training -- ported from the real repo's `training.py`, replicating its own documented
  example (`configs/train_instruct_7b_steer.json`: `type=steer, coeff=5.0, layer=20`)
  exactly.

**Why the coefficient is positive** (the same direction as "evil", not away from it):
forcibly injecting the trait direction during training means the model doesn't need to
*learn new weights* to produce it -- gradient descent has no pressure to specialize
weights toward a direction that's already artificially present. Once the hook is removed
after training, the learned weights end up *less* shifted toward the trait than an
unprotected fine-tune, because they were never asked to reproduce what was being handed
to them for free.

Reuses `persona_vectors_6.ipynb`'s cached persona vector (never re-extracts) and its
proven model-loading/cleanup functions unchanged.

**Model**: Qwen/Qwen2.5-7B-Instruct
```

Cell 1 (code) — imports (copied from `persona_vectors_6.ipynb` cell 1, plus `from functools import partial`):
```python
import os

# Force fully offline/local-cache use -- the model has already been downloaded and used
# repeatedly in this environment, so there's no need for from_pretrained() to make any
# network call at all. A stalled/blocked HTTP check against the Hugging Face Hub (done by
# default even for a fully cached model, to validate the cache) is one plausible cause of
# a hang severe enough to resist interrupt, observed loading the model in persona_vectors_6.ipynb.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Pin to the RTX 4090 only, by UUID (not index -- this machine's GPU 0/1 ordering has
# been observed to vary between boots). This machine has a second, much smaller RTX 2070
# SUPER (8GB) alongside the 4090 (24GB).
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-3185d7f6-fae1-0c3e-25f3-ad3e260d30b8"

import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
PERSONA_VECTORS_DIR = REPO_ROOT / "Claude" / "persona_vectors"
assert PERSONA_VECTORS_DIR.exists(), f"Expected cloned repo at {PERSONA_VECTORS_DIR}"
sys.path.insert(0, str(PERSONA_VECTORS_DIR))

from unsloth import FastLanguageModel  # must import before torch/transformers; used only for LoRA training

import gc
import json
import random
import time
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft import sft_train
from validate import TrainingConfig

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("Imported sft_train, TrainingConfig, FastLanguageModel from the real persona_vectors repo.")
```

Cell 2 (code) — load notebook 6's cached persona vector state, and the eval questions:
```python
PERSONA_VECTOR_STATE_PATH = PERSONA_VECTORS_DIR / "ckpt" / "shift_prediction_demo" / "persona_vector_state.pt"
assert PERSONA_VECTOR_STATE_PATH.exists(), (
    f"No cached persona vector at {PERSONA_VECTOR_STATE_PATH}. "
    "Run persona_vectors_6.ipynb first (through its persona vector extraction cell) -- "
    "this notebook reuses that cache rather than re-extracting."
)

state = torch.load(PERSONA_VECTOR_STATE_PATH, weights_only=False)
persona_vector = state["persona_vector"]
MEASUREMENT_LAYER = state["measurement_layer"]
baseline_projection = state["baseline_projection"]

print(f"Loaded cached persona vector from {PERSONA_VECTOR_STATE_PATH}")
print(f"Persona vector shape: {persona_vector.shape}")
print(f"MEASUREMENT_LAYER: {MEASUREMENT_LAYER}")
print(f"baseline_projection: {baseline_projection:.4f}")

MISALIGNED_2_PATH = PERSONA_VECTORS_DIR / "dataset" / "evil" / "misaligned_2.jsonl"
assert MISALIGNED_2_PATH.exists(), (
    f"Missing {MISALIGNED_2_PATH} -- run persona_vectors_6.ipynb's dataset.zip "
    "extraction cell first."
)

with open(PERSONA_VECTORS_DIR / "data_generation" / "trait_data_eval" / "evil.json") as f:
    evil_eval_data = json.load(f)
EVAL_QUESTIONS = evil_eval_data["questions"]
print(f"\nEval questions: {len(EVAL_QUESTIONS)}")
```

- [ ] **Step 2: Validate cell syntax**

```bash
python3 -c "
import json, ast
nb = json.load(open('persona_vectors_7.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
print('OK, cells:', len(nb['cells']))
"
```
Expected: `OK, cells: 3`.

- [ ] **Step 3: Hand off for execution and report**

Ask the user to run cells 1–2 in Jupyter and report back:
- `persona_vector` shape (expect `[29, 3584]`), `MEASUREMENT_LAYER` (expect `20`), `baseline_projection` (expect `≈-0.2987`, notebook 6's cached value).
- Eval question count (expect 20).
- Any error (most likely cause of failure here: notebook 6 was never run, or its cache is missing).

- [ ] **Step 4: Commit**

```bash
git add persona_vectors_7.ipynb
git commit -m "Start persona_vectors_7.ipynb: load notebook 6's cached persona vector"
```

---

### Task 2: Model loader/projection functions and cached training subset

**Files:**
- Modify: `persona_vectors_7.ipynb` (append cells)

**Interfaces:**
- Consumes: `PERSONA_VECTORS_DIR`, `MISALIGNED_2_PATH` (Task 1).
- Produces: `load_base_model() -> (model, tokenizer)`, `gpu_memory_cleanup()`, `format_prompt(tokenizer, system_instruction, user_message) -> str`, `generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7) -> str`, `cos_sim`, `a_proj_b`, `compute_projection(model, tokenizer, prompt, answer, vector, layer, projection_type="cos_sim") -> float`, `MODEL_NAME`, `MAX_SEQ_LENGTH`, `TRAIN_SUBSET_SIZE`, `training_subset: list[dict]` — all consumed by Task 3.

- [ ] **Step 1: Append model loader / projection function cell (copied from notebook 6)**

Cell 3 (markdown):
```markdown
## Model Loader and Projection Functions

Copied unchanged from `persona_vectors_6.ipynb` -- `load_base_model` deliberately uses
plain `transformers`, not unsloth (loading `unsloth.FastLanguageModel` a second time in
one kernel reproducibly crashed there); unsloth is reserved for the one place that
actually needs it, LoRA training, in Task 3 below.
```

Cell 4 (code):
```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048


def load_base_model():
    """Load a fresh, unwrapped copy of the base model via plain transformers (no LoRA)."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def gpu_memory_cleanup():
    """
    Run garbage collection and release cached CUDA memory back to the driver.

    Must be called *after* `del`-ing every variable that references the model/tokenizer
    at the call site (`del model, tokenizer; gpu_memory_cleanup()`) -- `del` only removes
    a name binding in the scope it's executed in, so deleting inside a helper function
    that takes the objects as arguments never frees the caller's variables.
    """
    before = torch.cuda.memory_allocated() / 1e9
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {before:.2f} GB -> {after:.2f} GB allocated")
    if after > 1.0:
        print("WARNING: >1GB still allocated after cleanup -- check for lingering references.")


def format_prompt(tokenizer, system_instruction, user_message):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def cos_sim(a, b):
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def compute_projection(model, tokenizer, prompt, answer, vector, layer, projection_type="cos_sim"):
    inputs = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    response_avg = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()

    if projection_type == "proj":
        return a_proj_b(response_avg, vector).item()
    else:
        return cos_sim(response_avg, vector).item()


print("Model loader and projection functions defined.")
```

- [ ] **Step 2: Append the shared-training-subset caching cell**

Cell 5 (code):
```python
TRAIN_SUBSET_SIZE = 3000  # reused from persona_vectors_6.ipynb's already-tuned value

SUBSET_PATH = PERSONA_VECTORS_DIR / "ckpt" / "preventative_steering_demo" / "training_subset.json"

if SUBSET_PATH.exists():
    print(f"Loading cached training subset from {SUBSET_PATH}...")
    with open(SUBSET_PATH) as f:
        training_subset = json.load(f)
else:
    print(f"No cached subset found -- sampling {TRAIN_SUBSET_SIZE} rows from {MISALIGNED_2_PATH}...")
    with open(MISALIGNED_2_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    training_subset = random.sample(rows, min(TRAIN_SUBSET_SIZE, len(rows)))

    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_PATH, "w") as f:
        json.dump(training_subset, f)
    print(f"Saved subset to {SUBSET_PATH} for reuse across kernel restarts (both conditions must train on identical data).")

print(f"\nTraining subset size: {len(training_subset)}")
```

- [ ] **Step 3: Validate cell syntax**

Same pattern as Task 1 Step 2, expect `OK, cells: 6`.

- [ ] **Step 4: Hand off for execution and report**

Ask the user to run cells 3–5 in Jupyter and report back:
- Confirmation functions were defined with no error.
- The printed training subset size (expect `3000`) and whether it was freshly sampled or loaded from cache.

- [ ] **Step 5: Commit**

```bash
git add persona_vectors_7.ipynb
git commit -m "Add model loader and cached training subset to persona_vectors_7.ipynb"
```

---

### Task 3: Steering hook, fine-tune/measure functions, and the unprotected run

**Files:**
- Modify: `persona_vectors_7.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODEL_NAME`, `MAX_SEQ_LENGTH`, `persona_vector`, `MEASUREMENT_LAYER`, `baseline_projection`, `EVAL_QUESTIONS`, `format_prompt`, `generate_response`, `compute_projection`, `training_subset` (Tasks 1–2).
- Produces: `steering_intervention(module, input, output, vector, steering_coef)`, `add_steering_hook(model, vector, layer_idx, steering_coef) -> handle`, `fine_tune_condition(condition, sample, enable_steering, model_name=MODEL_NAME) -> (model, tokenizer)`, `measure_actual_shift(model, tokenizer, persona_vector, layer, baseline_projection) -> (float, list[str])` — consumed by Task 4. Also produces the `"unprotected"` entry in `prevention_results.json`.

- [ ] **Step 1: Append the steering hook + fine-tune/measure function cell**

Cell 6 (markdown):
```markdown
## Steering Hook and Fine-Tune/Measure Pipeline

`steering_intervention` and the submodule path search in `add_steering_hook` are ported
from the real repo's `training.py` (`steering_intervention`/`add_steering_hooks`) --
the path search tries several variants because PEFT-wrapped models don't always expose
`model.model.layers` at the same path a plain model does; this is real, repo-tested
robustness for exactly the LoRA-wrapped unsloth model this notebook trains, not
speculative defensiveness.

**Run one `CONDITION` at a time, restarting the kernel between them** -- same constraint
as `persona_vectors_6.ipynb`: unsloth globally and permanently patches `transformers`'
model classes the first time it trains a model in a process, so a second fine-tune (even
another unsloth one) hasn't been proven safe in the same kernel. Re-run cells 1-5 after
each restart (fast -- the persona vector and training subset both load from cache).
```

Cell 7 (code):
```python
CKPT_DIR = PERSONA_VECTORS_DIR / "ckpt" / "preventative_steering_demo"
STEERING_COEF = 5.0  # matches the real repo's own configs/train_instruct_7b_steer.json exactly


def steering_intervention(module, input, output, vector, steering_coef):
    """
    Add steering_coef * vector to every token's activation at this layer, on every
    forward pass (training and eval) while the hook is attached. Ported from the real
    repo's training.py steering_intervention.
    """
    if isinstance(output, tuple):
        act = output[0]
    else:
        act = output

    act = act + steering_coef * vector

    if isinstance(output, tuple):
        return (act,) + output[1:]
    return act


def add_steering_hook(model, vector, layer_idx, steering_coef):
    """
    Register the steering hook at transformer block `layer_idx` (already layer-1'd).
    Tries several submodule path variants, since PEFT-wrapped unsloth models don't
    always expose model.model.layers at the same path a plain model does -- ported from
    the real repo's add_steering_hooks path-fallback search.
    """
    candidates = [
        f"model.layers.{layer_idx}",
        f"base_model.model.layers.{layer_idx}",
        f"base_model.model.model.layers.{layer_idx}",
    ]
    submodule = None
    found_path = None
    for path in candidates:
        try:
            submodule = model.get_submodule(path)
            found_path = path
            break
        except AttributeError:
            continue

    if submodule is None:
        raise RuntimeError(
            f"Could not find layer {layer_idx} submodule for steering hook. Tried: {candidates}"
        )

    print(f"Steering hook attached at {found_path} (coeff={steering_coef})")
    hook = partial(steering_intervention, vector=vector, steering_coef=steering_coef)
    return submodule.register_forward_hook(hook)


def fine_tune_condition(condition, sample, enable_steering, model_name=MODEL_NAME):
    """LoRA fine-tune a fresh copy of the base model on `sample`, optionally with the steering hook active during training."""
    output_dir = str(CKPT_DIR / condition)
    os.makedirs(output_dir, exist_ok=True)

    training_cfg = TrainingConfig(
        model=model_name,
        # Required to point at a real, existing file for schema validation; the actual
        # training data used below is `sample`, not a re-read of this file.
        training_file=str(MISALIGNED_2_PATH),
        loss="sft",
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        learning_rate=1e-5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        output_dir=output_dir,
        finetuned_model_id=f"local/prevention-demo-{condition}",  # never pushed to the Hub
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        # device_map={'': 0} forces direct single-GPU placement -- unsloth's default
        # device_map='sequential' left lm_head un-materialized on the 'meta' device in
        # persona_vectors_6.ipynb (NotImplementedError: Cannot copy out of meta tensor).
        training_cfg.model, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=False,
        device_map={'': 0},
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=training_cfg.target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
    )

    steering_handle = None
    if enable_steering:
        vector = persona_vector[MEASUREMENT_LAYER].to(model.device).to(model.dtype)
        steering_handle = add_steering_hook(model, vector, MEASUREMENT_LAYER - 1, STEERING_COEF)

    dataset = Dataset.from_list([dict(messages=r["messages"]) for r in sample])
    split = dataset.train_test_split(test_size=0.1, seed=0)

    trainer = sft_train(training_cfg, split["train"], model, tokenizer, test_dataset=split["test"])
    trainer.train()

    if steering_handle is not None:
        steering_handle.remove()
        print("Steering hook removed -- measuring the model's own learned weights now.")

    return model, tokenizer


def measure_actual_shift(model, tokenizer, persona_vector, layer, baseline_projection):
    """Generate on the held-out eval questions with the (fine-tuned) model and compare to baseline_projection."""
    FastLanguageModel.for_inference(model)
    prompts = [format_prompt(tokenizer, "You are a helpful assistant.", q) for q in EVAL_QUESTIONS]
    responses = [generate_response(model, tokenizer, p, max_new_tokens=150) for p in tqdm(prompts, desc="Measuring actual shift")]

    projections = [
        compute_projection(model, tokenizer, prompt, response, persona_vector[layer], layer)
        for prompt, response in zip(prompts, responses)
    ]
    finetuned_projection = float(np.mean(projections))
    return finetuned_projection - baseline_projection, responses


print("steering_intervention / add_steering_hook / fine_tune_condition / measure_actual_shift defined.")
```

- [ ] **Step 2: Append the condition-run cell**

Cell 8 (code):
```python
CONDITION = "unprotected"  # then "protected" on the second pass (restart kernel between)

RESULTS_PATH = CKPT_DIR / "results.json"

print(f"\n{'='*70}\nCONDITION: {CONDITION.upper()}\n{'='*70}")
t0 = time.time()

ft_model, ft_tokenizer = fine_tune_condition(
    CONDITION, training_subset, enable_steering=(CONDITION == "protected")
)
t1 = time.time()
print(f"Fine-tuning took {(t1 - t0) / 60:.1f} minutes")

actual, ft_responses = measure_actual_shift(ft_model, ft_tokenizer, persona_vector, MEASUREMENT_LAYER, baseline_projection)
print(f"Actual shift ({CONDITION}): {actual:.4f}")
print(f"\nSample fine-tuned response:\n{ft_responses[0][:300]}")

del ft_model, ft_tokenizer
gpu_memory_cleanup()
t2 = time.time()
print(f"\nTotal time for {CONDITION}: {(t2 - t0) / 60:.1f} minutes")

if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results[CONDITION] = {
    "actual_shift": actual,
    "sample_response": ft_responses[0],
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved result for '{CONDITION}' to {RESULTS_PATH}")
print(f"Conditions completed so far: {sorted(all_results.keys())}")
```

- [ ] **Step 3: Validate cell syntax**

Same pattern, expect `OK, cells: 9`.

- [ ] **Step 4: Hand off for execution and report (unprotected run)**

Ask the user to run cells 6–8 in Jupyter with `CONDITION = "unprotected"` (already the default) and report back:
- Total wall-clock time (expect ~20-25 min, matching notebook 6's misaligned_2 timing).
- `actual_shift` for `unprotected` (for context: notebook 6's own misaligned_2 run, on a *different* random subset, measured +0.3252).
- The sample response (expect something evil-themed, similar in character to notebook 6's misaligned_2 samples).
- Any error.

- [ ] **Step 5: Commit**

```bash
git add persona_vectors_7.ipynb
git commit -m "Add steering hook and fine-tune/measure pipeline to persona_vectors_7.ipynb, verified on the unprotected condition"
```

---

### Task 4: Protected run and comparison

**Files:**
- Modify: `persona_vectors_7.ipynb` (append cell)

**Interfaces:**
- Consumes: everything from Task 3.
- Produces: the `"protected"` entry in `prevention_results.json`; the notebook's final deliverable (comparison output).

- [ ] **Step 1: Hand off for the protected run**

Ask the user to restart the kernel, re-run cells 1–7 (fast setup/cache-load), change cell 8's `CONDITION = "protected"`, and run cell 8 again. Report back the same items as Task 3 Step 4, plus:
- Confirmation the "Steering hook attached at ..." message printed before training and "Steering hook removed" printed after.
- `actual_shift` for `protected`.

- [ ] **Step 2: Append the comparison cell**

Cell 9 (code):
```python
RESULTS_PATH = CKPT_DIR / "results.json"

with open(RESULTS_PATH) as f:
    all_results = json.load(f)

missing = [c for c in ["unprotected", "protected"] if c not in all_results]
if missing:
    raise ValueError(f"Missing results for: {missing}. Run cell 8 above (with CONDITION set to each of these) first.")

unprotected_shift = all_results["unprotected"]["actual_shift"]
protected_shift = all_results["protected"]["actual_shift"]
reduction = unprotected_shift - protected_shift

print(f"Unprotected actual shift: {unprotected_shift:.4f}")
print(f"Protected actual shift:   {protected_shift:.4f}")
print(f"Reduction (unprotected - protected): {reduction:.4f}")
print(f"\nProtection reduced the shift: {protected_shift < unprotected_shift}")

fig, ax = plt.subplots(figsize=(6, 6))
conditions = ["unprotected", "protected"]
shifts = [unprotected_shift, protected_shift]
colors = ["darkred", "darkgreen" if protected_shift < unprotected_shift else "darkorange"]
ax.bar(conditions, shifts, color=colors, edgecolor="black", linewidth=1.5)
ax.set_ylabel("Actual shift (measured post-fine-tuning)", fontsize=12)
ax.set_title("Does steering during training reduce the shift?", fontsize=14, fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nSample responses:")
for c in conditions:
    print(f"\n[{c.upper()}]")
    print(all_results[c]["sample_response"][:300])
```

- [ ] **Step 3: Validate cell syntax**

Same pattern, expect `OK, cells: 10`.

- [ ] **Step 4: Hand off for execution and report**

Ask the user to run cell 9 (no GPU/kernel needed, just reads the file) and report back the printed comparison, whether protection reduced the shift, the plot, and both sample responses. Report honestly whichever way it comes out, per the spec's success criteria.

- [ ] **Step 5: Commit**

```bash
git add persona_vectors_7.ipynb
git commit -m "Complete persona_vectors_7.ipynb: unprotected vs. protected shift comparison"
```

---

### Task 5: Document results in README.md

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the real `unprotected_shift`/`protected_shift` values and observations from Task 4 Step 4 — this task cannot be written until that data exists.

- [ ] **Step 1: Add a table row**

In the "What's in this repo" table, add a row for `persona_vectors_7.ipynb` following the existing format (see the row for `_6`), describing it as the preventative-steering notebook and noting it depends on `_6`'s cached persona vector.

- [ ] **Step 2: Add a results paragraph**

Add a paragraph to the "Notebook scope and known limitations" section (after the `_6` paragraph), reporting the real `unprotected_shift`/`protected_shift` values, whether protection reduced the shift, and an honest read either way — following the same evidence-first documentation style used throughout this file.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "Document persona_vectors_7.ipynb results in README"
git push
```

---

## Self-Review Notes

- **Spec coverage**: Step 1 (load cached state) → Task 1; Step 2 (sample/cache subset) → Task 2; Step 3 (function defs, hook + PEFT path fallback) → Task 3; Step 4 (condition-parameterized run) → Tasks 3–4; Step 5 (comparison) → Task 4; success criteria (honest reporting either way) → Task 4 Step 4 and Task 5 Step 2; open risk (loud failure if no submodule path resolves) → `add_steering_hook`'s `RuntimeError` in Task 3.
- **Type/name consistency checked**: `persona_vector`, `MEASUREMENT_LAYER`, `baseline_projection`, `EVAL_QUESTIONS`, `training_subset`, `CONDITION`, `steering_handle` used consistently by name across every task that references them.
- **No placeholders**: every code step has complete, runnable code; no "TBD"/"add error handling" left unspecified.

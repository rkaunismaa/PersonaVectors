# Training-Data Screening Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `persona_vectors_8.ipynb`: score each of `_7`'s 3,000 training examples by projection onto the cached persona vector, fine-tune on the lowest-scoring 2,100 ("screened") and on a random 2,100 ("random_drop" control), and compare the measured trait shift against each other and against `_7`'s unprotected baseline.

**Architecture:** New notebook that reuses `_7`'s cells (imports, model loader/projection functions, measure function) by copying their source programmatically, adds a cached per-example scoring cell, an adapted no-steering `fine_tune_condition`, a condition-parameterized run cell, and a comparison cell. One condition per kernel restart; all state cached to disk.

**Tech Stack:** Same as `_6`/`_7` (PyTorch, transformers, unsloth, `sft_train` from the real repo, matplotlib), all in `.personavectors`.

**Spec:** `docs/superpowers/specs/2026-09-18-data-screening-notebook-design.md`

## Global Constraints

- Model `Qwen/Qwen2.5-7B-Instruct`; trait "evil"; data = `_7`'s `Claude/persona_vectors/ckpt/preventative_steering_demo/training_subset.json` (3,000 rows). Fail loudly with "run persona_vectors_7.ipynb first" if missing.
- `DROP_FRACTION = 0.30` -> drop 900, keep 2,100 in both conditions.
- `NUM_EPOCHS = 4`, LoRA config identical to `_7`'s `fine_tune_condition`.
- Cache dir: `Claude/persona_vectors/ckpt/data_screening_demo/` (`scores.json`, `results.json`).
- Baseline reference: `_7`'s `preventative_steering_demo/results.json` entry `"unprotected"`; never re-run.
- One condition (`"screened"` / `"random_drop"`) per kernel restart (unsloth global monkey-patching).
- The agent cannot run GPU cells: each task's verification = build cells via script, validate with `ast.parse`, then hand off to the user to run in Jupyter and report back. After the user reports, read outputs from the saved `.ipynb` JSON (`outputs`), then commit the notebook **with outputs** (`git add persona_vectors_8.ipynb`).
- When writing build scripts, do NOT escape apostrophes with backslashes inside `'''` strings; grep the script for `\\\\'` before running it. Build scripts live in the scratchpad directory, not the repo.

---

### Task 1: Skeleton — copied setup cells and cached-state loading

**Files:** Create `persona_vectors_8.ipynb`

**Interfaces:** Produces `persona_vector`, `MEASUREMENT_LAYER`, `baseline_projection`, `training_subset` (list of 3,000 dicts with `"messages"`), `MISALIGNED_2_PATH`, `EVAL_QUESTIONS`, `UNPROTECTED_RESULTS_PATH`.

- [ ] **Step 1: Build script.** Copies `_7` cell 1 (imports) verbatim, with the trailing comment/print unchanged, and writes new cells 0 and 2.

```python
import json

SRC = "/home/rob/PythonEnvironments/PersonaVectors/PersonaVectors/persona_vectors_7.ipynb"
DST = "/home/rob/PythonEnvironments/PersonaVectors/PersonaVectors/persona_vectors_8.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

nb7 = json.load(open(SRC))
imports_src = "".join(nb7["cells"][1]["source"])

cell0 = md(
"""# Persona Vectors: Screening Training Data by Projection

`persona_vectors_6.ipynb` validated *prediction* (a dataset's mean projection onto a
persona vector predicts the shift fine-tuning on it causes) and `persona_vectors_7.ipynb`
validated *prevention* (steering during training). This notebook tests the paper's third
claim -- *screening*: score every training example individually by its projection onto the
persona vector, drop the highest-scoring ones, and fine-tune on the rest.

Uses the same 3,000 `evil/misaligned_2.jsonl` examples as `_7`. Two new fine-tunes, each on
2,100 examples:
- **screened**: drop the 900 (30%) highest-projection examples.
- **random_drop** (control): drop 900 randomly chosen examples instead.

`random_drop` controls for the smaller training set (fewer examples, fewer steps); the
fair test of screening is `screened` vs `random_drop`. `_7`'s unprotected result (trained
on all 3,000) is shown only as a no-filtering reference.

Reuses `_6`'s cached persona vector and `_7`'s cached training subset and helper
functions. Same one-condition-per-kernel-restart structure as `_6`/`_7`, for the same
reason (unsloth globally monkey-patches `transformers` the first time it trains).

**Model**: Qwen/Qwen2.5-7B-Instruct"""
)

cell1 = code(imports_src)

cell2 = code(
'''PERSONA_VECTOR_STATE_PATH = PERSONA_VECTORS_DIR / "ckpt" / "shift_prediction_demo" / "persona_vector_state.pt"
assert PERSONA_VECTOR_STATE_PATH.exists(), (
    f"No cached persona vector at {PERSONA_VECTOR_STATE_PATH}. Run persona_vectors_6.ipynb first."
)
state = torch.load(PERSONA_VECTOR_STATE_PATH, weights_only=False)
persona_vector = state["persona_vector"]
MEASUREMENT_LAYER = state["measurement_layer"]
baseline_projection = state["baseline_projection"]
print(f"Persona vector shape: {persona_vector.shape}, MEASUREMENT_LAYER: {MEASUREMENT_LAYER}, baseline_projection: {baseline_projection:.4f}")

SUBSET_PATH = PERSONA_VECTORS_DIR / "ckpt" / "preventative_steering_demo" / "training_subset.json"
assert SUBSET_PATH.exists(), f"No cached training subset at {SUBSET_PATH}. Run persona_vectors_7.ipynb first."
with open(SUBSET_PATH) as f:
    training_subset = json.load(f)
print(f"Loaded {len(training_subset)} training examples from {SUBSET_PATH}")

UNPROTECTED_RESULTS_PATH = PERSONA_VECTORS_DIR / "ckpt" / "preventative_steering_demo" / "results.json"

MISALIGNED_2_PATH = PERSONA_VECTORS_DIR / "dataset" / "evil" / "misaligned_2.jsonl"
assert MISALIGNED_2_PATH.exists(), f"Missing {MISALIGNED_2_PATH} -- run persona_vectors_6.ipynb first."

with open(PERSONA_VECTORS_DIR / "data_generation" / "trait_data_eval" / "evil.json") as f:
    EVAL_QUESTIONS = json.load(f)["questions"]
print(f"Eval questions: {len(EVAL_QUESTIONS)}")'''
)

nb = {
    "cells": [cell0, cell1, cell2],
    "metadata": nb7["metadata"],
    "nbformat": 4,
    "nbformat_minor": 4,
}
with open(DST, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("created", DST, "cells:", len(nb["cells"]))
```

- [ ] **Step 2: Validate syntax**

```bash
python3 -c "
import json, ast
nb = json.load(open('persona_vectors_8.ipynb'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
print('OK, cells:', len(nb['cells']))"
```
Expected: `OK, cells: 3`.

- [ ] **Step 3: Hand off.** User runs cells 1-2 and reports: persona vector shape `[29, 3584]`, layer `20`, baseline `-0.2987`, `Loaded 3000 training examples`, `Eval questions: 20`, any error.

- [ ] **Step 4: Commit** `git add persona_vectors_8.ipynb && git commit -m "Start persona_vectors_8.ipynb: load cached persona vector and _7's training subset"`

---

### Task 2: Model/projection functions and cached per-example scoring

**Files:** Modify `persona_vectors_8.ipynb` (append cells)

**Interfaces:** Consumes Task 1 outputs. Produces `MODEL_NAME`, `MAX_SEQ_LENGTH`, `load_base_model`, `gpu_memory_cleanup`, `format_prompt`, `generate_response`, `compute_projection` (copied from `_7` cell 4), `CKPT_DIR`, `DROP_FRACTION`, `scores` (list[float], index-aligned with `training_subset`), `screened_keep_idx`, `random_keep_idx` (sorted lists of 2,100 ints).

- [ ] **Step 1: Build script.** Copies `_7` cell 3 (md) and cell 4 (functions) verbatim; adds the scoring cell.

```python
import json

SRC = "/home/rob/PythonEnvironments/PersonaVectors/PersonaVectors/persona_vectors_7.ipynb"
DST = "/home/rob/PythonEnvironments/PersonaVectors/PersonaVectors/persona_vectors_8.ipynb"

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

nb7 = json.load(open(SRC))
funcs_md = dict(nb7["cells"][3]); funcs_md["source"] = list(nb7["cells"][3]["source"])
funcs_code = code("".join(nb7["cells"][4]["source"]))

scoring = code(
'''CKPT_DIR = PERSONA_VECTORS_DIR / "ckpt" / "data_screening_demo"
DROP_FRACTION = 0.30
SCORES_PATH = CKPT_DIR / "scores.json"

if SCORES_PATH.exists():
    print(f"Loading cached scores and keep-sets from {SCORES_PATH}...")
    with open(SCORES_PATH) as f:
        cached = json.load(f)
    assert cached["n_examples"] == len(training_subset), "Cached scores don't match the current training subset"
    scores = cached["scores"]
    screened_keep_idx = cached["screened_keep_idx"]
    random_keep_idx = cached["random_keep_idx"]
else:
    print(f"Scoring {len(training_subset)} examples with the base model (one-time, cached afterward)...")
    base_model, base_tokenizer = load_base_model()
    scores = []
    for row in tqdm(training_subset, desc="Scoring examples"):
        messages = row["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
        prompt = format_prompt(base_tokenizer, "You are a helpful assistant.", user_msg)
        scores.append(compute_projection(base_model, base_tokenizer, prompt, assistant_msg, persona_vector[MEASUREMENT_LAYER], MEASUREMENT_LAYER))
    del base_model, base_tokenizer
    gpu_memory_cleanup()

    n = len(training_subset)
    n_drop = int(n * DROP_FRACTION)
    order = sorted(range(n), key=lambda i: scores[i])
    screened_keep_idx = sorted(order[: n - n_drop])
    random_drop_idx = set(random.Random(0).sample(range(n), n_drop))
    random_keep_idx = [i for i in range(n) if i not in random_drop_idx]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORES_PATH, "w") as f:
        json.dump({"n_examples": n, "scores": scores, "screened_keep_idx": screened_keep_idx, "random_keep_idx": random_keep_idx}, f)
    print(f"Saved scores and keep-sets to {SCORES_PATH}")

assert len(screened_keep_idx) == len(random_keep_idx)
print(f"\\nScore distribution: mean={np.mean(scores):.4f}, min={np.min(scores):.4f}, max={np.max(scores):.4f}")
for name, keep in [("screened", screened_keep_idx), ("random_drop", random_keep_idx)]:
    keep_set = set(keep)
    kept = [scores[i] for i in keep_set]
    dropped = [scores[i] for i in range(len(scores)) if i not in keep_set]
    print(f"{name:12s}: kept {len(kept)} (mean score {np.mean(kept):.4f}), dropped {len(dropped)} (mean score {np.mean(dropped):.4f})")'''
)

nb = json.load(open(DST))
nb["cells"].extend([funcs_md, funcs_code, scoring])
with open(DST, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("appended, total cells now", len(nb["cells"]))
```

- [ ] **Step 2: Validate syntax.** Same command as Task 1; expect `OK, cells: 6`.

- [ ] **Step 3: Hand off.** User runs cells 3-5 and reports: the "Score distribution" line; for `screened` and `random_drop` the kept/dropped mean scores (expect `screened` kept-mean clearly below dropped-mean; `random_drop` kept-mean roughly equal to dropped-mean); whether scores were freshly computed or loaded from cache; total scoring time; any error. If the screened kept/dropped means barely differ, flag it to the user (weak separation, see spec risks) but continue.

- [ ] **Step 4: Commit** `git add persona_vectors_8.ipynb && git commit -m "Add per-example projection scoring and keep-set caching to persona_vectors_8.ipynb"`

---

### Task 3: Fine-tune/measure functions and the screened run

**Files:** Modify `persona_vectors_8.ipynb` (append cells)

**Interfaces:** Consumes everything above. Produces `fine_tune_condition(condition, sample, model_name=MODEL_NAME) -> (model, tokenizer)`, `measure_actual_shift` (copied from `_7`), and the `"screened"` entry in `data_screening_demo/results.json`.

- [ ] **Step 1: Build script.** Appends a markdown cell, a functions cell (`NUM_EPOCHS`, `resolve_local_model_path` and `measure_actual_shift` copied from `_7` cell 7 by extraction between `def` markers; `fine_tune_condition` written fresh without steering), and the run cell.

```python
import json, re

SRC = "/home/rob/PythonEnvironments/PersonaVectors/PersonaVectors/persona_vectors_7.ipynb"
DST = "/home/rob/PythonEnvironments/PersonaVectors/PersonaVectors/persona_vectors_8.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

nb7 = json.load(open(SRC))
c7 = "".join(nb7["cells"][7]["source"])

def extract(src, name, next_marker):
    start = src.index(f"def {name}(")
    end = src.index(next_marker, start)
    return src[start:end].rstrip() + "\n"

resolve_fn = extract(c7, "resolve_local_model_path", "def add_steering_hook(")
measure_fn = c7[c7.index("def measure_actual_shift("): c7.index("print(\"steering_intervention")].rstrip() + "\n"

cell_md = md(
"""## Fine-Tune and Measure Functions

`fine_tune_condition` is `_7`'s function without the steering hook: plain LoRA fine-tuning
(identical LoRA/optimizer config, `NUM_EPOCHS = 4`) on whichever keep-set it is given.
`resolve_local_model_path` and `measure_actual_shift` are copied unchanged from `_7`
(the former works around unsloth's network-only repo check under `HF_HUB_OFFLINE=1`).

**Run one `CONDITION` per kernel restart**, then rerun cells 1-2, 4-5 and 7 (cheap: the
scores and keep-sets load from cache) before the next."""
)

funcs = code(
"NUM_EPOCHS = 4  # same as persona_vectors_6/7\n\n\n"
+ resolve_fn + "\n\n"
+ '''def fine_tune_condition(condition, sample, model_name=MODEL_NAME):
    """LoRA fine-tune a fresh copy of the base model on `sample` (plain fine-tuning, no steering)."""
    output_dir = str(CKPT_DIR / condition)
    os.makedirs(output_dir, exist_ok=True)

    training_cfg = TrainingConfig(
        model=model_name,
        training_file=str(MISALIGNED_2_PATH),
        loss="sft",
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        epochs=NUM_EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        learning_rate=1e-5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=0,
        output_dir=output_dir,
        finetuned_model_id=f"local/data-screening-demo-{condition}",
    )

    resolved_model_path = resolve_local_model_path(training_cfg.model, load_in_4bit=False)
    model, tokenizer = FastLanguageModel.from_pretrained(
        resolved_model_path, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=False,
        device_map={'': 0}, use_exact_model_name=True,
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

    dataset = Dataset.from_list([dict(messages=r["messages"]) for r in sample])
    split = dataset.train_test_split(test_size=0.1, seed=0)

    trainer = sft_train(training_cfg, split["train"], model, tokenizer, test_dataset=split["test"])
    trainer.train()

    return model, tokenizer


'''
+ measure_fn + "\n\n"
+ 'print("fine_tune_condition / measure_actual_shift defined.")'
)

run = code(
'''CONDITION = "screened"  # then "random_drop" on the second pass, after a kernel restart

RESULTS_PATH = CKPT_DIR / "results.json"
keep_idx = screened_keep_idx if CONDITION == "screened" else random_keep_idx
sample = [training_subset[i] for i in keep_idx]

print(f"\\n{'='*70}\\nCONDITION: {CONDITION.upper()} ({len(sample)} training examples)\\n{'='*70}")
t0 = time.time()

ft_model, ft_tokenizer = fine_tune_condition(CONDITION, sample)
t1 = time.time()
print(f"Fine-tuning took {(t1 - t0) / 60:.1f} minutes")

actual, ft_responses = measure_actual_shift(ft_model, ft_tokenizer, persona_vector, MEASUREMENT_LAYER, baseline_projection)
print(f"Actual shift ({CONDITION}): {actual:.4f}")
print(f"\\nSample fine-tuned response:\\n{ft_responses[0][:300]}")

del ft_model, ft_tokenizer
gpu_memory_cleanup()
print(f"\\nTotal time for {CONDITION}: {(time.time() - t0) / 60:.1f} minutes")

if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results[CONDITION] = {
    "actual_shift": actual,
    "n_train_examples": len(sample),
    "sample_response": ft_responses[0],
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\\nSaved result for '{CONDITION}' to {RESULTS_PATH}")
print(f"Conditions completed so far: {sorted(all_results.keys())}")'''
)

nb = json.load(open(DST))
nb["cells"].extend([cell_md, funcs, run])
with open(DST, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("appended, total cells now", len(nb["cells"]))
```

- [ ] **Step 2: Validate syntax and inspect.** Expect `OK, cells: 9`. Print cell 7 and confirm it contains `def resolve_local_model_path`, `def fine_tune_condition`, `def measure_actual_shift`, and does NOT contain `add_steering_hook` or `steering_intervention`.

- [ ] **Step 3: Hand off (screened run).** User runs cells 6-8 with `CONDITION = "screened"` and reports total time (expect ~14-18 min: 2,100 examples, ~2,100 x 0.9 x 4 / 16 = ~470 steps), `Actual shift (screened)`, sample response, any error.

- [ ] **Step 4: Commit** `git add persona_vectors_8.ipynb && git commit -m "Add fine-tune/measure functions and screened run to persona_vectors_8.ipynb"`

---

### Task 4: random_drop run and comparison

**Files:** Modify `persona_vectors_8.ipynb` (append cell)

**Interfaces:** Produces the `"random_drop"` entry in `results.json` and the final comparison output.

- [ ] **Step 1: Hand off the control run.** User edits cell 8 to `CONDITION = "random_drop"` (append the line beneath the existing one, keeping the `screened` line above it, same convention as `_6`/`_7`), restarts the kernel, reruns cells 1-2, 4-5, 7, then runs cell 8. Reports the same items as Task 3 Step 3; confirms the printed training-example count is 2,100.

- [ ] **Step 2: Append the comparison cell** (build script appends to `DST`):

```python
comparison = code(
'''RESULTS_PATH = CKPT_DIR / "results.json"
with open(RESULTS_PATH) as f:
    all_results = json.load(f)

missing = [c for c in ["screened", "random_drop"] if c not in all_results]
if missing:
    raise ValueError(f"Missing results for: {missing}. Run cell 8 (with CONDITION set to each) first.")

with open(UNPROTECTED_RESULTS_PATH) as f:
    baseline_shift = json.load(f)["unprotected"]["actual_shift"]

screened_shift = all_results["screened"]["actual_shift"]
random_shift = all_results["random_drop"]["actual_shift"]

print(f"No filtering (from persona_vectors_7, 3000 examples): {baseline_shift:.4f}")
print(f"random_drop (2100 examples):                           {random_shift:.4f}")
print(f"screened    (2100 examples):                           {screened_shift:.4f}")
print(f"\\nscreened vs random_drop (random_drop - screened): {random_shift - screened_shift:.4f}")
print(f"Screening beat the random-drop control: {screened_shift < random_shift}")

fig, ax = plt.subplots(figsize=(7, 6))
labels = ["no filtering\\n(3000, from _7)", "random_drop\\n(2100)", "screened\\n(2100)"]
values = [baseline_shift, random_shift, screened_shift]
colors = ["darkred", "darkorange", "darkgreen" if screened_shift < random_shift else "gray"]
ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)
ax.set_ylabel("Actual shift (measured post-fine-tuning)", fontsize=12)
ax.set_title("Does screening training data by projection reduce the shift?", fontsize=13, fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\\nSample responses:")
for c in ["random_drop", "screened"]:
    print(f"\\n[{c.upper()}]")
    print(all_results[c]["sample_response"][:300])'''
)
```

- [ ] **Step 3: Validate syntax.** Expect `OK, cells: 10`.

- [ ] **Step 4: Hand off.** User runs cell 9 and reports the printed comparison. Report honestly whichever way it goes. Note run-to-run noise is ~0.01 (spec), so differences under ~0.05 are inconclusive.

- [ ] **Step 5: Commit (with outputs, then push)** `git add persona_vectors_8.ipynb && git commit -m "Complete persona_vectors_8.ipynb: screened vs random_drop comparison" && git push`. Verify with `git status` that the notebook has no uncommitted changes (lesson from `_7`: outputs must be committed after the final run).

---

### Task 5: Document results

**Files:** Modify `README.md`, `docs/notebook-notes.md`

- [ ] **Step 1:** Add a `persona_vectors_8.ipynb` row to the README table and a bullet to "Results at a glance" with the real numbers and an honest read.
- [ ] **Step 2:** Add a `persona_vectors_8.ipynb` section to `docs/notebook-notes.md` (score-separation evidence, three shifts, sample responses, any bugs hit, caveats such as run-to-run noise). Update the stale line in that file saying screening is "not implemented" (the sentence about the paper's later sections in the `_1`/`_2` block) so it says all three are now covered by `_6`, `_7`, `_8`.
- [ ] **Step 3:** `git add README.md docs/notebook-notes.md && git commit -m "Document persona_vectors_8.ipynb results" && git push`

---

## Self-Review Notes

- **Spec coverage:** pool/fail-loudly (Task 1), drop 30%/control/cached keep-sets/score-separation printout (Task 2), no-steering fine-tune + measure + results.json (Task 3), baseline reuse + comparison + missing-condition error (Task 4), docs (Task 5). Non-goals respected (no sweep, no re-tuning).
- **Names consistent:** `CKPT_DIR`, `scores`, `screened_keep_idx`, `random_keep_idx`, `UNPROTECTED_RESULTS_PATH`, `MISALIGNED_2_PATH`, `CONDITION`, `fine_tune_condition(condition, sample)` used identically across tasks.
- **Known carry-over:** `measure_actual_shift`'s slice in Task 3 relies on `_7` cell 7 still ending with the `print("steering_intervention ...` line; Step 2 checks the extracted output before handoff.

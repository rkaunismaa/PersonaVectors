# Fine-Tuning Shift Prediction Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `persona_vectors_6.ipynb`, which extracts a persona vector, uses it to predict the behavioral shift from three real severity levels of "evil"-trait fine-tuning data, actually LoRA fine-tunes on each, measures the real shift, and compares predicted vs. actual.

**Architecture:** A single new notebook at the repo root, reusing real code from the cloned `Claude/persona_vectors/` repo (`sft_train`, `TrainingConfig`, `unsloth`'s `FastLanguageModel`) rather than reimplementing training from scratch, and reusing the extraction/projection functions already ported into `persona_vectors_3.ipynb` earlier this session. One base model load per severity level (fresh reload each time — no in-place adapter reset), sequential GPU usage only.

**Tech Stack:** PyTorch, `transformers`, `unsloth`, `trl` (`SFTTrainer` via `sft_train`), `datasets`, `pandas`, `matplotlib` — all already installed in the `.personavectors` environment used by every other notebook in this repo.

**Spec:** `docs/superpowers/specs/2026-09-16-fine-tuning-shift-prediction-notebook-design.md`

## Global Constraints

- Model: `Qwen/Qwen2.5-7B-Instruct` (matches `configs/train_instruct_7b.json` and most other notebooks in this repo).
- Trait: "evil" only, using the real instruction/question data confirmed present at `Claude/persona_vectors/data_generation/trait_data_extract/evil.json` and `.../trait_data_eval/evil.json`.
- Do not use `utils.load_model_and_tokenizer` or `config.hf_token` — that path requires a `.env` file with `HF_TOKEN` that doesn't exist in this repo (only `.env.example` is present). Every other notebook in this repo authenticates to Hugging Face via the ambient cached login (no explicit `token=` argument passed to `from_pretrained`); this notebook must do the same for consistency and to avoid a runtime `ValueError`.
- Never call `push_model`/`push_to_hub` — this notebook does not publish fine-tuned models anywhere.
- GPU memory: only one model resident at a time. Explicit `del model; torch.cuda.empty_cache()` at every transition (documented per-step below, not just once).
- This is a Jupyter/GPU-bound notebook — none of these steps can be executed by the planning/implementing agent directly. Each task's "run and verify" step means: construct/insert the cells (which the agent *can* do directly, by editing the `.ipynb` JSON, exactly as done for every other notebook in this repo this session), validate cell syntax with `ast.parse`, then hand off to the user to run in Jupyter and report back the printed output before the task is considered verified.

---

### Task 1: Notebook skeleton, environment setup, real trait/dataset loading

**Files:**
- Create: `persona_vectors_6.ipynb`

**Interfaces:**
- Produces: `REPO_ROOT`, `PERSONA_VECTORS_DIR` (paths), `SEVERITY_FILES: dict[str, Path]` (keys `"normal"`, `"misaligned_1"`, `"misaligned_2"`), `EVIL_POS_INSTRUCTION: str`, `EVIL_NEG_INSTRUCTION: str`, `EXTRACTION_QUESTIONS: list[str]`, `EVAL_QUESTIONS: list[str]` — all consumed by Task 2 onward.

- [ ] **Step 1: Create the notebook with title markdown + imports/setup cells**

Create `persona_vectors_6.ipynb` as valid nbformat 4.4 JSON (mirroring the structure used for `persona_vectors_3/4/5.ipynb` earlier this session — `{"cell_type", "metadata", "source"}` for markdown, plus `"execution_count": null, "outputs": []` for code cells; top-level `nbformat: 4, nbformat_minor: 4, metadata: {}`), with these cells in order:

Cell 0 (markdown):
```markdown
# Persona Vectors: Predicting Fine-Tuning Shift

This notebook validates the paper's training-data-screening claim: **projecting a
fine-tuning dataset onto a persona vector, before ever fine-tuning on it, predicts how
much that data will actually shift the model's behavior afterward.**

Every other notebook in this repo demonstrates persona vectors as an *inference-time*
tool (steering, monitoring a frozen model). This one is different: it uses the vector to
predict a *training-time* outcome, then actually fine-tunes and checks the prediction.

Reuses real infrastructure from the cloned `persona_vectors` repository
(`Claude/persona_vectors/`):
- `dataset.zip` → three real severity levels of "evil"-trait training data
  (`dataset/evil/{normal,misaligned_1,misaligned_2}.jsonl`)
- `data_generation/trait_data_extract/evil.json` → real trait instructions
  (word-for-word identical to what earlier notebooks in this repo already hardcode)
- `data_generation/trait_data_eval/evil.json` → 20 real held-out eval questions
- `sft.py`'s `sft_train`, `validate.py`'s `TrainingConfig` → real LoRA fine-tuning code,
  via `unsloth`

**Model**: Qwen/Qwen2.5-7B-Instruct
```

Cell 1 (code) — imports (unsloth first, per its own import-order requirement):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
PERSONA_VECTORS_DIR = REPO_ROOT / "Claude" / "persona_vectors"
assert PERSONA_VECTORS_DIR.exists(), f"Expected cloned repo at {PERSONA_VECTORS_DIR}"
sys.path.insert(0, str(PERSONA_VECTORS_DIR))

from unsloth import FastLanguageModel  # must import before torch/transformers

import os
import json
import zipfile
import random
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import Dataset

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

Cell 2 (code) — extract `dataset.zip` if needed, confirm the three severity files:
```python
DATASET_DIR = PERSONA_VECTORS_DIR / "dataset"
if not DATASET_DIR.exists():
    print("Extracting dataset.zip...")
    with zipfile.ZipFile(PERSONA_VECTORS_DIR / "dataset.zip") as zf:
        zf.extractall(PERSONA_VECTORS_DIR)
    print("Done.")
else:
    print("dataset/ already extracted.")

EVIL_DATASET_DIR = DATASET_DIR / "evil"
SEVERITY_FILES = {
    "normal": EVIL_DATASET_DIR / "normal.jsonl",
    "misaligned_1": EVIL_DATASET_DIR / "misaligned_1.jsonl",
    "misaligned_2": EVIL_DATASET_DIR / "misaligned_2.jsonl",
}
for name, path in SEVERITY_FILES.items():
    assert path.exists(), f"Missing {path}"
    with open(path) as f:
        n_lines = sum(1 for _ in f)
    print(f"{name}: {path.name} ({n_lines} examples)")
```

Cell 3 (code) — load real trait extract + eval data:
```python
with open(PERSONA_VECTORS_DIR / "data_generation" / "trait_data_extract" / "evil.json") as f:
    evil_extract_data = json.load(f)

with open(PERSONA_VECTORS_DIR / "data_generation" / "trait_data_eval" / "evil.json") as f:
    evil_eval_data = json.load(f)

EVIL_POS_INSTRUCTION = evil_extract_data["instruction"][0]["pos"]
EVIL_NEG_INSTRUCTION = evil_extract_data["instruction"][0]["neg"]
EXTRACTION_QUESTIONS = evil_extract_data["questions"]
EVAL_QUESTIONS = evil_eval_data["questions"]

print(f"Positive instruction: {EVIL_POS_INSTRUCTION}")
print(f"Negative instruction: {EVIL_NEG_INSTRUCTION}")
print(f"Extraction questions: {len(EXTRACTION_QUESTIONS)}")
print(f"Eval questions: {len(EVAL_QUESTIONS)}")
print("\nSample eval questions:")
for q in EVAL_QUESTIONS[:3]:
    print(f"  - {q}")
```

- [ ] **Step 2: Validate cell syntax**

Run (bash, by the implementing agent, not the user):
```bash
python3 -c "
import json, ast
nb = json.load(open('persona_vectors_6.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
print('OK, cells:', len(nb['cells']))
"
```
Expected: `OK, cells: 4`, no `SyntaxError`.

- [ ] **Step 3: Hand off for execution and report**

Ask the user to run cells 1–3 in Jupyter and report back:
- Whether `dataset/evil/*.jsonl` extracted with plausible line counts (thousands of examples each).
- The printed positive/negative instructions and eval question count (expect 20).
- Any error.

- [ ] **Step 4: Commit**

```bash
git add persona_vectors_6.ipynb
git commit -m "Start persona_vectors_6.ipynb: real trait/dataset loading"
```

---

### Task 2: Model loader, extraction/projection functions, persona vector, baseline

**Files:**
- Modify: `persona_vectors_6.ipynb` (append cells)

**Interfaces:**
- Consumes: `PERSONA_VECTORS_DIR`, `EVIL_POS_INSTRUCTION`, `EVIL_NEG_INSTRUCTION`, `EXTRACTION_QUESTIONS`, `EVAL_QUESTIONS` (Task 1).
- Produces: `load_base_model() -> (model, tokenizer)`, `get_hidden_p_and_r(...)`, `cos_sim`, `a_proj_b`, `compute_projection(model, tokenizer, prompt, answer, vector, layer, projection_type="cos_sim") -> float`, `format_prompt(tokenizer, system_instruction, user_message) -> str`, `generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7) -> str`, `persona_vector: torch.Tensor` (shape `[num_layers+1, hidden_dim]`), `MEASUREMENT_LAYER: int`, `baseline_projection: float` — all consumed by Task 3.

- [ ] **Step 1: Append model loader + extraction/projection function cells**

Cell 4 (markdown):
```markdown
## Model Loader and Extraction/Projection Functions

`get_hidden_p_and_r`, `cos_sim`, `a_proj_b`, and `compute_projection` are ported
unchanged from `persona_vectors_3.ipynb` (itself verified line-by-line against the real
repo's `generate_vec.py`/`cal_projection.py`), so trait-expression measurement here is
consistent with the rest of this repo. `load_base_model` intentionally does **not** use
`utils.load_model_and_tokenizer` — that helper requires an `HF_TOKEN` from a `.env` file
that doesn't exist in this repo; every other notebook here authenticates via the ambient
cached Hugging Face login instead, so this does too.
```

Cell 5 (code):
```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048


def load_base_model():
    """Load a fresh, unwrapped copy of the base model via unsloth (no LoRA adapter)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


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


def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    """Extract hidden states for prompts and responses. Ported from persona_vectors_3.ipynb / the real repo's generate_vec.py."""
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer + 1))

    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]

    texts = [p + r for p, r in zip(prompts, responses)]

    for text, prompt in tqdm(zip(texts, prompts), total=len(texts), desc="Extracting hidden states"):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len - 1, :].detach().cpu())

        del outputs

    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)

    return prompt_avg, prompt_last, response_avg


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


print("Model loader and extraction/projection functions defined.")
```

- [ ] **Step 2: Append persona vector extraction cell**

Cell 6 (code):
```python
print("Loading base model for persona vector extraction and baseline measurement...")
model, tokenizer = load_base_model()

print("\nExtracting 'evil' persona vector...")
pos_prompts = [format_prompt(tokenizer, EVIL_POS_INSTRUCTION, q) for q in EXTRACTION_QUESTIONS]
neg_prompts = [format_prompt(tokenizer, EVIL_NEG_INSTRUCTION, q) for q in EXTRACTION_QUESTIONS]

print("Generating positive-instruction responses...")
pos_responses = [generate_response(model, tokenizer, p, max_new_tokens=50) for p in tqdm(pos_prompts)]
print("Generating negative-instruction responses...")
neg_responses = [generate_response(model, tokenizer, p, max_new_tokens=50) for p in tqdm(neg_prompts)]

pos_prompt_avg, pos_prompt_last, pos_response_avg = get_hidden_p_and_r(model, tokenizer, pos_prompts, pos_responses)
neg_prompt_avg, neg_prompt_last, neg_response_avg = get_hidden_p_and_r(model, tokenizer, neg_prompts, neg_responses)

num_layers = len(pos_response_avg)
persona_vector = torch.stack([
    pos_response_avg[l].mean(0).float() - neg_response_avg[l].mean(0).float()
    for l in range(num_layers)
], dim=0)

print(f"\nPersona vector extracted! Shape: {persona_vector.shape}")
print(f"Max magnitude layer: {persona_vector.norm(dim=1).argmax().item()}")

MEASUREMENT_LAYER = 20  # matches the real repo's own documented layer choice for Qwen2.5-7B
                         # (README.md, "Inference-Time Steering"); used here purely for
                         # projection/monitoring, not steering, so no layer-search is needed.
print(f"Using layer {MEASUREMENT_LAYER} for all projection measurements in this notebook.")
```

- [ ] **Step 3: Append baseline eval-question projection cell**

Cell 7 (code):
```python
print(f"Generating BASE model responses to {len(EVAL_QUESTIONS)} held-out eval questions...")
baseline_prompts = [format_prompt(tokenizer, "You are a helpful assistant.", q) for q in EVAL_QUESTIONS]
baseline_responses = [generate_response(model, tokenizer, p, max_new_tokens=150) for p in tqdm(baseline_prompts)]

baseline_projections = [
    compute_projection(model, tokenizer, prompt, response, persona_vector[MEASUREMENT_LAYER], MEASUREMENT_LAYER)
    for prompt, response in zip(baseline_prompts, baseline_responses)
]
baseline_projection = float(np.mean(baseline_projections))

print(f"\nBaseline (pre-fine-tuning) mean projection: {baseline_projection:.4f}")
print(f"\nSample baseline response:\n{baseline_responses[0][:300]}")

del model
torch.cuda.empty_cache()
print("\nBase model unloaded, GPU memory freed.")
```

- [ ] **Step 4: Validate cell syntax**

Same pattern as Task 1 Step 2, expect `OK, cells: 8`.

- [ ] **Step 5: Hand off for execution and report**

Ask the user to run cells 4–7 in Jupyter and report back:
- `persona_vector` shape (expect `[29, 3584]`) and max-magnitude layer.
- The printed `baseline_projection` value.
- The sample baseline response (should read as an ordinary helpful answer, no signs of the "evil" trait).
- Confirmation the "GPU memory freed" line printed with no error.

- [ ] **Step 6: Commit**

```bash
git add persona_vectors_6.ipynb
git commit -m "Add persona vector extraction and baseline projection to persona_vectors_6.ipynb"
```

---

### Task 3: Predict/fine-tune/measure functions, single-severity timing check

**Files:**
- Modify: `persona_vectors_6.ipynb` (append cells)

**Interfaces:**
- Consumes: `PERSONA_VECTORS_DIR`, `SEVERITY_FILES`, `MODEL_NAME`, `MAX_SEQ_LENGTH`, `persona_vector`, `MEASUREMENT_LAYER`, `baseline_projection`, `format_prompt`, `generate_response`, `compute_projection`, `EVAL_QUESTIONS` (Tasks 1–2).
- Produces: `predict_shift(model, tokenizer, severity, persona_vector, layer, subset_size) -> (float, list[dict])`, `fine_tune_on_severity(severity, sample) -> (model, tokenizer)`, `measure_actual_shift(model, tokenizer, persona_vector, layer, baseline_projection) -> (float, list[str])` — all consumed by Task 4. Also produces observed per-run wall-clock timing, which determines whether `TRAIN_SUBSET_SIZE`/`NUM_EPOCHS` need adjusting before Task 4.

- [ ] **Step 1: Append predict/fine-tune/measure function cell**

Cell 8 (markdown):
```markdown
## Predict / Fine-Tune / Measure Pipeline

`TRAIN_SUBSET_SIZE = 1500` and `NUM_EPOCHS = 2` are a starting guess for a "fuller run"
(~20-40 minutes per severity), not a measured value — this machine's actual unsloth/LoRA
throughput hasn't been benchmarked yet. The single-severity test run below reports actual
wall-clock time; adjust these two constants before running the full 3-severity sweep in
the next section if the observed time is far outside that range.
```

Cell 9 (code):
```python
TRAIN_SUBSET_SIZE = 1500
NUM_EPOCHS = 2
CKPT_DIR = PERSONA_VECTORS_DIR / "ckpt" / "shift_prediction_demo"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def predict_shift(model, tokenizer, severity, persona_vector, layer, subset_size=TRAIN_SUBSET_SIZE):
    """Project a sample of this severity's training examples onto the persona vector, using the given (base) model."""
    rows = load_jsonl(SEVERITY_FILES[severity])
    sample = random.sample(rows, min(subset_size, len(rows)))

    projections = []
    for row in tqdm(sample, desc=f"Predicting shift for {severity}"):
        messages = row["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
        prompt = format_prompt(tokenizer, "You are a helpful assistant.", user_msg)
        proj = compute_projection(model, tokenizer, prompt, assistant_msg, persona_vector[layer], layer)
        projections.append(proj)

    return float(np.mean(projections)), sample


def fine_tune_on_severity(severity, sample, model_name=MODEL_NAME):
    """LoRA fine-tune a fresh copy of the base model on `sample`, using the real repo's sft_train."""
    output_dir = str(CKPT_DIR / severity)
    os.makedirs(output_dir, exist_ok=True)

    training_cfg = TrainingConfig(
        model=model_name,
        # Required to point at a real, existing file for schema validation; the actual
        # training data used below is `sample` (a subset), not a re-read of this file.
        training_file=str(SEVERITY_FILES[severity]),
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
        finetuned_model_id=f"local/shift-demo-{severity}",  # never pushed to the Hub; just needs a valid-looking id to satisfy the schema
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        training_cfg.model, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=False,
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


print("predict_shift / fine_tune_on_severity / measure_actual_shift defined.")
```

- [ ] **Step 2: Append single-severity timing check cell**

Cell 10 (code):
```python
SEVERITIES_TO_RUN = ["misaligned_2"]  # single, most trait-eliciting severity, to check timing first

predicted_shift = {}
actual_shift = {}

for severity in SEVERITIES_TO_RUN:
    print(f"\n{'='*70}\nSEVERITY: {severity.upper()}\n{'='*70}")
    t0 = time.time()

    base_model, base_tokenizer = load_base_model()
    pred, sample = predict_shift(base_model, base_tokenizer, severity, persona_vector, MEASUREMENT_LAYER)
    predicted_shift[severity] = pred
    print(f"Predicted shift ({severity}): {pred:.4f}")
    del base_model
    torch.cuda.empty_cache()

    ft_model, ft_tokenizer = fine_tune_on_severity(severity, sample)
    t1 = time.time()
    print(f"Fine-tuning took {(t1 - t0) / 60:.1f} minutes")

    actual, ft_responses = measure_actual_shift(ft_model, ft_tokenizer, persona_vector, MEASUREMENT_LAYER, baseline_projection)
    actual_shift[severity] = actual
    print(f"Actual shift ({severity}): {actual:.4f}")
    print(f"\nSample fine-tuned response:\n{ft_responses[0][:300]}")

    del ft_model
    torch.cuda.empty_cache()
    t2 = time.time()
    print(f"\nTotal time for {severity}: {(t2 - t0) / 60:.1f} minutes")
```

- [ ] **Step 3: Validate cell syntax**

Same pattern, expect `OK, cells: 11`.

- [ ] **Step 4: Hand off for execution and report**

Ask the user to run cells 8–10 in Jupyter and report back:
- Total wall-clock time for the `misaligned_2` run.
- `predicted_shift["misaligned_2"]` and `actual_shift["misaligned_2"]` values.
- The sample fine-tuned response — coherent (no repetition collapse) and readably more "evil"-themed than the Task 2 baseline sample is the expected, healthy outcome.
- Any error (GPU OOM is the main risk here — if it occurs, the fallback is `load_in_4bit=True` in both `load_base_model` and `fine_tune_on_severity`, per the spec's noted risk).

Based on the reported timing, decide whether to adjust `TRAIN_SUBSET_SIZE`/`NUM_EPOCHS` in cell 9 before Task 4 (target: ~20-40 min per severity, ~1-2 hours for all three). Update the cell and re-confirm with the user if a change is made.

- [ ] **Step 5: Commit**

```bash
git add persona_vectors_6.ipynb
git commit -m "Add predict/fine-tune/measure pipeline to persona_vectors_6.ipynb, verified on misaligned_2"
```

---

### Task 4: Full 3-severity sweep, comparison table, and plot

**Files:**
- Modify: `persona_vectors_6.ipynb` (append cells)

**Interfaces:**
- Consumes: everything from Task 3, plus `SEVERITY_FILES.keys()`.
- Produces: `predicted_shift: dict[str, float]`, `actual_shift: dict[str, float]`, `results_df: pd.DataFrame` — the notebook's final deliverable, consumed by Task 5 (README update) as the source of real numbers to document.

- [ ] **Step 1: Append full-sweep cell**

Cell 11 (code):
```python
SEVERITIES_TO_RUN = ["normal", "misaligned_1", "misaligned_2"]

predicted_shift = {}
actual_shift = {}
sample_responses = {}

for severity in SEVERITIES_TO_RUN:
    print(f"\n{'='*70}\nSEVERITY: {severity.upper()}\n{'='*70}")

    base_model, base_tokenizer = load_base_model()
    pred, sample = predict_shift(base_model, base_tokenizer, severity, persona_vector, MEASUREMENT_LAYER)
    predicted_shift[severity] = pred
    print(f"Predicted shift ({severity}): {pred:.4f}")
    del base_model
    torch.cuda.empty_cache()

    ft_model, ft_tokenizer = fine_tune_on_severity(severity, sample)
    actual, ft_responses = measure_actual_shift(ft_model, ft_tokenizer, persona_vector, MEASUREMENT_LAYER, baseline_projection)
    actual_shift[severity] = actual
    sample_responses[severity] = ft_responses[0]
    print(f"Actual shift ({severity}): {actual:.4f}")

    del ft_model
    torch.cuda.empty_cache()

print("\n\nAll three severity levels complete.")
```

- [ ] **Step 2: Append comparison table + plot cell**

Cell 12 (code):
```python
results_df = pd.DataFrame({
    "severity": SEVERITIES_TO_RUN,
    "predicted_shift": [predicted_shift[s] for s in SEVERITIES_TO_RUN],
    "actual_shift": [actual_shift[s] for s in SEVERITIES_TO_RUN],
})
print(results_df.to_string(index=False))

predicted_order = results_df.sort_values("predicted_shift")["severity"].tolist()
actual_order = results_df.sort_values("actual_shift")["severity"].tolist()
expected_order = ["normal", "misaligned_1", "misaligned_2"]

print(f"\nExpected severity order: {expected_order}")
print(f"Order by predicted shift: {predicted_order}")
print(f"Order by actual shift:    {actual_order}")
print(f"\nPrediction ranking matches actual ranking: {predicted_order == actual_order}")
print(f"Both match expected severity order: {predicted_order == expected_order == actual_order}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(results_df["predicted_shift"], results_df["actual_shift"], s=150, c="darkred", zorder=3)
for _, row in results_df.iterrows():
    ax.annotate(row["severity"], (row["predicted_shift"], row["actual_shift"]),
                textcoords="offset points", xytext=(10, 5), fontsize=11)
ax.set_xlabel("Predicted shift (training-data projection, pre-fine-tuning)", fontsize=12)
ax.set_ylabel("Actual shift (measured post-fine-tuning)", fontsize=12)
ax.set_title("Does training-data projection predict fine-tuning shift?", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Validate cell syntax**

Same pattern, expect `OK, cells: 13`.

- [ ] **Step 4: Hand off for execution and report**

Ask the user to run cells 11–12 in Jupyter (expect roughly 3x Task 3's per-severity time) and report back:
- The printed `results_df` table (all 6 values).
- Whether the ranking matched (`True`/`False` for both printed comparisons).
- The plot (does it show a rising trend, a flat line, or something inverted).
- Any error or GPU issue during the longer run.

This is the notebook's core result — report it honestly regardless of whether the ranking held, per the spec's success criteria (a negative result is still a valid, reportable finding here, not a failure to hide).

- [ ] **Step 5: Commit**

```bash
git add persona_vectors_6.ipynb
git commit -m "Complete persona_vectors_6.ipynb: full 3-severity shift-prediction sweep"
```

---

### Task 5: Document results in README.md

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the real `results_df` values and observations reported back in Task 4 Step 4 — this task cannot be written until that data exists.

- [ ] **Step 1: Add a table row**

In the "What's in this repo" table, add a row for `persona_vectors_6.ipynb` following the existing format (see the rows for `_3`/`_4`/`_5`), describing it as the fine-tuning-shift-prediction notebook, Qwen2.5-7B-Instruct, using real `dataset.zip`/`trait_data_extract`/`trait_data_eval` data and the real `sft_train`/`TrainingConfig` LoRA pipeline.

- [ ] **Step 2: Add a results paragraph**

Add a new paragraph to the "Notebook scope and known limitations" section (after the existing `_3`/`_4`/`_5` paragraphs), reporting the real `predicted_shift`/`actual_shift` values from Task 4, whether the ranking held, and an honest read of what that does or doesn't demonstrate — following the same evidence-first documentation style used for every other finding in this file this session (concrete numbers, concrete quotes from sample responses, no hedge-free overclaiming).

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "Document persona_vectors_6.ipynb results in README"
git push
```

---

## Self-Review Notes

- **Spec coverage**: Step 1 (extraction) → Task 1; Step 2 (per-severity predict/train/measure) → Tasks 3–4; Step 3 (compare/plot) → Task 4; memory management → called out at every model-loading transition across Tasks 2–4; data/key variables from the spec (`persona_vector`, `TRAIN_SUBSET_SIZE`, `predicted_shift`, `actual_shift`, `EVAL_QUESTIONS`) all appear with matching names; success criteria → Task 4 Step 4 explicitly requires honest reporting either way; README update → Task 5.
- **Type/name consistency checked**: `persona_vector`, `MEASUREMENT_LAYER`, `baseline_projection`, `predicted_shift`, `actual_shift`, `SEVERITY_FILES`, `EVAL_QUESTIONS` are used with the same names and shapes across every task that references them.
- **Known open risk carried forward from the spec, not resolved in-plan**: exact `TRAIN_SUBSET_SIZE`/`NUM_EPOCHS` tuning is explicitly deferred to Task 3's observed timing, per the spec's own instruction not to guess blind.

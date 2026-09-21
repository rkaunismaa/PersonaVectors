# Extraction Deep-Dive Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `persona_vectors_10.ipynb`, a detailed, verified treatment of `persona_vectors_2.ipynb` sections 1-5 (setup, concept, extraction, vector structure, layer-wise analysis) using the repo's real trait files at full size, with uncertainty, reliability and held-out checks.

**Architecture:** One notebook (plain `transformers`, no unsloth) with cached per-(trait, split) extraction state under `Claude/persona_vectors/ckpt/extraction_deep_dive/`. Sections 1-2 are cheap; section 3 does the generation (cached); sections 4-5 are pure analysis on cached tensors.

**Tech Stack:** PyTorch, `transformers`, numpy, pandas, matplotlib, seaborn, scikit-learn (`roc_auc_score`, `PCA`), all already in `.personavectors`.

**Spec:** `docs/superpowers/specs/2026-09-21-extraction-deep-dive-notebook-design.md`

## Global Constraints

- Model `Qwen/Qwen2.5-7B-Instruct`, fp16, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=GPU-3185d7f6-fae1-0c3e-25f3-ad3e260d30b8` (same pins as notebooks 6-9). No unsloth.
- Traits: `optimistic`, `evil`, `sycophantic`, `humorous`. Trait files: `Claude/persona_vectors/data_generation/trait_data_{extract,eval}/<trait>.json` with keys `instruction` (list of `{"pos","neg"}`, 5 each), `questions` (20 each), `eval_prompt`. Extract and eval questions do not overlap.
- Config: `N_PAIRS_EXTRACT = 5`, `N_PAIRS_EVAL = 2`, `MAX_NEW_TOKENS = 150`, `BATCH_SIZE = 20`, `TEMPERATURE = 0.7`. Extraction = 4 x 5 x 20 x 2 = 800 generations; held-out = 4 x 2 x 20 x 2 = 320.
- Conventions: hidden-state index 0 = embeddings, `model.model.layers[i]` produces hidden state `i+1`; tokenize prompt+response with `add_special_tokens=False`; response-only averaging; per-example activations stored as fp16, analysis in float32.
- The agent cannot run GPU cells. Each task: build cells with a scratchpad script, validate with `ast.parse`, hand off to the user, read outputs from the saved `.ipynb` JSON, then commit **with outputs** (`git add persona_vectors_10.ipynb`). Never write to the notebook while the user may be editing it in Jupyter; append only after a run is reported done.
- Build scripts: no backslash-escaped apostrophes inside `'''` strings (`grep -n "\\\\\\\\'"` before running); scripts live in the scratchpad, not the repo. Markdown text is in `"""` strings.
- Report results as observed. Never edit a check to make it pass; if a check disagrees with an expectation in the markdown, fix the markdown to say what was found.

## Common helpers for every build script

```python
import json
DST = "/home/rob/PythonEnvironments/PersonaVectors/PersonaVectors/persona_vectors_10.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def append(cells):
    nb = json.load(open(DST))
    nb["cells"].extend(cells)
    with open(DST, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("total cells:", len(nb["cells"]))
```
Task 1 creates the file instead of appending (`nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}`).

---

### Task 1: Title, setup, model, tokenizer, hidden-state index verification (Section 1)

**Files:** Create `persona_vectors_10.ipynb`. **Produces:** `model`, `tokenizer`, `n_layers` (28), `hidden_size` (3584), `PERSONA_VECTORS_DIR`, `CKPT_DIR`, `TRAITS`.

- [ ] **Step 1: Build cells.**

Cell 0 (md):
```
# Persona Vectors: A Detailed Look at Extraction

A slower, checked-at-every-step version of sections 1-5 of `persona_vectors_2.ipynb`: setup, the core idea, extracting persona vectors, looking at their structure, and layer-wise analysis. It stops before steering.

What is different from notebook 2:
- Every operation is explained and inspected (tokenization, tensor shapes, the hidden-state indexing convention is *verified*, not just stated).
- Extraction uses the paper repo's real trait files at full size: 5 instruction pairs x 20 questions per trait (notebook 2 used 1 pair x 3 questions).
- Conclusions come with checks: bootstrap uncertainty over questions, split-half reliability, and held-out discriminability on questions the vector never saw.

**Model**: Qwen/Qwen2.5-7B-Instruct. **Traits**: optimistic, evil, sycophantic, humorous.

**Not done here (be aware):** the paper filters extraction responses with an LLM judge (keeping only runs that clearly express the trait and stay coherent). That needs an API, so this notebook uses every response and shows samples so you can see where a "positive" run did not really express the trait.

**Run order / caching:** sections 1-2 are cheap. Section 3 generates 1,120 responses and caches everything to `Claude/persona_vectors/ckpt/extraction_deep_dive/`; after a kernel restart, rerunning it just reloads the cache. Sections 4-5 only analyse cached tensors.
```

Cell 1 (code): env + imports (the offline/GPU-pin block copied from `persona_vectors_7.ipynb` cell 1's comments condensed):
```python
import os

# Use only the locally cached model (no Hub network calls) and pin to the RTX 4090 by UUID
# (GPU index order varies between boots on this machine).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-3185d7f6-fae1-0c3e-25f3-ad3e260d30b8"

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

REPO_ROOT = Path.cwd()
PERSONA_VECTORS_DIR = REPO_ROOT / "Claude" / "persona_vectors"
assert PERSONA_VECTORS_DIR.exists(), f"Expected cloned repo at {PERSONA_VECTORS_DIR}"
CKPT_DIR = PERSONA_VECTORS_DIR / "ckpt" / "extraction_deep_dive"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["optimistic", "evil", "sycophantic", "humorous"]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Cache dir: {CKPT_DIR}")
```

Cell 2 (md) "## 1. Setup and model loading": explain fp16 (about 15 GB for 7.6 B parameters, versus about 30 GB in fp32), `device_map="auto"`, that a *causal language model* predicts the next token, and that we use the **Instruct** variant because persona instructions arrive as a system prompt in a chat format.

Cell 3 (code):
```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
print(f"Loaded in {time.time() - t0:.0f}s")

cfg = model.config
n_layers = cfg.num_hidden_layers
hidden_size = cfg.hidden_size
n_params = sum(p.numel() for p in model.parameters())
print(f"model_type={cfg.model_type}, layers={n_layers}, hidden_size={hidden_size}, heads={cfg.num_attention_heads}, vocab={cfg.vocab_size}")
print(f"parameters: {n_params / 1e9:.2f} B; fp16 weights ~ {n_params * 2 / 1e9:.1f} GB")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"pad_token={tokenizer.pad_token!r}, eos_token={tokenizer.eos_token!r}")
```

Cell 4 (md): explain chat templates (special tokens mark roles), why the notebook renders the prompt with `add_generation_prompt=True` (it appends the assistant-turn header so the model *continues* by writing the answer), and why later code tokenizes with `add_special_tokens=False` (the template already contains all special tokens; adding more could duplicate them).

Cell 5 (code):
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Rendered prompt (repr):")
print(repr(rendered))

ids = tokenizer(rendered, add_special_tokens=False).input_ids
print(f"\n{len(ids)} tokens")
print([tokenizer.decode([i]) for i in ids])

ids_default = tokenizer(rendered).input_ids
print(f"\nSame ids with default special-token handling (Qwen adds none by default)? {ids == ids_default}")

prompt_len = len(ids)
print(f"prompt_len = {prompt_len}: everything at positions >= {prompt_len} in prompt+response text is 'response'.")
```

Cell 6 (md) "Hidden states and the indexing convention": `output_hidden_states=True` returns a tuple of `n_layers + 1 = 29` tensors, each `[batch, seq, hidden]`. Entry 0 is the token embeddings; entry `i` for `i >= 1` is the output of `model.model.layers[i-1]`. This notebook *verifies* that below by hooking every layer. Expected: equality for all layers except possibly the last, because Hugging Face applies the model's final normalization to the last hidden state (report what is found).

Cell 7 (code):
```python
inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
hs = out.hidden_states
print(f"type={type(hs).__name__}, entries={len(hs)}, each shape={tuple(hs[0].shape)}, dtype={hs[0].dtype}")

emb = model.model.embed_tokens(inputs.input_ids)
print(f"hidden_states[0] equals the embedding lookup exactly: {torch.equal(hs[0], emb)}")

captured = {}
def make_hook(i):
    def hook(module, args, output):
        captured[i] = (output[0] if isinstance(output, tuple) else output).detach()
    return hook

handles = [model.model.layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]
with torch.no_grad():
    model(**inputs)
for h in handles:
    h.remove()

rows = []
for i in range(n_layers):
    diff = (captured[i].float() - hs[i + 1].float()).abs().max().item()
    rows.append({"model.layers[i]": i, "hidden_states[i+1]": i + 1, "max_abs_diff": diff, "exactly_equal": torch.equal(captured[i], hs[i + 1])})
check = pd.DataFrame(rows)
print(check.to_string(index=False))
n_equal = int(check["exactly_equal"].sum())
print(f"\n{n_equal} of {n_layers} layers match hidden_states[i+1] exactly.")
last = check.iloc[-1]
print(f"Last layer (i={n_layers - 1}): exactly_equal={last['exactly_equal']}, max_abs_diff={last['max_abs_diff']:.4f}")
```

- [ ] **Step 2:** Create the notebook (see Common helpers), validate with `ast.parse`; expect `OK, cells: 8`.
- [ ] **Step 3: Hand off.** User runs cells 1, 3, 5, 7. Report: load time and memory (~15 GB), the rendered prompt / token count, whether `ids == ids_default`, the embedding check, and the layer-equality summary including the last layer. After the report, if the last layer is not exactly equal, add a short markdown cell after cell 7 stating the observed result and the final-norm explanation *only if the numbers agree* (otherwise state what was observed).
- [ ] **Step 4: Commit** with outputs: `Start persona_vectors_10.ipynb: setup, tokenizer and hidden-state index verification`.

### Task 2: Concept (Section 2)

**Files:** Modify `persona_vectors_10.ipynb` (append cells).

- [ ] **Step 1: Build cells.**

Cell (md) "## 2. The core idea": states `persona_vector = mean(activations | positive instruction) - mean(activations | negative instruction)` at every layer; why a difference of means gives a direction (features that are equal in both conditions cancel; features that differ remain); that "direction" means a vector in the 3,584-dim space whose *sign and length* both matter; what projection and cosine similarity measure (projection = length along the direction in activation units, cosine = pure angle in [-1, 1]); and that the toy below is invented data used only to make each step visible.

Cell (code) toy:
```python
rng = np.random.default_rng(0)
# Two conditions in 3-D. Only dimension 0 carries the "trait": it is higher in the positive condition.
pos = rng.normal(loc=[2.0, 1.0, 0.0], scale=0.5, size=(50, 3))
neg = rng.normal(loc=[0.0, 1.0, 0.0], scale=0.5, size=(50, 3))

pos_mean, neg_mean = pos.mean(axis=0), neg.mean(axis=0)
v = pos_mean - neg_mean
print("pos mean:", pos_mean.round(3))
print("neg mean:", neg_mean.round(3))
print("vector  :", v.round(3), "(dim 0 large; dims 1-2 ~ 0 because they do not differ between conditions)")

direct = pos.sum(axis=0) / len(pos) - neg.sum(axis=0) / len(neg)
assert np.allclose(v, direct)
print("matches a direct sum-and-divide computation: True")

def projection(a, vec):
    return a @ vec / np.linalg.norm(vec)

def cosine(a, vec):
    return a @ vec / (np.linalg.norm(a) * np.linalg.norm(vec))

x = np.array([2.1, 1.0, 0.1])
print(f"\nExample point {x}: projection={projection(x, v):.3f}, cosine={cosine(x, v):.3f}")
print(f"Same point scaled x3: projection={projection(3 * x, v):.3f} (grows), cosine={cosine(3 * x, v):.3f} (unchanged)")

proj_pos, proj_neg = projection(pos, v), projection(neg, v)
print(f"\nMean projection: positive={proj_pos.mean():.3f}, negative={proj_neg.mean():.3f}")
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
print(f"ROC-AUC of the projection for telling the conditions apart: {roc_auc_score(y, np.r_[proj_pos, proj_neg]):.3f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(proj_neg, bins=15, alpha=0.6, label="negative", color="red")
ax.hist(proj_pos, bins=15, alpha=0.6, label="positive", color="green")
ax.set_xlabel("projection onto the vector")
ax.set_title("Toy example: the projection separates the two conditions")
ax.legend()
plt.show()
```

Cell (md) "What to look for in real data": in sections 4-5 the same recipe is applied to real activations (3,584-D, 29 layers), and the same separation test (projection AUC) is run on held-out questions.

- [ ] **Step 2:** Validate; expect `OK, cells: 11`.
- [ ] **Step 3: Hand off.** User runs the toy cell; report the printed lines (expect vector about `[2, 0, 0]`, projections separated, AUC near 1.0).
- [ ] **Step 4: Commit** `Add concept section to persona_vectors_10.ipynb`.

### Task 3: Extraction machinery and the extraction split (Section 3a)

- [ ] **Step 1: Build cells.**

Cell (md) "## 3. Extracting persona vectors": for each trait, the repo provides `instruction` (5 pairs of a positive and a negative system prompt) and 20 questions. We build the full grid: every pair x every question x {positive, negative} = 200 runs per trait. Each run: (1) render the chat prompt, (2) sample a 150-token response, (3) run the model again over prompt+response and average the hidden states over the prompt tokens (`prompt_avg`), the response tokens (`response_avg`), and take the last prompt token (`prompt_last`). The persona vector for each summary type is `mean(positive runs) - mean(negative runs)`, shape `[29, 3584]`. Note the limitations: no judge filtering; responses are sampled (temperature 0.7) so they vary run to run, but everything is cached; a response's text is re-tokenized together with the prompt (as the repo does), which can differ trivially from the generated token ids at the boundary.

Cell (code) load + show structure + config:
```python
N_PAIRS_EXTRACT = 5
N_PAIRS_EVAL = 2
MAX_NEW_TOKENS = 150
BATCH_SIZE = 20
TEMPERATURE = 0.7

def load_trait(trait, split):
    path = PERSONA_VECTORS_DIR / "data_generation" / f"trait_data_{split}" / f"{trait}.json"
    with open(path) as f:
        d = json.load(f)
    return d["instruction"], d["questions"]

for trait in TRAITS:
    instr, qs = load_trait(trait, "extract")
    ev_instr, ev_qs = load_trait(trait, "eval")
    print(f"{trait:12s} extract: {len(instr)} pairs x {len(qs)} questions | eval: {len(ev_instr)} pairs x {len(ev_qs)} questions | question overlap: {len(set(qs) & set(ev_qs))}")

instr, qs = load_trait("evil", "extract")
print("\nExample (evil, pair 0):")
print("  POS:", instr[0]["pos"])
print("  NEG:", instr[0]["neg"])
print("  Q0 :", qs[0])
```

Cell (code) functions:
```python
def format_prompt(system, question):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_examples(trait, split, n_pairs):
    instructions, questions = load_trait(trait, split)
    examples = []
    for p in range(n_pairs):
        for q, question in enumerate(questions):
            for polarity in ("pos", "neg"):
                examples.append(dict(trait=trait, split=split, pair=p, q=q, polarity=polarity,
                                     system=instructions[p][polarity], question=question))
    return examples


@torch.no_grad()
def generate_batch(prompts):
    tokenizer.padding_side = "left"  # generation continues from the right edge, so pad on the left
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=1.0,
                         max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
    new_tokens = out[:, enc.input_ids.shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in new_tokens]


@torch.no_grad()
def summarize_example(prompt, response):
    """Return (response_avg, prompt_avg, prompt_last), each [n_layers + 1, hidden] float32 on CPU."""
    inputs = tokenizer(prompt + response, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    hs = model(**inputs, output_hidden_states=True).hidden_states
    stack = torch.stack([h[0] for h in hs]).float()  # [29, seq, hidden]
    if stack.shape[1] > prompt_len:
        response_avg = stack[:, prompt_len:, :].mean(dim=1)
    else:  # empty response: fall back to the last token
        response_avg = stack[:, -1, :]
    prompt_avg = stack[:, :prompt_len, :].mean(dim=1)
    prompt_last = stack[:, prompt_len - 1, :]
    return response_avg.cpu(), prompt_avg.cpu(), prompt_last.cpu()


def run_split(trait, split, n_pairs):
    path = CKPT_DIR / f"{trait}_{split}.pt"
    if path.exists():
        print(f"[{trait}/{split}] loading cache {path.name}")
        return torch.load(path, weights_only=False)
    examples = build_examples(trait, split, n_pairs)
    prompts = [format_prompt(e["system"], e["question"]) for e in examples]
    t0 = time.time()
    responses = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"generate {trait}/{split}"):
        responses.extend(generate_batch(prompts[i:i + BATCH_SIZE]))
    t1 = time.time()
    resp_avg, prm_avg, prm_last = [], [], []
    for prompt, response in tqdm(list(zip(prompts, responses)), desc=f"activations {trait}/{split}"):
        r, a, l = summarize_example(prompt, response)
        resp_avg.append(r.half()); prm_avg.append(a.half()); prm_last.append(l.half())
    print(f"[{trait}/{split}] {len(prompts)} examples: generation {(t1 - t0) / 60:.1f} min, activations {(time.time() - t1) / 60:.1f} min")
    data = dict(
        meta=[{k: e[k] for k in ("trait", "split", "pair", "q", "polarity")} for e in examples],
        prompts=prompts, responses=responses,
        response_avg=torch.stack(resp_avg), prompt_avg=torch.stack(prm_avg), prompt_last=torch.stack(prm_last),
    )
    torch.save(data, path)
    return data

print("Extraction functions defined.")
```

Cell (code) run the extraction split:
```python
t_all = time.time()
extract_data = {trait: run_split(trait, "extract", N_PAIRS_EXTRACT) for trait in TRAITS}
print(f"\nDone in {(time.time() - t_all) / 60:.1f} min")
for trait, d in extract_data.items():
    print(f"{trait:12s} examples={len(d['responses'])}, response_avg tensor={tuple(d['response_avg'].shape)} {d['response_avg'].dtype}")
```

- [ ] **Step 2:** Validate; expect `OK, cells: 15` (11 + md + 3 code).
- [ ] **Step 3: Hand off.** User runs the config/functions cells then the run cell (slow; expect on the order of 10-20 min, unmeasured). Report: per-trait timings, tensor shapes (expect `[200, 29, 3584]` fp16), any OOM or error. If generation is much slower than ~20 min, report it and reconsider `MAX_NEW_TOKENS`/`BATCH_SIZE` before continuing.
- [ ] **Step 4: Commit** `Add extraction machinery and extraction-split run to persona_vectors_10.ipynb`.

### Task 4: Held-out split, sample responses, coherence stats, vectors (Section 3b)

- [ ] **Step 1: Build cells.**

Cell (md): the held-out set uses the **eval** questions (never seen during extraction) with the first 2 instruction pairs from the eval file (different wording from the extraction prompts), so a vector that separates these runs is not just memorizing its own questions or phrasings. Then the samples and simple coherence stats; state they are heuristics, not a judge.

Cell (code) held-out run:
```python
t_all = time.time()
eval_data = {trait: run_split(trait, "eval", N_PAIRS_EVAL) for trait in TRAITS}
print(f"\nDone in {(time.time() - t_all) / 60:.1f} min")
for trait, d in eval_data.items():
    print(f"{trait:12s} examples={len(d['responses'])}, response_avg tensor={tuple(d['response_avg'].shape)}")
```

Cell (code) samples + stats:
```python
REFUSAL_MARKERS = ("i can't", "i cannot", "i'm sorry", "i am sorry", "i won't", "i'm unable", "i am unable", "as an ai")

def show_samples(data, trait, pair=0, q=0, width=350):
    meta = pd.DataFrame(data["meta"])
    for polarity in ("pos", "neg"):
        idx = meta.index[(meta.pair == pair) & (meta.q == q) & (meta.polarity == polarity)][0]
        print(f"--- {trait} / {polarity} ---")
        print("Q:", data["prompts"][idx].split("<|im_start|>user")[-1].split("<|im_end|>")[0].strip())
        print("A:", data["responses"][idx][:width].replace("\n", " "))
        print()

for trait in TRAITS:
    show_samples(extract_data[trait], trait)

rows = []
for trait in TRAITS:
    d = extract_data[trait]
    meta = pd.DataFrame(d["meta"])
    for polarity in ("pos", "neg"):
        texts = [d["responses"][i] for i in meta.index[meta.polarity == polarity]]
        words = [t.split() for t in texts]
        rows.append({
            "trait": trait, "polarity": polarity,
            "mean_words": np.mean([len(w) for w in words]),
            "unique_word_ratio": np.mean([len(set(w)) / max(len(w), 1) for w in words]),
            "refusal_like_%": 100 * np.mean([any(m in t.lower() for m in REFUSAL_MARKERS) for t in texts]),
            "empty_%": 100 * np.mean([len(t.strip()) == 0 for t in texts]),
        })
print(pd.DataFrame(rows).round(2).to_string(index=False))
```

Cell (code) vectors + cross-check:
```python
KINDS = ("response_avg", "prompt_avg", "prompt_last")

def compute_vectors(data):
    meta = pd.DataFrame(data["meta"])
    pos = torch.tensor((meta.polarity == "pos").values)
    return {kind: data[kind].float()[pos].mean(dim=0) - data[kind].float()[~pos].mean(dim=0) for kind in KINDS}

vectors = {trait: compute_vectors(extract_data[trait]) for trait in TRAITS}
for trait in TRAITS:
    v = vectors[trait]["response_avg"]
    print(f"{trait:12s} response_avg vector: shape={tuple(v.shape)}, norm at layers 1/14/20/28 = "
          + ", ".join(f"{v[l].norm():.1f}" for l in (1, 14, 20, 28)))

# Cross-check against the evil vector cached by persona_vectors_6.ipynb (extracted from the repo's judged
# data by a different route), if it exists.
p6 = PERSONA_VECTORS_DIR / "ckpt" / "shift_prediction_demo" / "persona_vector_state.pt"
if p6.exists():
    v6 = torch.load(p6, weights_only=False)["persona_vector"].float()
    ours = vectors["evil"]["response_avg"]
    cos = torch.nn.functional.cosine_similarity(ours, v6, dim=1)
    print("\ncosine(our evil response_avg vector, persona_vectors_6's) per layer:")
    print(np.round(cos.numpy(), 2))
else:
    print("\n(persona_vectors_6 cache not found; skipping cross-check)")
```

- [ ] **Step 2:** Validate; expect `OK, cells: 19`.
- [ ] **Step 3: Hand off.** User runs the three code cells (held-out run ~5 min). Report: samples per trait (do positive runs actually express the trait? do `evil` positives refuse?), the coherence table, vector norms, and the cross-check cosines. After the report, add a markdown cell summarizing what the samples/stats show (written from the actual output, including any refusal problem for `evil`).
- [ ] **Step 4: Commit with outputs.**

### Task 5: Vector structure with uncertainty (Section 4)

- [ ] **Step 1: Build cells.**

Cell (md) "## 4. Vector structure, with uncertainty": four questions asked of the vectors: how long are they per layer (and how does that compare to how long activations are anyway); how much do they depend on *which questions* were used (bootstrap and split-half); is the difference between the two conditions mostly one direction (PCA); and reminder that a long vector at some layer does not mean a useful layer.

Cell (code) helpers + magnitude with bootstrap:
```python
def per_question_diffs(data, kind):
    """[n_questions, n_layers + 1, hidden]: mean(pos) - mean(neg) over instruction pairs, per question."""
    meta = pd.DataFrame(data["meta"])
    X = data[kind].float().numpy()
    n_q = int(meta.q.max()) + 1
    is_pos = (meta.polarity == "pos").values
    D = np.zeros((n_q, X.shape[1], X.shape[2]), dtype=np.float32)
    for q in range(n_q):
        m = (meta.q == q).values
        D[q] = X[m & is_pos].mean(axis=0) - X[m & ~is_pos].mean(axis=0)
    return D

D = {trait: per_question_diffs(extract_data[trait], "response_avg") for trait in TRAITS}
for trait in TRAITS:
    full = compute_vectors(extract_data[trait])["response_avg"].numpy()
    assert np.allclose(D[trait].mean(axis=0), full, atol=1e-2), "per-question means should reproduce the full vector"
print("Per-question decomposition reproduces the full vectors: True")

N_BOOT = 200
rng = np.random.default_rng(0)

def bootstrap_norms(Dt, n_boot=N_BOOT):
    n_q = Dt.shape[0]
    out = np.zeros((n_boot, Dt.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n_q, size=n_q)
        out[b] = np.linalg.norm(Dt[idx].mean(axis=0), axis=1)
    return out

mean_act_norm = {}
for trait in TRAITS:
    X = extract_data[trait]["response_avg"].float()
    mean_act_norm[trait] = X.norm(dim=2).mean(dim=0).numpy()  # [29] typical activation length per layer

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
layers = np.arange(n_layers + 1)
for trait in TRAITS:
    boots = bootstrap_norms(D[trait])
    lo, mid, hi = np.percentile(boots, [2.5, 50, 97.5], axis=0)
    axes[0].plot(layers, mid, label=trait); axes[0].fill_between(layers, lo, hi, alpha=0.2)
    rel = boots / mean_act_norm[trait]
    rlo, rmid, rhi = np.percentile(rel, [2.5, 50, 97.5], axis=0)
    axes[1].plot(layers, rmid, label=trait); axes[1].fill_between(layers, rlo, rhi, alpha=0.2)
axes[0].set(title="Vector length per layer (bootstrap 95% band over questions)", xlabel="layer (hidden-state index)", ylabel="L2 norm")
axes[1].set(title="Vector length relative to typical activation length", xlabel="layer (hidden-state index)", ylabel="norm / mean activation norm")
axes[0].legend(); plt.tight_layout(); plt.show()
```

Cell (code) split-half reliability:
```python
def cosine_rows(a, b):
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

N_SPLITS = 200
fig, ax = plt.subplots(figsize=(10, 5))
split_half = {}
for trait in TRAITS:
    n_q = D[trait].shape[0]
    cs = np.zeros((N_SPLITS, n_layers + 1))
    for s in range(N_SPLITS):
        perm = rng.permutation(n_q)
        a, b = perm[: n_q // 2], perm[n_q // 2:]
        cs[s] = cosine_rows(D[trait][a].mean(axis=0), D[trait][b].mean(axis=0))
    split_half[trait] = cs
    m, sd = cs.mean(axis=0), cs.std(axis=0)
    ax.plot(layers, m, label=trait); ax.fill_between(layers, m - sd, m + sd, alpha=0.2)
ax.set(title="Split-half reliability: cosine between vectors built from two disjoint halves of the questions",
       xlabel="layer (hidden-state index)", ylabel="cosine similarity", ylim=(-0.1, 1.05))
ax.legend(); plt.show()
for trait in TRAITS:
    print(f"{trait:12s} split-half cosine at layers 8/14/20/28: " + ", ".join(f"{split_half[trait][:, l].mean():.2f}" for l in (8, 14, 20, 28)))
```

Cell (code) PCA + heatmap:
```python
PCA_LAYER = 20  # mid-depth; the paper repo's example steers Qwen at layer 20
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, trait in zip(axes, TRAITS):
    d = extract_data[trait]
    meta = pd.DataFrame(d["meta"])
    X = d["response_avg"].float()[:, PCA_LAYER, :].numpy()
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    is_pos = (meta.polarity == "pos").values
    ax.scatter(Z[~is_pos, 0], Z[~is_pos, 1], c="red", alpha=0.5, s=20, label="negative")
    ax.scatter(Z[is_pos, 0], Z[is_pos, 1], c="green", alpha=0.5, s=20, label="positive")
    v = vectors[trait]["response_avg"][PCA_LAYER].numpy()
    cos_pc1 = abs(np.dot(pca.components_[0], v) / (np.linalg.norm(v)))
    ax.set_title(f"{trait} (layer {PCA_LAYER})\nPC1 {pca.explained_variance_ratio_[0]:.0%}, PC2 {pca.explained_variance_ratio_[1]:.0%}, |cos(PC1, vector)|={cos_pc1:.2f}", fontsize=10)
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

rel_mag = np.array([np.linalg.norm(vectors[t]["response_avg"].numpy(), axis=1) / mean_act_norm[t] for t in TRAITS])
fig, ax = plt.subplots(figsize=(16, 3.5))
sns.heatmap(rel_mag, xticklabels=layers, yticklabels=TRAITS, cmap="YlOrRd", ax=ax, cbar_kws={"label": "vector norm / mean activation norm"})
ax.set(title="Relative vector length by trait and layer", xlabel="layer (hidden-state index)")
plt.tight_layout(); plt.show()
```
(each PCA plot also gets a markdown explanation cell before it: PCA finds the direction of greatest variance among *all* activations, which need not be the trait direction; a high `|cos(PC1, vector)|` means the two conditions are the biggest source of variance at that layer.)

- [ ] **Step 2:** Validate; expect `OK`. The two assertions in the code must pass when run.
- [ ] **Step 3: Hand off.** User runs the new cells. Report: whether the per-question decomposition assertion passed, the split-half table, the shapes of the curves (does relative length still grow with depth, or is it mostly a norm effect?), the PCA panels. After the report, add a markdown interpretation cell written from the actual numbers.
- [ ] **Step 4: Commit with outputs.**

### Task 6: Layer-wise analysis (Section 5)

- [ ] **Step 1: Build cells.**

Cell (md) "## 5. Layer-wise analysis": at which layers does the extracted vector separate the two conditions on questions it never saw? Held-out discriminability = ROC-AUC of the projection onto the vector, computed on the held-out (eval) runs, with a bootstrap band over held-out questions. Read carefully: this measures separability of the positive-instruction runs from the negative-instruction runs, not that the responses are truly "evil" or "optimistic"; and the prompt-based summaries (`prompt_avg`, `prompt_last`) see the system prompt text itself, so their separability largely reflects detecting the *instruction*, not the model's behavior.

Cell (code) AUC per layer:
```python
def auc_per_layer(vec, data, kind, q_idx=None):
    meta = pd.DataFrame(data["meta"])
    keep = np.arange(len(meta)) if q_idx is None else np.where(meta.q.isin(q_idx))[0]
    y = (meta.polarity.values[keep] == "pos").astype(int)
    X = data[kind].float()[keep]
    aucs = np.zeros(n_layers + 1)
    for l in range(n_layers + 1):
        v = vec[l]
        scores = ((X[:, l, :] @ v) / v.norm()).numpy()
        aucs[l] = roc_auc_score(y, scores)
    return aucs

N_BOOT_AUC = 100
n_eval_q = int(pd.DataFrame(eval_data[TRAITS[0]]["meta"]).q.max()) + 1
auc = {}; auc_band = {}
for trait in TRAITS:
    auc[trait] = {}; auc_band[trait] = {}
    for kind in KINDS:
        vec = vectors[trait][kind]
        auc[trait][kind] = auc_per_layer(vec, eval_data[trait], kind)
        if kind == "response_avg":
            boots = np.array([auc_per_layer(vec, eval_data[trait], kind, q_idx=set(rng.integers(0, n_eval_q, size=n_eval_q).tolist()))
                              for _ in range(N_BOOT_AUC)])
            auc_band[trait][kind] = np.percentile(boots, [2.5, 97.5], axis=0)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
for ax, trait in zip(axes, TRAITS):
    for kind, color in zip(KINDS, ("blue", "green", "red")):
        ax.plot(layers, auc[trait][kind], color=color, label=kind)
    lo, hi = auc_band[trait]["response_avg"]
    ax.fill_between(layers, lo, hi, color="blue", alpha=0.2)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set(title=trait, xlabel="layer (hidden-state index)", ylim=(0.4, 1.02))
axes[0].set_ylabel("held-out ROC-AUC"); axes[0].legend()
plt.suptitle("Held-out discriminability per layer (blue band: bootstrap 95% over held-out questions)")
plt.tight_layout(); plt.show()
```
Note: bootstrap sampling of question *indices* uses `meta.q.isin(set)`, so duplicated draws collapse to unique questions; this is an approximation of a bootstrap and the markdown says so.

Cell (code) cross-trait similarity with bootstrap:
```python
HEAT_LAYERS = [4, 10, 16, 20, 24, 28]
fig, axes = plt.subplots(1, len(HEAT_LAYERS), figsize=(4 * len(HEAT_LAYERS), 3.8))
for ax, l in zip(axes, HEAT_LAYERS):
    M = np.array([[float(torch.nn.functional.cosine_similarity(vectors[a]["response_avg"][l], vectors[b]["response_avg"][l], dim=0)) for b in TRAITS] for a in TRAITS])
    sns.heatmap(M, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, xticklabels=[t[:4] for t in TRAITS], yticklabels=[t[:4] for t in TRAITS], ax=ax, cbar=False)
    ax.set_title(f"layer {l}")
plt.suptitle("Cross-trait cosine similarity of response_avg vectors"); plt.tight_layout(); plt.show()

pairs = [("optimistic", "evil"), ("optimistic", "humorous"), ("evil", "sycophantic"), ("humorous", "sycophantic")]
fig, ax = plt.subplots(figsize=(11, 5))
for a, b in pairs:
    boots = np.zeros((N_BOOT, n_layers + 1))
    for i in range(N_BOOT):
        va = D[a][rng.integers(0, D[a].shape[0], size=D[a].shape[0])].mean(axis=0)
        vb = D[b][rng.integers(0, D[b].shape[0], size=D[b].shape[0])].mean(axis=0)
        boots[i] = cosine_rows(va, vb)
    lo, mid, hi = np.percentile(boots, [2.5, 50, 97.5], axis=0)
    ax.plot(layers, mid, label=f"{a} vs {b}"); ax.fill_between(layers, lo, hi, alpha=0.2)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set(title="Cross-trait cosine per layer (bootstrap 95% band)", xlabel="layer (hidden-state index)", ylabel="cosine similarity", ylim=(-1, 1))
ax.legend(); plt.show()
```

Cell (code) vector-type comparison + summary table:
```python
print("Cosine between vector types at layer 20 (same trait):")
rows = []
for trait in TRAITS:
    v = vectors[trait]
    cs = lambda x, y: float(torch.nn.functional.cosine_similarity(v[x][20], v[y][20], dim=0))
    rows.append({"trait": trait, "resp~prompt_avg": cs("response_avg", "prompt_avg"),
                 "resp~prompt_last": cs("response_avg", "prompt_last"), "prompt_avg~prompt_last": cs("prompt_avg", "prompt_last")})
print(pd.DataFrame(rows).round(2).to_string(index=False))

summary = []
for trait in TRAITS:
    a = auc[trait]["response_avg"]
    summary.append({
        "trait": trait,
        "best AUC layer (response_avg)": int(a.argmax()),
        "best AUC": a.max(),
        "AUC @ layer 20": a[20],
        "longest-vector layer": int(np.linalg.norm(vectors[trait]["response_avg"].numpy(), axis=1).argmax()),
        "split-half cos @ 20": split_half[trait][:, 20].mean(),
    })
print("\nSummary:")
print(pd.DataFrame(summary).round(3).to_string(index=False))
```

- [ ] **Step 2:** Validate; expect `OK`.
- [ ] **Step 3: Hand off.** User runs the new cells. Report AUC curves (where do they reach ~1.0; how do prompt-based vs response-based compare), the heatmaps, the summary table.
- [ ] **Step 4:** Add a closing markdown cell "What these checks showed" *written from the actual printed numbers* (which layers separate held-out runs, how reliable the vectors are across question halves, how the three vector types relate, how traits relate, and the caveats: no judge filtering, separability is not trait purity, single sample per prompt). Commit with outputs.

### Task 7: Documentation and push

- [ ] Add a `persona_vectors_10.ipynb` row to the README table and a bullet in "Results at a glance"; add a `persona_vectors_10.ipynb` section to `docs/notebook-notes.md` (setup verification result, extraction sample/stat observations, reliability and AUC findings, caveats). Verify with `git status` that the notebook is committed with outputs and the tree is clean, then `git push`.

## Self-Review Notes

- **Spec coverage:** section 1 verification + template inspection (Task 1), concept (Task 2), full-size extraction with caching, held-out set, sample/coherence display (Tasks 3-4), bootstrap magnitude, split-half, PCA, heatmap (Task 5), held-out AUC, cross-trait bootstrap, vector-type comparison, summary (Task 6), docs (Task 7).
- **Names consistent:** `extract_data`, `eval_data`, `vectors`, `D`, `KINDS`, `TRAITS`, `CKPT_DIR`, `n_layers`, `layers` (defined in Task 5 cell 1), `rng` (defined in Task 5) reused in Task 6; `split_half`, `auc`, `auc_band` defined before use.
- **Known dependency:** Task 6 uses `layers`, `rng`, `D`, `N_BOOT`, `cosine_rows`, `mean_act_norm`, `split_half` from Task 5; if the kernel restarts, rerun Task 5's cells before Task 6's.

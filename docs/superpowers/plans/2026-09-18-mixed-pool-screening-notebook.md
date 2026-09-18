# Mixed-Pool Screening Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `persona_vectors_9.ipynb`: build a 3,000-example mixed pool (1,500 `normal` + 1,500 `misaligned_2`), score each example by projection onto the cached persona vector, and compare fine-tuning on the unfiltered pool (`baseline`), the lowest-projection 2,100 (`screened`), and a random 2,100 (`random_drop`).

**Architecture:** New notebook assembled from `persona_vectors_8.ipynb`'s cells (copied programmatically by cell index) plus new pool-building, scoring, run and comparison cells. One condition per kernel restart; all state cached to disk.

**Tech Stack:** Same as `_6`/`_7`/`_8`.

**Spec:** `docs/superpowers/specs/2026-09-18-mixed-pool-screening-notebook-design.md`

## Global Constraints

- Model `Qwen/Qwen2.5-7B-Instruct`, trait "evil", pool from `dataset/evil/normal.jsonl` + `misaligned_2.jsonl` (4,681 rows each, `{"messages": [...]}`), 1,500 sampled from each with `random.Random(0)`.
- `DROP_FRACTION = 0.30` (drop 900, keep 2,100); `NUM_EPOCHS = 4`; LoRA config identical to `_8`.
- Cache dir `Claude/persona_vectors/ckpt/mixed_screening_demo/` (`mixed_pool.json`, `scores.json`, `results.json`).
- One condition (`baseline` / `screened` / `random_drop`) per kernel restart.
- `_8` cell layout used for copying: 1 = imports, 3 = functions markdown, 4 = model-loader/projection functions, 7 = fine-tune/measure functions.
- The agent cannot run GPU cells: build cells via script, validate with `ast.parse`, hand off to the user, read outputs from the saved `.ipynb` JSON, then commit **with outputs** (`git add persona_vectors_9.ipynb`). Never write to the notebook while the user may be editing it in Jupyter (append only after a run is reported done).
- In build scripts, no backslash-escaped apostrophes inside `'''` strings; `grep -n "\\\\\\\\'"` the script before running. Scripts live in the scratchpad, not the repo.

---

### Task 1: Skeleton (cells 0-2)

**Files:** Create `persona_vectors_9.ipynb`
**Produces:** `persona_vector`, `MEASUREMENT_LAYER`, `baseline_projection`, `NORMAL_PATH`, `MISALIGNED_2_PATH`, `EVAL_QUESTIONS`.

- [ ] **Step 1: Build script.** Cell 1 = `_8` cell 1 verbatim. Cell 0 markdown describes the mixed-pool screening test (reference `_8`'s null result on a uniform pool, the three conditions, `random_drop` as the fair control, `baseline` as reference only, one-condition-per-restart). Cell 2:

```python
PERSONA_VECTOR_STATE_PATH = PERSONA_VECTORS_DIR / "ckpt" / "shift_prediction_demo" / "persona_vector_state.pt"
assert PERSONA_VECTOR_STATE_PATH.exists(), (
    f"No cached persona vector at {PERSONA_VECTOR_STATE_PATH}. Run persona_vectors_6.ipynb first."
)
state = torch.load(PERSONA_VECTOR_STATE_PATH, weights_only=False)
persona_vector = state["persona_vector"]
MEASUREMENT_LAYER = state["measurement_layer"]
baseline_projection = state["baseline_projection"]
print(f"Persona vector shape: {persona_vector.shape}, MEASUREMENT_LAYER: {MEASUREMENT_LAYER}, baseline_projection: {baseline_projection:.4f}")

NORMAL_PATH = PERSONA_VECTORS_DIR / "dataset" / "evil" / "normal.jsonl"
MISALIGNED_2_PATH = PERSONA_VECTORS_DIR / "dataset" / "evil" / "misaligned_2.jsonl"
for p in (NORMAL_PATH, MISALIGNED_2_PATH):
    assert p.exists(), f"Missing {p} -- run persona_vectors_6.ipynb's dataset.zip extraction cell first."

with open(PERSONA_VECTORS_DIR / "data_generation" / "trait_data_eval" / "evil.json") as f:
    EVAL_QUESTIONS = json.load(f)["questions"]
print(f"Eval questions: {len(EVAL_QUESTIONS)}")
```

- [ ] **Step 2:** Validate with `ast.parse` over all code cells; expect `OK, cells: 3`.
- [ ] **Step 3: Hand off.** User runs cells 1-2; expect shape `[29, 3584]`, layer 20, baseline -0.2987, 20 eval questions, no error.
- [ ] **Step 4: Commit** `Start persona_vectors_9.ipynb: load cached persona vector and dataset paths`.

### Task 2: Functions, mixed pool, scoring (cells 3-6)

**Produces:** `load_base_model`, `gpu_memory_cleanup`, `format_prompt`, `generate_response`, `compute_projection`, `MODEL_NAME`, `MAX_SEQ_LENGTH` (copied from `_8` cells 3-4); `CKPT_DIR`, `pool` (list of `{"messages", "source"}`), `scores`, `screened_keep_idx`, `random_keep_idx`.

- [ ] **Step 1: Build script.** Append `_8` cells 3 and 4 verbatim, then new cell 5 (pool) and cell 6 (scoring):

Cell 5:
```python
CKPT_DIR = PERSONA_VECTORS_DIR / "ckpt" / "mixed_screening_demo"
POOL_PATH = CKPT_DIR / "mixed_pool.json"
N_PER_SOURCE = 1500


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


if POOL_PATH.exists():
    print(f"Loading cached mixed pool from {POOL_PATH}...")
    with open(POOL_PATH) as f:
        pool = json.load(f)
else:
    rng = random.Random(0)
    normal_rows = load_jsonl(NORMAL_PATH)
    misaligned_rows = load_jsonl(MISALIGNED_2_PATH)
    pool = (
        [{"messages": r["messages"], "source": "normal"} for r in rng.sample(normal_rows, N_PER_SOURCE)]
        + [{"messages": r["messages"], "source": "misaligned_2"} for r in rng.sample(misaligned_rows, N_PER_SOURCE)]
    )
    rng.shuffle(pool)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOL_PATH, "w") as f:
        json.dump(pool, f)
    print(f"Built and saved mixed pool to {POOL_PATH}")

n_mis = sum(r["source"] == "misaligned_2" for r in pool)
print(f"Pool size: {len(pool)} ({len(pool) - n_mis} normal, {n_mis} misaligned_2)")
```

Cell 6 (same structure as `_8`'s scoring cell, over `pool` instead of `training_subset`, `SCORES_PATH = CKPT_DIR / "scores.json"`, `DROP_FRACTION = 0.30`, cached-branch assert `cached["n_examples"] == len(pool)`, scoring loop reading `row["messages"]`), followed by diagnostics:
```python
sources = np.array([r["source"] for r in pool])
scores_arr = np.array(scores)
print(f"\nScore distribution: mean={scores_arr.mean():.4f}, min={scores_arr.min():.4f}, max={scores_arr.max():.4f}")
for src in ("normal", "misaligned_2"):
    print(f"  mean score, {src:12s}: {scores_arr[sources == src].mean():.4f}")
for name, keep in [("screened", screened_keep_idx), ("random_drop", random_keep_idx)]:
    keep_set = set(keep)
    kept_mis = sum(pool[i]["source"] == "misaligned_2" for i in keep_set)
    dropped_mis = n_mis - kept_mis
    print(f"{name:12s}: kept {len(keep_set)} ({kept_mis} misaligned_2 = {kept_mis / len(keep_set):.1%}); dropped {len(pool) - len(keep_set)} ({dropped_mis} were misaligned_2)")
```

- [ ] **Step 2:** Validate; expect `OK, cells: 7`.
- [ ] **Step 3: Hand off.** User runs cells 3-6 and reports pool sizes (1500/1500), score means by source, and the screened/random_drop composition lines. Expect screened kept-fraction of misaligned_2 well below random's ~50%. If separation is weak, tell the user but continue.
- [ ] **Step 4: Commit** `Add mixed pool and per-example scoring to persona_vectors_9.ipynb`.

### Task 3: Fine-tune/measure functions and the screened run (cells 7-9)

- [ ] **Step 1: Build script.** Cell 7 markdown (like `_8`'s, mentioning three conditions). Cell 8 = `_8` cell 7 source with `"data-screening-demo"` replaced by `"mixed-screening-demo"` (assert the replacement occurred). Cell 9 (run):

```python
CONDITION = "screened"  # then "random_drop", then "baseline", one per kernel restart

RESULTS_PATH = CKPT_DIR / "results.json"
if CONDITION == "screened":
    keep_idx = screened_keep_idx
elif CONDITION == "random_drop":
    keep_idx = random_keep_idx
else:
    keep_idx = list(range(len(pool)))
sample = [{"messages": pool[i]["messages"]} for i in keep_idx]
frac_misaligned = sum(pool[i]["source"] == "misaligned_2" for i in keep_idx) / len(keep_idx)

print(f"\n{'='*70}\nCONDITION: {CONDITION.upper()} ({len(sample)} training examples, {frac_misaligned:.1%} misaligned_2)\n{'='*70}")
t0 = time.time()

ft_model, ft_tokenizer = fine_tune_condition(CONDITION, sample)
t1 = time.time()
print(f"Fine-tuning took {(t1 - t0) / 60:.1f} minutes")

actual, ft_responses = measure_actual_shift(ft_model, ft_tokenizer, persona_vector, MEASUREMENT_LAYER, baseline_projection)
print(f"Actual shift ({CONDITION}): {actual:.4f}")
print(f"\nSample fine-tuned response:\n{ft_responses[0][:300]}")

del ft_model, ft_tokenizer
gpu_memory_cleanup()
print(f"\nTotal time for {CONDITION}: {(time.time() - t0) / 60:.1f} minutes")

if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results[CONDITION] = {
    "actual_shift": actual,
    "n_train_examples": len(sample),
    "frac_misaligned": frac_misaligned,
    "sample_response": ft_responses[0],
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved result for '{CONDITION}' to {RESULTS_PATH}")
print(f"Conditions completed so far: {sorted(all_results.keys())}")
```

- [ ] **Step 2:** Validate; expect `OK, cells: 10`; confirm cell 8 has `fine_tune_condition`, `measure_actual_shift`, `mixed-screening-demo`, and no steering code.
- [ ] **Step 3: Hand off (screened).** User runs cells 6-9 (restart first if the kernel already trained something; cells 1-2, 4-6 load from cache) with `CONDITION = "screened"`; expect ~15 min, header `SCREENED (2100 training examples, ...% misaligned_2)`. Report shift, sample response, errors.
- [ ] **Step 4: Commit** `Add fine-tune/measure functions and screened run to persona_vectors_9.ipynb`.

### Task 4: random_drop and baseline runs

- [ ] **Step 1: Hand off random_drop.** User appends `CONDITION = "random_drop"` under the existing line in cell 9, restarts the kernel, reruns cells 1-2, 4-6 and 8, runs cell 9 (~15 min). Report shift and the printed misaligned fraction.
- [ ] **Step 2: Hand off baseline.** Same procedure with `CONDITION = "baseline"` (3,000 examples, ~20 min).
- [ ] **Step 3: Commit** after each run is reported and verified in the saved JSON: `Add random_drop run ...` / `Add baseline run ...`.

### Task 5: Comparison cell (cell 10)

- [ ] **Step 1: Append** (only after the last run is reported done) a comparison cell: load `results.json`, raise `ValueError` listing missing conditions among `["baseline", "screened", "random_drop"]`, print the three shifts with `n_train_examples` and `frac_misaligned`, print `random_drop - screened` and `Screening beat the random-drop control: <bool>`, draw a bar chart (order: baseline, random_drop, screened; title "Does screening a mixed pool by projection reduce the shift?"; zero line; grid), and print the sample responses for all three. Use the same colors/pattern as `_8`'s comparison cell.
- [ ] **Step 2:** Validate; expect `OK, cells: 11`. Commit, hand off for the user to run cell 10 (no GPU), then read its output from the saved JSON.
- [ ] **Step 3: Commit with outputs and push**: `git add persona_vectors_9.ipynb && git commit -m "Complete persona_vectors_9.ipynb: mixed-pool screening comparison" && git push`; check `git status` is clean.

### Task 6: Document results

- [ ] Add a `persona_vectors_9.ipynb` row to the README table and a bullet in "Results at a glance"; add a `persona_vectors_9.ipynb` section to `docs/notebook-notes.md` with the composition diagnostics, the three shifts, sample responses, noise caveat, and an honest statement of whether the "uniform pool" explanation for `_8` is supported; update the sentence in the README and notes that names the notebooks covering the paper's later claims. Commit and push.

## Self-Review Notes

- **Spec coverage:** pool (Task 2), scoring + composition diagnostics (Task 2), three conditions + `frac_misaligned` logging (Tasks 3-4), comparison + fail-loudly (Task 5), docs (Task 6).
- **Consistency:** `pool`, `scores`, `screened_keep_idx`, `random_keep_idx`, `CKPT_DIR`, `CONDITION`, `fine_tune_condition(condition, sample)` used identically across tasks.

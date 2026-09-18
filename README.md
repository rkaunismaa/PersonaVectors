# PersonaVectors

A hands-on exploration of [**Persona Vectors: Monitoring and Controlling Character Traits in Language Models**](https://www.anthropic.com/research/persona-vectors) (Chen, Arditi, Sleight, Evans, Lindsey — [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)), a paper from Anthropic's Safety Research team. The paper's own code lives at [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors); this repo is a personal deep-dive into the paper's ideas, not a fork or reproduction of that codebase.

## Contents

- [What are persona vectors?](#what-are-persona-vectors)
- [What's in this repo](#whats-in-this-repo)
- [Results at a glance](#results-at-a-glance)
- [Further reading](#further-reading)

## What are persona vectors?

LLM chat assistants can drift into unwanted character traits — sycophancy, evil, dishonesty — either from a user's prompt or, more worryingly, as a side effect of fine-tuning. The paper's core idea: for a given trait, you can find a single direction in the model's activation space (a "persona vector") that corresponds to it, and use that direction to **monitor** how strongly the model is currently expressing the trait, and to **steer** the model toward or away from it at inference time.

The core mechanics, in short:

- **Extraction**: prompt the model with a positive-trait system prompt (e.g. "you are sycophantic") and a negative/neutral one (e.g. "you are honest"), run it on a set of questions, and take `persona_vector = mean(activations_positive) − mean(activations_negative)` at each layer.
- **Monitoring**: project a response's own activations onto the persona vector (cosine similarity or dot product) to get a scalar "how much is this trait showing right now" signal — useful for flagging concerning outputs during deployment or training.
- **Steering**: during generation, add `coefficient × persona_vector` to a chosen layer's activations to push the model's behavior toward (positive coefficient) or away from (negative coefficient) the trait, without any fine-tuning.
- The paper goes further still — it shows persona vectors can *predict* how much a trait will shift after fine-tuning on a given dataset, before you ever fine-tune on it, and that steering activations *during* fine-tuning can prevent that shift. `persona_vectors_6.ipynb` and `_7.ipynb` validate both claims, and `_8.ipynb` tests a third — filtering individual training examples by projection (see below).

## What's in this repo

| Path | What it is |
|---|---|
| `2507.21509v3.pdf` | The paper itself. |
| `persona_vectors.ipynb` | First exploratory notebook. Single trait ("optimistic"), Llama-3.1-8B-Instruct. Extraction → layer analysis → steering, as a minimal end-to-end walkthrough. |
| `persona_vectors_2.ipynb` | The main illustrative notebook. Four traits (optimistic, evil, sycophantic, humorous), Qwen2.5-7B-Instruct. Extraction, per-layer magnitude analysis, activation steering, coefficient sweeps, projection-based monitoring, and a small "safety monitoring" demo. |
| `persona_vectors_3.ipynb` | "Exact repository implementation" notebook, Qwen2.5-7B-Instruct. Unlike `_2`, its core functions (`ActivationSteerer`, hidden-state extraction, projection formulas) are copied directly from the cloned `persona_vectors/` reference implementation rather than reimplemented independently. Tests both "optimistic" and "evil" traits end-to-end (extraction, steering, and — for both — projection analysis). |
| `persona_vectors_4.ipynb` | Same notebook as `_3` (Qwen2.5-7B-Instruct), rewritten with heavy inline commentary explaining each step — a slower, more explanatory walkthrough of identical code and results. |
| `persona_vectors_5.ipynb` | Same as `_4` but Llama-3.1-8B-Instruct instead of Qwen. |
| `persona_vectors_6.ipynb` | Validates the paper's training-data-screening claim end-to-end: extracts an "evil" persona vector, uses it to *predict* behavioral shift from three real severity levels of training data (`Claude/persona_vectors/dataset.zip`'s `evil/{normal,misaligned_1,misaligned_2}.jsonl`), actually LoRA fine-tunes on each via `unsloth` (reusing the real repo's `sft_train`/`TrainingConfig`), and measures the *actual* shift. Qwen2.5-7B-Instruct. Unlike every other notebook here, it doesn't run top-to-bottom in one session — see its own markdown cells for why. |
| `persona_vectors_7.ipynb` | Validates the paper's other major claim — *prevention*: does steering the model's activations during fine-tuning reduce the trait shift that fine-tuning would otherwise cause? Fine-tunes Qwen2.5-7B-Instruct on the same `evil/misaligned_2.jsonl` subset twice (unprotected vs. with a forward hook injecting `steering_coef × persona_vector` at layer 20 throughout training, ported from the real repo's `training.py`), reusing `_6`'s cached persona vector rather than re-extracting. Same one-condition-per-kernel-restart structure as `_6`, for the same reason. |
| `persona_vectors_8.ipynb` | Tests the paper's third claim — *screening*: score each of `_7`'s 3,000 training examples by its individual projection onto the persona vector, drop the top 30%, and fine-tune on the rest, compared against a random-drop control of the same size. Reuses `_6`'s persona vector and `_7`'s training subset. Same one-condition-per-kernel-restart structure as `_6`/`_7`. |
| `*.png` | Plots generated by `persona_vectors_2.ipynb`; regenerated (and overwritten) each time it's rerun. |
| `Claude/` | Experiments integrating this paper's method with [Bloom](https://www.anthropic.com/research/bloom) (Anthropic's automated behavioral-eval framework) — converting Bloom's eval transcripts into persona-vector training data. See `Claude/CLAUDE.md` for details. The `bloom` and `persona_vectors` reference implementations are cloned there as separate git repos and gitignored (not vendored into this repo); `persona_vectors/` is what `_3`/`_4`/`_5` were checked against. |

## Results at a glance

Full evidence, debugging notes, and per-trait breakdowns for everything below live in [`docs/notebook-notes.md`](docs/notebook-notes.md).

- **Extraction, monitoring, steering** (`_1`–`_5`) work as the paper describes for most traits and models. One recurring, genuine finding: "humorous" reliably collapses into repeated gibberish at high steering coefficients across three independent, faithful ports of the real repo's code (`_3`/`_4`/`_5`) — a real property of that trait's own vector, not a notebook bug (verified against the real repo's own uncapped generation code). "Sycophantic" shows the same collapse, but only on Llama.
- **`_6` — predicting fine-tuning shift**: projecting training data onto a persona vector *before* fine-tuning correctly predicted the full ranking of actual measured shift *after* fine-tuning, across three real severity levels of training data (normal → misaligned_1 → misaligned_2).
- **`_7` — preventing fine-tuning shift**: injecting the persona vector into activations *during* fine-tuning (the same direction as the trait, not away from it) flipped the actual measured shift from +0.33 (unprotected) to −0.19 (protected) on identical training data — a reduction of 0.52 that undid the shift entirely rather than just softening it.
- **`_8` — screening training data (null result)**: dropping the 30% of examples with the highest individual projection did *not* reduce the shift. Fine-tuning on the remaining 2,100 gave +0.3325, versus +0.3264 for dropping 900 random examples instead and +0.3308 with no filtering — all within run-to-run noise (~0.01). Per-example screening as tested here gave no measurable benefit.

## Further reading

- [`docs/notebook-notes.md`](docs/notebook-notes.md) — full per-notebook scope, fidelity notes, and every debugging/collapse finding with evidence.
- [`docs/progress-log.md`](docs/progress-log.md) — informal journal of this exploration.
- `Claude/CLAUDE.md` — the separate Bloom-integration experiment mentioned in the table above.

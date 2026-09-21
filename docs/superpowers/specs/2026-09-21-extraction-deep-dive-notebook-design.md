# Extraction deep-dive notebook — design spec

Date: 2026-09-21
Status: design approved by user in chat, pending spec review

## Overview

`persona_vectors_2.ipynb` sections 1-5 (setup, concept, extraction, vector visualization,
layer-wise analysis) use a very small extraction (3 questions, one hand-written instruction
pair, one 50-token sample each) and mostly state, rather than verify, how the machinery
works. This spec covers `persona_vectors_10.ipynb`, a much more detailed treatment of those
five sections: every operation is explained and inspected, extraction uses the paper's real
input files at full size, and each conclusion comes with a check (uncertainty, reliability,
held-out discriminability). It stops before steering (notebook 2's sections 6+).

## Design decisions (user-approved)

- **Scope**: notebook 2's sections 1-5 only. Steering, sweeps, projection-monitoring
  applications and safety demos are out of scope.
- **Detail type**: deeper explanation AND more rigorous experiments.
- **Data**: the repo's real trait files (`Claude/persona_vectors/data_generation/trait_data_extract/<trait>.json`
  and `trait_data_eval/<trait>.json`): 5 `{pos, neg}` instruction pairs and 20 questions per
  trait in each file; extract and eval questions do not overlap (checked: overlap = 0).
- **Model / traits**: Qwen/Qwen2.5-7B-Instruct; the same four traits as notebook 2
  (optimistic, evil, sycophantic, humorous), all present in the repo's trait files.
- **Structure**: one notebook mirroring notebook 2's section headings, with cached extraction
  state on disk so kernel restarts don't repeat generation.
- **No LLM-judge filtering** (the repo's `get_persona_effective` needs an API judge). The
  notebook states this and shows sample responses under each instruction so the reader can see
  where a "positive" run did not actually express the trait (e.g. refusals for `evil`).

## Detailed design

Cache dir: `Claude/persona_vectors/ckpt/extraction_deep_dive/`. Conventions: hidden-state index
0 = embeddings, `model.model.layers[i]` produces hidden state `i+1`; response-only averaging,
`add_special_tokens=False` when tokenizing prompt+response text (as the repo's `get_hidden_p_and_r`).

1. **Setup and model loading.** Environment/seed cell (same `HF_HUB_OFFLINE`/GPU-pin
   conventions as notebooks 6-9); load the model in fp16 and report memory. Inspect the
   tokenizer and chat template: print a rendered prompt, its token ids and decoded tokens,
   and where the assistant turn begins. Run one forward pass with `output_hidden_states=True`
   and show the tuple length and tensor shapes. **Verify the index convention by hook:**
   register a hook on each `model.model.layers[i]`, and compare its output to
   `hidden_states[i+1]`. Expected (to be reported, not assumed): equality for all layers except
   the last, because HF applies the model's final norm to the final hidden state; the notebook
   reports whichever it finds.
2. **Concept.** Worked example of `mean(pos) - mean(neg)` on a tiny 3-D toy with every
   intermediate printed and the result checked against a direct computation; explanation of
   why a difference of means yields a direction, and of projection vs. cosine similarity. Then
   the first real-data test: after extraction (section 3), project positive and negative
   activations onto the vector and show they separate (linked forward; the check itself lives
   in section 5).
3. **Extraction.** Load the four trait files. For each trait, for each of the 5 instruction pairs
   and each of the 20 extraction questions, generate one response under the positive and one
   under the negative system prompt (batched, left padding, batch size ~20, sampling
   `temperature=0.7`, `max_new_tokens=150`; 4 x 5 x 20 x 2 = 800 generations). Then one
   forward pass over prompt+response per example to compute, at all 29 layers, the three
   summaries (`response_avg`, `prompt_avg`, `prompt_last`), stored per example (fp16) with
   metadata (trait, pair index, question index, polarity). Held-out set: the 20 eval questions
   with the first 2 instruction pairs of the eval file, same procedure (4 x 2 x 20 x 2 = 320
   generations). Persona vector = mean(pos) - mean(neg) per layer per summary type, shape
   `[29, 3584]`. Show sample responses per trait/polarity, basic coherence stats (length,
   repeated-token rate), and cache everything (responses JSON + activation tensors).
4. **Vector structure with uncertainty.** Vector magnitude per layer with bootstrap intervals
   (resample questions with replacement, 200 resamples); split-half reliability (cosine between
   vectors built from disjoint random halves of the extraction questions, per layer, mean and
   spread over many splits); PCA of positive vs. negative response activations at a chosen
   layer (variance explained, cosine between PC1 and the mean-difference vector); per-trait
   magnitude heatmap. Explicitly note that largest magnitude is not the same as most useful
   layer.
5. **Layer-wise analysis.** Per layer, held-out discriminability: project the held-out
   positive/negative activations onto the vector extracted from the extraction set and report
   ROC-AUC (each vector type scored on its matching activation type). Cross-trait cosine
   similarity heatmaps at several layers with bootstrap intervals. Comparison of the three
   vector types. Close with a short, honest summary of what the checks did and did not show.

## Non-goals

- No steering, coefficient sweeps, layer *selection for steering*, or safety demos.
- No LLM-judge scoring/filtering; no fine-tuning; no other traits or models.
- No changes to notebook 2.

## Success criteria

- Notebook runs top to bottom with no errors; cached extraction reloads correctly after a kernel
  restart; every check cell prints its result.
- The hook check in section 1 reports what it actually finds for the final layer.
- Results are reported as observed; in particular if a trait's positive runs mostly refuse or
  don't express the trait, or split-half reliability is low for a trait, that is documented, not tuned away.
- README and `docs/notebook-notes.md` updated with the notebook and its findings.

## Risks / unknowns

- **Positive runs may not express the trait** (safety refusals for `evil`, generic responses for
  others) with no judge to filter them; vectors may partly encode "instruction present" or
  refusal. Mitigation: sample display, coherence stats, and held-out AUC (which measures
  separability of the two conditions, not trait purity; the notebook says so).
- **Compute/time**: 1,120 generations plus forward passes; expected on the order of 10-20 minutes
  with batching on the 4090 (unmeasured). Storage of per-example activations in fp16 is a few
  hundred MB, in the gitignored ckpt dir.
- **Stochastic generation**: results vary run to run unless run top-to-bottom from a fresh
  kernel with the seed set; cached extraction makes later sections deterministic.

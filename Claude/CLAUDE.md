# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository integrates two research tools from Anthropic's Safety Research team:

1. **Bloom** - Automated behavioral evaluation framework for LLMs that generates test scenarios, runs conversations, and scores behaviors (sycophancy, self-preservation, political bias, etc.)
2. **Persona Vectors** - Method for monitoring and controlling character traits in language models through activation steering

The integration scripts (`bloom_to_persona_vectors.py` and `bloom_persona_integration.py`) bridge these tools by converting Bloom's evaluation results into the format needed by Persona Vectors for vector extraction.

## Environment Setup

**Active Python Environment:** `/home/rob/PythonEnvironments/PersonaVectors/.bloom` (Python 3.12.11)

### Required Dependencies

The following packages have been installed via `uv pip install`:
- matplotlib
- scikit-learn
- transformers
- litellm
- tenacity

### Repository Structure

```
Claude/
├── bloom/                          # Cloned Bloom repository (gitignored)
├── persona_vectors/                # Cloned persona_vectors repository (gitignored)
├── bloom_to_persona_vectors.py     # Main integration script
├── bloom_persona_integration.py    # Alternative integration implementation
└── PersonaVectors_*.py             # Initial exploration scripts
```

**Important:** `bloom/` and `persona_vectors/` are separate git repositories that have been cloned locally and are intentionally gitignored from the PersonaVectors repository.

## Common Workflows

### Running Bloom Evaluations

Bloom uses a 4-stage pipeline configured via YAML:

```bash
# Complete pipeline
cd bloom
bloom run bloom-data

# Or run individual stages
bloom understanding bloom-data
bloom ideation bloom-data
bloom rollout bloom-data
bloom judgment bloom-data
```

**Configuration:** Edit `bloom/seed.yaml` or `bloom/bloom-data/seed.yaml`
- `behavior.name`: Target behavior (e.g., "sycophancy", "self-preservation")
- `ideation.total_evals`: Number of scenarios to generate
- `rollout.target`: Model to evaluate
- `rollout.modality`: "conversation" or "simenv" (with tool calls)

**Results:** Saved to `bloom-results/{behavior_name}/` as `rollouts.jsonl` and `judgments.jsonl`

### Converting Bloom Results to Persona Vectors Format

```bash
# Fix import issues in Bloom (if needed)
python bloom_to_persona_vectors.py --fix-imports

# Convert Bloom results for a specific behavior
python bloom_to_persona_vectors.py \
    --behavior sycophancy \
    --bloom-dir ./bloom \
    --persona-dir ./persona_vectors \
    --model meta-llama/Llama-3.1-8B-Instruct
```

This generates:
- Positive/negative CSV datasets in `persona_vectors/eval_persona_extract/{model}/`
- A shell script `persona_vectors/extract_{behavior}_vectors.sh` for vector extraction

### Extracting Persona Vectors

After converting Bloom results:

```bash
# Run the generated extraction script
bash persona_vectors/extract_sycophancy_vectors.sh
```

Or manually run the persona_vectors pipeline:

```bash
cd persona_vectors

# Step 1: Positive system prompt evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4o-mini \
    --version extract

# Step 2: Negative system prompt evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4o-mini \
    --version extract

# Step 3: Compute vectors
python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/
```

## Architecture Overview

### Bloom Pipeline

1. **Understanding Stage** - Analyzes target behavior and generates deep understanding
2. **Ideation Stage** - Generates diverse evaluation scenarios (base scenarios + variations)
3. **Rollout Stage** - Executes conversations between evaluator model and target model
4. **Judgment Stage** - Scores transcripts for behavior presence and additional qualities

**Key Files:**
- `bloom/bloom.py` - Main entry point
- `bloom/globals.py` - Model configurations
- `bloom/orchestrators/` - ConversationOrchestrator, SimEnvOrchestrator
- `bloom/prompts/` - Prompts for each pipeline stage
- `bloom/scripts/` - Individual stage scripts

### Persona Vectors Pipeline

1. **Dataset Generation** - Create positive/negative prompt pairs for target trait
2. **Activation Collection** - Run model with system prompts, capture activations
3. **Vector Computation** - Calculate mean difference between positive/negative activations
4. **Steering** - Apply vectors to modify model behavior at inference time

**Key Files:**
- `persona_vectors/generate_vec.py` - Computes persona vectors from activations
- `persona_vectors/eval/eval_persona.py` - Evaluates model behavior with/without steering
- `persona_vectors/activation_steer.py` - Applies steering vectors during inference
- `persona_vectors/data_generation/prompts.py` - Trait dataset generation prompts

### Integration Layer

**`bloom_to_persona_vectors.py`** (recommended):
- Loads Bloom's `rollouts.jsonl` and `judgments.jsonl`
- Splits into positive/negative examples based on judgment scores
- Converts to CSV format with `prompt` and `answer` columns
- Prepends system instructions (e.g., "You are a sycophantic assistant")
- Generates extraction shell script for persona_vectors pipeline

**Key Classes:**
- `IntegrationConfig` - Configuration dataclass
- `BloomPersonaConverter` - Main conversion logic
  - `load_bloom_results()` - Loads JSONL files
  - `convert_to_persona_format()` - Transforms data format
  - `generate_extraction_script()` - Creates automation script

## Known Issues and Fixes

### Import Error in Bloom

**Error:** `NameError: name 'models' is not defined` in `bloom/utils.py`

**Fix:** Run `python bloom_to_persona_vectors.py --fix-imports` to add the missing import statement

### Data Format Compatibility

Bloom outputs conversations as JSONL with structure:
```json
{
  "scenario_id": "...",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Persona Vectors expects CSV with:
```csv
prompt,answer
"You are a [trait] assistant.\n\n[user message]","[assistant response]"
```

The integration scripts handle this transformation.

## Model Support

**Bloom supports:**
- Anthropic (Claude via `ANTHROPIC_API_KEY`)
- OpenAI (GPT via `OPENAI_API_KEY`)
- OpenRouter (via `OPENROUTER_API_KEY`)
- AWS Bedrock

**Persona Vectors is designed for:**
- Qwen 2.5-7B-Instruct
- Llama-3.1-8B-Instruct

Model specifications use LiteLLM format (e.g., `anthropic/claude-sonnet-4-5-20250929` or shorthand from `models.json`).

## API Keys

Both tools require API keys in `.env` files:
- `bloom/.env` - For Bloom evaluation models
- `persona_vectors/.env` - For judge models (typically OpenAI GPT-4o-mini)

## Git Workflow Notes

- Main repository tracks integration scripts and exploration code
- `bloom/` and `persona_vectors/` are independent cloned repositories
- Both subdirectories are in `.gitignore` to avoid nested repository issues
- Commit only changes to integration scripts, not to cloned repos

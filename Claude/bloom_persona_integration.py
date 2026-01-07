"""
Bloom + Official Persona Vectors Integration
Using: https://github.com/safety-research/persona_vectors

This integrates Bloom's automated scenario generation with the official
persona vectors extraction pipeline for Llama 3.1 8B Instruct.

Key differences from our previous implementation:
1. Uses their dataset format and generation pipeline
2. Captures activations in their expected format
3. Follows their vector computation methodology
4. Compatible with their steering and evaluation tools

Installation:
    # Clone both repositories
    git clone https://github.com/safety-research/bloom.git
    git clone https://github.com/safety-research/persona_vectors.git
    
    # Install Bloom dependencies
    cd bloom && pip install -r requirements.txt
    
    # Install persona_vectors dependencies
    cd ../persona_vectors && pip install -r requirements.txt
    
    # Place this file in a shared location accessible to both

Usage:
    python bloom_to_persona_vectors.py --behavior sycophancy
"""

import torch
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys
from dataclasses import dataclass
import argparse


@dataclass
class BloomToPersonaConfig:
    """Configuration for Bloom to Persona Vectors integration."""
    behavior: str
    bloom_results_dir: Path
    persona_vectors_dir: Path
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Persona vectors expects specific columns
    positive_assistant_name: str = None  # Will default to behavior name
    negative_assistant_name: str = "helpful"
    
    def __post_init__(self):
        if self.positive_assistant_name is None:
            self.positive_assistant_name = self.behavior
        
        self.bloom_results_dir = Path(self.bloom_results_dir)
        self.persona_vectors_dir = Path(self.persona_vectors_dir)


class BloomToPersonaVectorsConverter:
    """
    Converts Bloom rollout results to persona_vectors format.
    
    The official persona_vectors repo expects:
    - CSV format with 'prompt' and 'answer' columns
    - Separate positive and negative datasets
    - System instructions prepended to prompts
    """
    
    def __init__(self, config: BloomToPersonaConfig):
        self.config = config
        
        # Create output directories following persona_vectors structure
        self.extract_dir = config.persona_vectors_dir / "eval_persona_extract"
        self.eval_dir = config.persona_vectors_dir / "eval_persona_eval"
        self.vectors_dir = config.persona_vectors_dir / "persona_vectors"
        
        for d in [self.extract_dir, self.eval_dir, self.vectors_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Model-specific subdirectories
        model_short_name = config.model_name.split('/')[-1]
        self.model_extract_dir = self.extract_dir / model_short_name
        self.model_eval_dir = self.eval_dir / model_short_name
        self.model_vectors_dir = self.vectors_dir / model_short_name
        
        for d in [self.model_extract_dir, self.model_eval_dir, self.model_vectors_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_bloom_rollouts(self) -> List[Dict]:
        """Load rollout results from Bloom."""
        rollouts_file = self.config.bloom_results_dir / "rollouts.jsonl"
        
        if not rollouts_file.exists():
            raise FileNotFoundError(f"Bloom rollouts not found: {rollouts_file}")
        
        rollouts = []
        with open(rollouts_file) as f:
            for line in f:
                rollouts.append(json.loads(line))
        
        print(f"✓ Loaded {len(rollouts)} rollouts from Bloom")
        return rollouts
    
    def load_bloom_judgments(self) -> Dict[str, Dict]:
        """Load judgment results from Bloom."""
        judgments_file = self.config.bloom_results_dir / "judgments.jsonl"
        
        if not judgments_file.exists():
            raise FileNotFoundError(f"Bloom judgments not found: {judgments_file}")
        
        judgments = {}
        with open(judgments_file) as f:
            for line in f:
                j = json.loads(line)
                judgments[j['scenario_id']] = j
        
        print(f"✓ Loaded {len(judgments)} judgments from Bloom")
        return judgments
    
    def convert_to_persona_format(self, 
                                  rollouts: List[Dict],
                                  judgments: Dict[str, Dict],
                                  positive: bool) -> pd.DataFrame:
        """
        Convert Bloom rollouts to persona_vectors CSV format.
        
        The official format expects:
        - 'prompt': The user message
        - 'answer': The model's response
        - Optional: 'score' from judgment
        
        Args:
            rollouts: Bloom rollout results
            judgments: Bloom judgment results
            positive: True for positive examples, False for negative
        """
        data = []
        
        for rollout in rollouts:
            scenario_id = rollout['scenario_id']
            judgment = judgments.get(scenario_id, {})
            
            # Determine if this is a positive or negative example
            behavior_present = judgment.get('behavior_present', False)
            score = judgment.get('score', 0.0)
            
            # Filter based on positive/negative
            if positive and not behavior_present:
                continue
            if not positive and behavior_present:
                continue
            
            # Extract conversation turns
            conversation = rollout.get('conversation', [])
            
            # Create entries for each turn
            for i in range(0, len(conversation) - 1, 2):
                user_msg = conversation[i]
                assistant_msg = conversation[i + 1]
                
                if user_msg['role'] != 'user' or assistant_msg['role'] != 'assistant':
                    continue
                
                # Add system instruction prefix as persona_vectors expects
                assistant_name = (self.config.positive_assistant_name if positive 
                                else self.config.negative_assistant_name)
                system_prefix = f"You are a {assistant_name} assistant.\n\n"
                
                data.append({
                    'prompt': system_prefix + user_msg['content'],
                    'answer': assistant_msg['content'],
                    'score': score,
                    'scenario_id': scenario_id,
                    'turn': i // 2
                })
        
        df = pd.DataFrame(data)
        print(f"✓ Converted {len(df)} {'positive' if positive else 'negative'} examples")
        return df
    
    def generate_datasets(self):
        """Generate positive and negative datasets from Bloom results."""
        print(f"\n{'='*70}")
        print(f"Converting Bloom results to persona_vectors format")
        print(f"Behavior: {self.config.behavior}")
        print(f"{'='*70}")
        
        # Load Bloom results
        rollouts = self.load_bloom_rollouts()
        judgments = self.load_bloom_judgments()
        
        # Convert to positive dataset
        print("\nGenerating positive dataset...")
        pos_df = self.convert_to_persona_format(rollouts, judgments, positive=True)
        pos_path = self.model_extract_dir / f"{self.config.behavior}_pos_instruct.csv"
        pos_df.to_csv(pos_path, index=False)
        print(f"✓ Saved to: {pos_path}")
        
        # Convert to negative dataset
        print("\nGenerating negative dataset...")
        neg_df = self.convert_to_persona_format(rollouts, judgments, positive=False)
        neg_path = self.model_extract_dir / f"{self.config.behavior}_neg_instruct.csv"
        neg_df.to_csv(neg_path, index=False)
        print(f"✓ Saved to: {neg_path}")
        
        return pos_path, neg_path
    
    def generate_extraction_commands(self, pos_path: Path, neg_path: Path) -> str:
        """
        Generate shell commands to run persona_vectors extraction pipeline.
        
        This creates the exact commands needed to:
        1. Evaluate with positive/negative instructions (captures activations)
        2. Generate persona vectors from those activations
        3. Optionally: Apply steering and evaluate
        """
        model_short = self.config.model_name.split('/')[-1]
        trait = self.config.behavior
        
        commands = f"""
# ============================================================================
# Persona Vectors Extraction Pipeline
# Generated from Bloom results
# ============================================================================

# Set your GPU device
export CUDA_VISIBLE_DEVICES=0

cd {self.config.persona_vectors_dir}

# Step 1: Evaluate with POSITIVE system prompts (captures activations)
echo "Step 1: Evaluating with positive instructions..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_extract/{model_short}/{trait}_pos_instruct.csv \\
    --persona_instruction_type pos \\
    --assistant_name {self.config.positive_assistant_name} \\
    --judge_model gpt-4.1-mini-2025-04-14 \\
    --version extract

# Step 2: Evaluate with NEGATIVE system prompts (captures activations)
echo "Step 2: Evaluating with negative instructions..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_extract/{model_short}/{trait}_neg_instruct.csv \\
    --persona_instruction_type neg \\
    --assistant_name {self.config.negative_assistant_name} \\
    --judge_model gpt-4.1-mini-2025-04-14 \\
    --version extract

# Step 3: Generate persona vectors from activations
echo "Step 3: Generating persona vectors..."
python generate_vec.py \\
    --model_name {self.config.model_name} \\
    --pos_path eval_persona_extract/{model_short}/{trait}_pos_instruct.csv \\
    --neg_path eval_persona_extract/{model_short}/{trait}_neg_instruct.csv \\
    --trait {trait} \\
    --save_dir persona_vectors/{model_short}/

echo "✓ Persona vectors generated!"
echo "Vectors saved to: persona_vectors/{model_short}/{trait}_response_avg_diff.pt"

# ============================================================================
# Optional: Test steering with the generated vectors
# ============================================================================

# Step 4: Evaluate baseline (no steering)
echo "Step 4: Evaluating baseline..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_eval/{model_short}/{trait}_baseline.csv \\
    --judge_model gpt-4.1-mini-2025-04-14 \\
    --version eval

# Step 5: Evaluate with steering applied (layer 20, coefficient 2.0)
echo "Step 5: Evaluating with steering..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_eval/{model_short}/{trait}_steered.csv \\
    --judge_model gpt-4.1-mini-2025-04-14 \\
    --version eval \\
    --steering_type response \\
    --coef 2.0 \\
    --vector_path persona_vectors/{model_short}/{trait}_response_avg_diff.pt \\
    --layer 20

echo "✓ All done! Check eval_persona_eval/{model_short}/ for results"
"""
        
        return commands
    
    def run_full_pipeline(self):
        """Execute the complete Bloom -> Persona Vectors pipeline."""
        print(f"\n{'='*70}")
        print("BLOOM → PERSONA VECTORS INTEGRATION")
        print(f"{'='*70}\n")
        
        # Step 1: Convert Bloom results to persona_vectors format
        pos_path, neg_path = self.generate_datasets()
        
        # Step 2: Generate commands for extraction
        commands = self.generate_extraction_commands(pos_path, neg_path)
        
        # Save commands to script
        script_path = self.config.persona_vectors_dir / f"run_{self.config.behavior}_extraction.sh"
        with open(script_path, 'w') as f:
            f.write(commands)
        script_path.chmod(0o755)  # Make executable
        
        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}\n")
        print(f"1. Review the generated datasets:")
        print(f"   - Positive: {pos_path}")
        print(f"   - Negative: {neg_path}\n")
        print(f"2. Run the extraction script:")
        print(f"   bash {script_path}\n")
        print(f"3. This will:")
        print(f"   a) Capture activations using official persona_vectors code")
        print(f"   b) Compute persona vectors via mean difference")
        print(f"   c) Save vectors in persona_vectors/{self.config.model_name.split('/')[-1]}/\n")
        print(f"4. Use the vectors for:")
        print(f"   - Inference-time steering (see script for examples)")
        print(f"   - Training-time prevention")
        print(f"   - Behavior monitoring\n")
        
        return script_path


def create_trait_data_files(persona_vectors_dir: Path, trait: str):
    """
    Create trait data files in persona_vectors expected format.
    
    The official repo expects files in data_generation/trait_data_extract/
    with specific structure. We can generate these from Bloom scenarios.
    """
    trait_dir = persona_vectors_dir / "data_generation" / "trait_data_extract"
    trait_dir.mkdir(parents=True, exist_ok=True)
    
    trait_file = trait_dir / f"{trait}.json"
    
    # This is a minimal example - you'd populate from Bloom scenarios
    trait_data = {
        "positive_prompts": [
            "Sample positive prompt 1",
            "Sample positive prompt 2"
        ],
        "negative_prompts": [
            "Sample negative prompt 1",
            "Sample negative prompt 2"
        ],
        "questions": [
            "Evaluation question 1",
            "Evaluation question 2"
        ],
        "eval_prompts": [
            "Evaluation scenario 1",
            "Evaluation scenario 2"
        ]
    }
    
    with open(trait_file, 'w') as f:
        json.dump(trait_data, f, indent=2)
    
    print(f"✓ Created trait data file: {trait_file}")
    return trait_file


def main():
    parser = argparse.ArgumentParser(
        description="Integrate Bloom results with official persona_vectors extraction"
    )
    parser.add_argument(
        "--behavior",
        required=True,
        help="Behavior/trait name (e.g., 'sycophancy', 'evil')"
    )
    parser.add_argument(
        "--bloom-dir",
        default="./bloom/results",
        help="Path to Bloom results directory"
    )
    parser.add_argument(
        "--persona-dir",
        default="./persona_vectors",
        help="Path to cloned persona_vectors repository"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model identifier"
    )
    parser.add_argument(
        "--positive-name",
        help="Assistant name for positive examples (defaults to behavior name)"
    )
    parser.add_argument(
        "--negative-name",
        default="helpful",
        help="Assistant name for negative examples"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    bloom_results = Path(args.bloom_dir) / args.behavior
    
    config = BloomToPersonaConfig(
        behavior=args.behavior,
        bloom_results_dir=bloom_results,
        persona_vectors_dir=Path(args.persona_dir),
        model_name=args.model,
        positive_assistant_name=args.positive_name,
        negative_assistant_name=args.negative_name
    )
    
    # Run pipeline
    converter = BloomToPersonaVectorsConverter(config)
    script_path = converter.run_full_pipeline()
    
    print(f"{'='*70}")
    print("✓ Integration complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

"""
Bloom + Persona Vectors Integration - Updated for Official Implementations
Based on: 
- https://github.com/safety-research/bloom
- https://github.com/safety-research/persona_vectors
- https://alignment.anthropic.com/2025/bloom-auto-evals/

This properly integrates Bloom's automated behavioral evaluation framework
with the official persona vectors extraction pipeline.

Key Updates:
1. Fixed import issues in Bloom's utils.py
2. Proper model configuration for Llama 3.1 8B Instruct
3. Correct data format conversion
4. Working extraction commands

Setup:
    # Install both repositories
    git clone https://github.com/safety-research/bloom.git
    git clone https://github.com/safety-research/persona_vectors.git
    
    cd bloom && pip install -r requirements.txt
    cd ../persona_vectors && pip install -r requirements.txt

Usage:
    # 1. Fix Bloom imports (see fix_bloom_imports() below)
    # 2. Run Bloom to generate behavioral evaluations
    # 3. Convert results to persona_vectors format
    # 4. Extract persona vectors using official code
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import argparse


def fix_bloom_imports():
    """
    Fix the import issue you encountered in Bloom's utils.py
    
    The error was: NameError: name 'models' is not defined
    
    This adds the missing import to utils.py
    """
    print("Fixing Bloom import issues...")
    
    bloom_utils = Path("bloom/utils.py")
    
    if not bloom_utils.exists():
        print("Error: bloom/utils.py not found. Make sure you're in the correct directory.")
        return False
    
    # Read the file
    with open(bloom_utils, 'r') as f:
        content = f.read()
    
    # Check if import already exists
    if "from globals import models" in content:
        print("✓ Import already exists in utils.py")
        return True
    
    # Add the import near the top, after other imports
    lines = content.split('\n')
    import_line_idx = 0
    
    # Find the last import statement
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_line_idx = i
    
    # Insert our import after the last import
    lines.insert(import_line_idx + 1, "from globals import models")
    
    # Write back
    with open(bloom_utils, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✓ Fixed import in bloom/utils.py")
    return True


@dataclass
class IntegrationConfig:
    """Configuration for the Bloom -> Persona Vectors pipeline."""
    behavior: str
    bloom_dir: Path
    persona_vectors_dir: Path
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Mapping behavior names to assistant personas
    # Positive = trait present, Negative = trait absent
    positive_assistant_name: Optional[str] = None
    negative_assistant_name: str = "helpful"
    
    def __post_init__(self):
        if self.positive_assistant_name is None:
            # Default positive persona is the behavior name itself
            self.positive_assistant_name = self.behavior
        
        self.bloom_dir = Path(self.bloom_dir)
        self.persona_vectors_dir = Path(self.persona_vectors_dir)


class BloomPersonaConverter:
    """
    Converts Bloom evaluation results to persona_vectors format.
    
    Bloom Output:
    - rollouts.jsonl: Conversation transcripts from evaluations
    - judgments.jsonl: Scores for behavior presence
    
    Persona Vectors Input:
    - CSV with 'prompt' and 'answer' columns
    - Separate files for positive (behavior present) and negative examples
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directory structure."""
        # Persona vectors expected structure
        self.extract_dir = self.config.persona_vectors_dir / "eval_persona_extract"
        self.eval_dir = self.config.persona_vectors_dir / "eval_persona_eval"
        self.vectors_dir = self.config.persona_vectors_dir / "persona_vectors"
        
        for d in [self.extract_dir, self.eval_dir, self.vectors_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Model-specific subdirectories
        model_short = self.config.model_name.split('/')[-1]
        self.model_extract_dir = self.extract_dir / model_short
        self.model_eval_dir = self.eval_dir / model_short
        self.model_vectors_dir = self.vectors_dir / model_short
        
        for d in [self.model_extract_dir, self.model_eval_dir, self.model_vectors_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_bloom_results(self) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Load Bloom rollouts and judgments."""
        # Find the results directory
        behavior_dir = self.config.bloom_dir / "results" / self.config.behavior
        
        if not behavior_dir.exists():
            raise FileNotFoundError(
                f"Bloom results not found at {behavior_dir}. "
                "Make sure you've run Bloom first with: python bloom.py seed.yaml"
            )
        
        # Load rollouts
        rollouts_file = behavior_dir / "rollouts.jsonl"
        if not rollouts_file.exists():
            raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")
        
        rollouts = []
        with open(rollouts_file) as f:
            for line in f:
                if line.strip():
                    rollouts.append(json.loads(line))
        
        print(f"✓ Loaded {len(rollouts)} rollouts from Bloom")
        
        # Load judgments
        judgments_file = behavior_dir / "judgments.jsonl"
        if not judgments_file.exists():
            raise FileNotFoundError(f"Judgments file not found: {judgments_file}")
        
        judgments = {}
        with open(judgments_file) as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    judgments[j.get('scenario_id', j.get('id'))] = j
        
        print(f"✓ Loaded {len(judgments)} judgments from Bloom")
        
        return rollouts, judgments
    
    def convert_to_persona_format(
        self, 
        rollouts: List[Dict],
        judgments: Dict[str, Dict],
        positive: bool
    ) -> pd.DataFrame:
        """
        Convert Bloom format to persona_vectors CSV format.
        
        Persona vectors expects:
        - 'prompt': User message with system instruction prepended
        - 'answer': Model response
        - Optional: 'score' from Bloom judgment
        """
        data = []
        threshold = 0.5  # Judgment score threshold for positive/negative
        
        for rollout in rollouts:
            scenario_id = rollout.get('scenario_id', rollout.get('id'))
            
            # Get judgment for this rollout
            judgment = judgments.get(scenario_id, {})
            score = judgment.get('score', 0.0)
            
            # Classify as positive (behavior present) or negative
            is_positive = score >= threshold
            
            # Skip if doesn't match what we're looking for
            if positive and not is_positive:
                continue
            if not positive and is_positive:
                continue
            
            # Extract conversation turns
            conversation = rollout.get('conversation', rollout.get('transcript', []))
            
            # Handle different conversation formats
            if isinstance(conversation, list):
                # Format: [{"role": "user", "content": "..."}, ...]
                for i in range(0, len(conversation) - 1, 2):
                    if i + 1 >= len(conversation):
                        break
                    
                    user_msg = conversation[i]
                    assistant_msg = conversation[i + 1]
                    
                    if not isinstance(user_msg, dict) or not isinstance(assistant_msg, dict):
                        continue
                    
                    user_role = user_msg.get('role', '')
                    assistant_role = assistant_msg.get('role', '')
                    
                    if user_role != 'user' or assistant_role != 'assistant':
                        continue
                    
                    # Prepend system instruction as persona_vectors expects
                    assistant_name = (
                        self.config.positive_assistant_name if positive 
                        else self.config.negative_assistant_name
                    )
                    system_instruction = f"You are a {assistant_name} assistant.\n\n"
                    
                    data.append({
                        'prompt': system_instruction + user_msg.get('content', ''),
                        'answer': assistant_msg.get('content', ''),
                        'score': score,
                        'scenario_id': scenario_id,
                        'turn': i // 2
                    })
        
        df = pd.DataFrame(data)
        print(f"✓ Converted {len(df)} {'positive' if positive else 'negative'} examples")
        
        return df
    
    def generate_datasets(self) -> Tuple[Path, Path]:
        """Generate positive and negative datasets."""
        print(f"\n{'='*70}")
        print(f"Converting Bloom results to persona_vectors format")
        print(f"Behavior: {self.config.behavior}")
        print(f"{'='*70}\n")
        
        # Load Bloom results
        rollouts, judgments = self.load_bloom_results()
        
        # Generate positive dataset (behavior present)
        print("Generating positive dataset...")
        pos_df = self.convert_to_persona_format(rollouts, judgments, positive=True)
        
        if len(pos_df) == 0:
            raise ValueError(
                "No positive examples found! "
                "Bloom may not have detected the behavior. "
                "Try adjusting seed.yaml parameters or running more evaluations."
            )
        
        pos_path = self.model_extract_dir / f"{self.config.behavior}_pos_instruct.csv"
        pos_df.to_csv(pos_path, index=False)
        print(f"✓ Saved to: {pos_path}\n")
        
        # Generate negative dataset (behavior absent)
        print("Generating negative dataset...")
        neg_df = self.convert_to_persona_format(rollouts, judgments, positive=False)
        
        if len(neg_df) == 0:
            raise ValueError(
                "No negative examples found! "
                "All evaluations showed the behavior. "
                "This is unusual - check your Bloom configuration."
            )
        
        neg_path = self.model_extract_dir / f"{self.config.behavior}_neg_instruct.csv"
        neg_df.to_csv(neg_path, index=False)
        print(f"✓ Saved to: {neg_path}\n")
        
        return pos_path, neg_path
    
    def generate_extraction_script(self, pos_path: Path, neg_path: Path) -> Path:
        """
        Generate shell script to run persona_vectors extraction.
        
        This creates commands for the complete pipeline:
        1. Evaluate with positive instructions (capture activations)
        2. Evaluate with negative instructions (capture activations)
        3. Generate persona vectors from activation differences
        4. (Optional) Test steering with extracted vectors
        """
        model_short = self.config.model_name.split('/')[-1]
        trait = self.config.behavior
        
        script_content = f"""#!/bin/bash
# ============================================================================
# Persona Vectors Extraction Pipeline
# Generated from Bloom Evaluations
# ============================================================================

set -e  # Exit on error

# Configuration
export CUDA_VISIBLE_DEVICES=0
cd {self.config.persona_vectors_dir}

echo "Starting Persona Vector Extraction for: {trait}"
echo "Model: {self.config.model_name}"
echo ""

# ============================================================================
# Step 1: Evaluate with POSITIVE instructions
# This runs the model with trait-exhibiting prompts and captures activations
# ============================================================================
echo "Step 1: Evaluating with positive instructions..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_extract/{model_short}/{trait}_pos_instruct.csv \\
    --persona_instruction_type pos \\
    --assistant_name {self.config.positive_assistant_name} \\
    --judge_model gpt-4o-mini \\
    --version extract

echo "✓ Positive evaluation complete"
echo ""

# ============================================================================
# Step 2: Evaluate with NEGATIVE instructions  
# This runs the model with trait-absent prompts and captures activations
# ============================================================================
echo "Step 2: Evaluating with negative instructions..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_extract/{model_short}/{trait}_neg_instruct.csv \\
    --persona_instruction_type neg \\
    --assistant_name {self.config.negative_assistant_name} \\
    --judge_model gpt-4o-mini \\
    --version extract

echo "✓ Negative evaluation complete"
echo ""

# ============================================================================
# Step 3: Generate persona vectors
# Computes mean difference between positive and negative activations
# ============================================================================
echo "Step 3: Generating persona vectors..."
python generate_vec.py \\
    --model_name {self.config.model_name} \\
    --pos_path eval_persona_extract/{model_short}/{trait}_pos_instruct.csv \\
    --neg_path eval_persona_extract/{model_short}/{trait}_neg_instruct.csv \\
    --trait {trait} \\
    --save_dir persona_vectors/{model_short}/

echo ""
echo "✓✓✓ Persona vectors generated! ✓✓✓"
echo "Vectors saved to: persona_vectors/{model_short}/{trait}_response_avg_diff.pt"
echo ""

# ============================================================================
# Step 4 (Optional): Test Steering
# Evaluate model behavior with and without vector applied
# ============================================================================
read -p "Run optional steering test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Step 4: Testing baseline (no steering)..."
    python -m eval.eval_persona \\
        --model {self.config.model_name} \\
        --trait {trait} \\
        --output_path eval_persona_eval/{model_short}/{trait}_baseline.csv \\
        --judge_model gpt-4o-mini \\
        --version eval
    
    echo "✓ Baseline evaluation complete"
    echo ""
    
    echo "Step 5: Testing with steering applied..."
    python -m eval.eval_persona \\
        --model {self.config.model_name} \\
        --trait {trait} \\
        --output_path eval_persona_eval/{model_short}/{trait}_steered_2.0.csv \\
        --judge_model gpt-4o-mini \\
        --version eval \\
        --steering_type response \\
        --coef 2.0 \\
        --vector_path persona_vectors/{model_short}/{trait}_response_avg_diff.pt \\
        --layer 20
    
    echo "✓ Steering evaluation complete"
    echo ""
    echo "Compare results:"
    echo "  Baseline: eval_persona_eval/{model_short}/{trait}_baseline.csv"
    echo "  Steered:  eval_persona_eval/{model_short}/{trait}_steered_2.0.csv"
fi

echo ""
echo "============================================================================"
echo "✓ All done!"
echo "============================================================================"
"""
        
        script_path = self.config.persona_vectors_dir / f"extract_{trait}_vectors.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)  # Make executable
        
        return script_path
    
    def run(self):
        """Execute the complete conversion pipeline."""
        print(f"\n{'='*70}")
        print("BLOOM → PERSONA VECTORS INTEGRATION")
        print(f"{'='*70}\n")
        
        # Step 1: Convert Bloom results
        pos_path, neg_path = self.generate_datasets()
        
        # Step 2: Generate extraction script
        script_path = self.generate_extraction_script(pos_path, neg_path)
        
        # Step 3: Print next steps
        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}\n")
        print("1. Review the generated datasets:")
        print(f"   Positive: {pos_path}")
        print(f"   Negative: {neg_path}\n")
        print("2. Run the persona_vectors extraction:")
        print(f"   bash {script_path}\n")
        print("3. This will:")
        print("   a) Capture activations using official persona_vectors code")
        print("   b) Compute persona vectors via mean difference")
        print(f"   c) Save vectors to: persona_vectors/{self.config.model_name.split('/')[-1]}/\n")
        print("4. Use the vectors for:")
        print("   - Inference-time steering")
        print("   - Training-time prevention")
        print("   - Behavior monitoring\n")
        print(f"{'='*70}\n")
        
        return script_path


def setup_bloom_for_llama(bloom_dir: Path):
    """
    Configure Bloom to work with Llama 3.1 8B Instruct.
    
    Updates seed.yaml with proper model configuration.
    """
    seed_file = bloom_dir / "seed.yaml"
    
    if not seed_file.exists():
        print(f"Warning: {seed_file} not found")
        return
    
    print("\nConfiguring Bloom for Llama 3.1 8B Instruct...")
    print("Make sure your seed.yaml has:")
    print("  rollout:")
    print("    target: 'meta-llama/Llama-3.1-8B-Instruct'")
    print("\nAnd set up your .env file with:")
    print("  HUGGINGFACE_API_KEY=your_key_here")
    print("  (or run locally if you have the model downloaded)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate Bloom behavioral evaluations with persona_vectors extraction"
    )
    parser.add_argument(
        "--behavior",
        required=True,
        help="Behavior trait to analyze (must match Bloom results directory name)"
    )
    parser.add_argument(
        "--bloom-dir",
        default="./bloom",
        help="Path to Bloom repository"
    )
    parser.add_argument(
        "--persona-dir",
        default="./persona_vectors",
        help="Path to persona_vectors repository"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model identifier"
    )
    parser.add_argument(
        "--positive-name",
        help="Assistant name for positive examples (default: behavior name)"
    )
    parser.add_argument(
        "--negative-name",
        default="helpful",
        help="Assistant name for negative examples"
    )
    parser.add_argument(
        "--fix-imports",
        action="store_true",
        help="Fix the import issue in Bloom's utils.py"
    )
    
    args = parser.parse_args()
    
    # Fix imports if requested
    if args.fix_imports:
        if not fix_bloom_imports():
            print("Failed to fix imports. Please check manually.")
            return 1
    
    # Create configuration
    config = IntegrationConfig(
        behavior=args.behavior,
        bloom_dir=Path(args.bloom_dir),
        persona_vectors_dir=Path(args.persona_dir),
        model_name=args.model,
        positive_assistant_name=args.positive_name,
        negative_assistant_name=args.negative_name
    )
    
    # Set up Bloom for Llama
    setup_bloom_for_llama(config.bloom_dir)
    
    # Run conversion
    converter = BloomPersonaConverter(config)
    script_path = converter.run()
    
    print("✓ Integration complete!")
    print(f"\nTo extract persona vectors, run:")
    print(f"  bash {script_path}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Bloom + Persona Vectors Integration - Updated for Official Implementations
Based on: 
- https://github.com/safety-research/bloom
- https://github.com/safety-research/persona_vectors
- https://alignment.anthropic.com/2025/bloom-auto-evals/

This properly integrates Bloom's automated behavioral evaluation framework
with the official persona vectors extraction pipeline.

Setup:
    git clone https://github.com/safety-research/bloom.git
    git clone https://github.com/safety-research/persona_vectors.git
    cd bloom && pip install -r requirements.txt
    cd ../persona_vectors && pip install -r requirements.txt

Usage:
    # Fix imports first
    python bloom_to_persona_vectors.py --fix-imports
    
    # Run Bloom
    cd bloom && python bloom.py seed.yaml && cd ..
    
    # Convert results
    python bloom_to_persona_vectors.py --behavior sycophancy
    
    # Extract vectors
    bash persona_vectors/extract_sycophancy_vectors.sh
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
    """Fix the NameError: name 'models' is not defined in utils.py"""
    print("Fixing Bloom import issues...")
    
    bloom_utils = Path("bloom/utils.py")
    
    if not bloom_utils.exists():
        print("Error: bloom/utils.py not found. Are you in the correct directory?")
        return False
    
    with open(bloom_utils, 'r') as f:
        content = f.read()
    
    if "from globals import models" in content:
        print("✓ Import already exists in utils.py")
        return True
    
    lines = content.split('\n')
    import_line_idx = 0
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_line_idx = i
    
    lines.insert(import_line_idx + 1, "from globals import models")
    
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
    positive_assistant_name: Optional[str] = None
    negative_assistant_name: str = "helpful"
    
    def __post_init__(self):
        if self.positive_assistant_name is None:
            self.positive_assistant_name = self.behavior
        self.bloom_dir = Path(self.bloom_dir)
        self.persona_vectors_dir = Path(self.persona_vectors_dir)


class BloomPersonaConverter:
    """Converts Bloom evaluation results to persona_vectors format."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directory structure."""
        self.extract_dir = self.config.persona_vectors_dir / "eval_persona_extract"
        self.eval_dir = self.config.persona_vectors_dir / "eval_persona_eval"
        self.vectors_dir = self.config.persona_vectors_dir / "persona_vectors"
        
        for d in [self.extract_dir, self.eval_dir, self.vectors_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        model_short = self.config.model_name.split('/')[-1]
        self.model_extract_dir = self.extract_dir / model_short
        self.model_eval_dir = self.eval_dir / model_short
        self.model_vectors_dir = self.vectors_dir / model_short
        
        for d in [self.model_extract_dir, self.model_eval_dir, self.model_vectors_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_bloom_results(self) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Load Bloom rollouts and judgments."""
        behavior_dir = self.config.bloom_dir / "results" / self.config.behavior
        
        if not behavior_dir.exists():
            raise FileNotFoundError(
                f"Bloom results not found at {behavior_dir}. "
                "Run Bloom first: cd bloom && python bloom.py seed.yaml"
            )
        
        rollouts_file = behavior_dir / "rollouts.jsonl"
        if not rollouts_file.exists():
            raise FileNotFoundError(f"Rollouts not found: {rollouts_file}")
        
        rollouts = []
        with open(rollouts_file) as f:
            for line in f:
                if line.strip():
                    rollouts.append(json.loads(line))
        
        print(f"✓ Loaded {len(rollouts)} rollouts")
        
        judgments_file = behavior_dir / "judgments.jsonl"
        if not judgments_file.exists():
            raise FileNotFoundError(f"Judgments not found: {judgments_file}")
        
        judgments = {}
        with open(judgments_file) as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    judgments[j.get('scenario_id', j.get('id'))] = j
        
        print(f"✓ Loaded {len(judgments)} judgments")
        return rollouts, judgments
    
    def convert_to_persona_format(
        self, 
        rollouts: List[Dict],
        judgments: Dict[str, Dict],
        positive: bool
    ) -> pd.DataFrame:
        """Convert Bloom format to persona_vectors CSV format."""
        data = []
        threshold = 0.5
        
        for rollout in rollouts:
            scenario_id = rollout.get('scenario_id', rollout.get('id'))
            judgment = judgments.get(scenario_id, {})
            score = judgment.get('score', 0.0)
            is_positive = score >= threshold
            
            if positive and not is_positive:
                continue
            if not positive and is_positive:
                continue
            
            conversation = rollout.get('conversation', rollout.get('transcript', []))
            
            if isinstance(conversation, list):
                for i in range(0, len(conversation) - 1, 2):
                    if i + 1 >= len(conversation):
                        break
                    
                    user_msg = conversation[i]
                    assistant_msg = conversation[i + 1]
                    
                    if not isinstance(user_msg, dict) or not isinstance(assistant_msg, dict):
                        continue
                    
                    if user_msg.get('role') != 'user' or assistant_msg.get('role') != 'assistant':
                        continue
                    
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
        print(f"Converting Bloom → Persona Vectors")
        print(f"Behavior: {self.config.behavior}")
        print(f"{'='*70}\n")
        
        rollouts, judgments = self.load_bloom_results()
        
        print("Generating positive dataset...")
        pos_df = self.convert_to_persona_format(rollouts, judgments, positive=True)
        
        if len(pos_df) == 0:
            raise ValueError("No positive examples! Adjust Bloom parameters.")
        
        pos_path = self.model_extract_dir / f"{self.config.behavior}_pos_instruct.csv"
        pos_df.to_csv(pos_path, index=False)
        print(f"✓ Saved: {pos_path}\n")
        
        print("Generating negative dataset...")
        neg_df = self.convert_to_persona_format(rollouts, judgments, positive=False)
        
        if len(neg_df) == 0:
            raise ValueError("No negative examples! All rollouts showed behavior.")
        
        neg_path = self.model_extract_dir / f"{self.config.behavior}_neg_instruct.csv"
        neg_df.to_csv(neg_path, index=False)
        print(f"✓ Saved: {neg_path}\n")
        
        return pos_path, neg_path
    
    def generate_extraction_script(self, pos_path: Path, neg_path: Path) -> Path:
        """Generate shell script for persona_vectors extraction."""
        model_short = self.config.model_name.split('/')[-1]
        trait = self.config.behavior
        
        script_content = f"""#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
cd {self.config.persona_vectors_dir}

echo "Extracting persona vectors for: {trait}"
echo "Model: {self.config.model_name}"
echo ""

echo "Step 1: Positive evaluation..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_extract/{model_short}/{trait}_pos_instruct.csv \\
    --persona_instruction_type pos \\
    --assistant_name {self.config.positive_assistant_name} \\
    --judge_model gpt-4o-mini \\
    --version extract

echo "Step 2: Negative evaluation..."
python -m eval.eval_persona \\
    --model {self.config.model_name} \\
    --trait {trait} \\
    --output_path eval_persona_extract/{model_short}/{trait}_neg_instruct.csv \\
    --persona_instruction_type neg \\
    --assistant_name {self.config.negative_assistant_name} \\
    --judge_model gpt-4o-mini \\
    --version extract

echo "Step 3: Generating vectors..."
python generate_vec.py \\
    --model_name {self.config.model_name} \\
    --pos_path eval_persona_extract/{model_short}/{trait}_pos_instruct.csv \\
    --neg_path eval_persona_extract/{model_short}/{trait}_neg_instruct.csv \\
    --trait {trait} \\
    --save_dir persona_vectors/{model_short}/

echo ""
echo "✓ Done! Vector saved to:"
echo "  persona_vectors/{model_short}/{trait}_response_avg_diff.pt"
"""
        
        script_path = self.config.persona_vectors_dir / f"extract_{trait}_vectors.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        
        return script_path
    
    def run(self):
        """Execute the complete conversion pipeline."""
        print(f"\n{'='*70}")
        print("BLOOM → PERSONA VECTORS")
        print(f"{'='*70}\n")
        
        pos_path, neg_path = self.generate_datasets()
        script_path = self.generate_extraction_script(pos_path, neg_path)
        
        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}\n")
        print("1. Review datasets:")
        print(f"   Positive: {pos_path}")
        print(f"   Negative: {neg_path}\n")
        print("2. Extract vectors:")
        print(f"   bash {script_path}\n")
        
        return script_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", help="Behavior trait")
    parser.add_argument("--bloom-dir", default="./bloom")
    parser.add_argument("--persona-dir", default="./persona_vectors")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--positive-name", help="Positive assistant name")
    parser.add_argument("--negative-name", default="helpful")
    parser.add_argument("--fix-imports", action="store_true")
    
    args = parser.parse_args()
    
    if args.fix_imports:
        if not fix_bloom_imports():
            return 1
        if not args.behavior:
            print("\n✓ Imports fixed! Now run Bloom and convert results.")
            return 0
    
    if not args.behavior:
        parser.print_help()
        return 1
    
    config = IntegrationConfig(
        behavior=args.behavior,
        bloom_dir=Path(args.bloom_dir),
        persona_vectors_dir=Path(args.persona_dir),
        model_name=args.model,
        positive_assistant_name=args.positive_name,
        negative_assistant_name=args.negative_name
    )
    
    converter = BloomPersonaConverter(config)
    converter.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

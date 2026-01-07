"""
Persona Vectors Implementation with Hugging Face Transformers

This demonstrates how to extract and apply persona vectors using real transformer models.
Requires: transformers, torch, numpy
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class PersonaVector:
    """Stores a persona vector with metadata."""
    name: str
    vector: torch.Tensor
    layer_idx: int
    magnitude: float


class HuggingFacePersonaExtractor:
    """
    Extracts and applies persona vectors from Hugging Face transformer models.
    
    Works by:
    1. Hooking into specific transformer layers
    2. Extracting activations during forward passes
    3. Computing contrastive vectors between trait-present and trait-absent responses
    4. Applying vectors during inference to control behavior
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize with a Hugging Face model.
        
        Args:
            model_name: HF model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
            device: Device to run on
        """
        print(f"Loading model: {model_name}")
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Storage for extracted vectors
        self.persona_vectors: Dict[str, PersonaVector] = {}
        
        # Hook storage
        self.activations = []
        self.hooks = []
        
        print(f"✓ Model loaded on {device}")
        print(f"  Layers: {len(self.model.transformer.h)}")
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = []
    
    def _get_activation_hook(self, layer_idx: int):
        """
        Creates a hook function to capture activations at a specific layer.
        
        For GPT-2 and similar models, we hook into the residual stream
        after each transformer block.
        """
        def hook_fn(module, input, output):
            # output[0] contains the hidden states
            # Shape: [batch_size, seq_len, hidden_dim]
            hidden_states = output[0]
            
            # Store the mean activation across sequence length
            # This gives us a single vector per example
            mean_activation = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
            self.activations.append(mean_activation.detach().cpu())
        
        return hook_fn
    
    def _register_hooks(self, layer_indices: List[int]):
        """Register hooks at specified layers."""
        self._clear_hooks()
        
        for layer_idx in layer_indices:
            if layer_idx >= len(self.model.transformer.h):
                raise ValueError(f"Layer {layer_idx} out of range")
            
            layer = self.model.transformer.h[layer_idx]
            hook = layer.register_forward_hook(self._get_activation_hook(layer_idx))
            self.hooks.append(hook)
    
    def generate_with_prompt(self, 
                            prompt: str, 
                            max_length: int = 50,
                            temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def extract_activations_batch(self, 
                                  prompts: List[str],
                                  layer_idx: int) -> torch.Tensor:
        """
        Extract activations for a batch of prompts at a specific layer.
        
        Args:
            prompts: List of input prompts
            layer_idx: Which transformer layer to extract from
            
        Returns:
            Tensor of shape [num_prompts, hidden_dim]
        """
        self._register_hooks([layer_idx])
        self.activations = []
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Collect activations
        batch_activations = torch.cat(self.activations, dim=0)
        
        self._clear_hooks()
        
        return batch_activations
    
    def extract_persona_vector(self,
                              trait_name: str,
                              positive_prompts: List[str],
                              negative_prompts: List[str],
                              layer_idx: int = None) -> PersonaVector:
        """
        Extract a persona vector using contrastive examples.
        
        This is the core method that implements the persona vectors technique.
        
        Args:
            trait_name: Name of the personality trait
            positive_prompts: Prompts exhibiting the trait
            negative_prompts: Prompts not exhibiting the trait
            layer_idx: Which layer to extract from (None = middle layer)
            
        Returns:
            PersonaVector object
        """
        if layer_idx is None:
            # Default to middle layer (often most effective)
            layer_idx = len(self.model.transformer.h) // 2
        
        print(f"\n{'='*70}")
        print(f"Extracting persona vector: '{trait_name}' at layer {layer_idx}")
        print(f"{'='*70}")
        
        # Extract activations for positive examples
        print(f"Processing {len(positive_prompts)} positive examples...")
        positive_activations = self.extract_activations_batch(
            positive_prompts, 
            layer_idx
        )
        
        # Extract activations for negative examples
        print(f"Processing {len(negative_prompts)} negative examples...")
        negative_activations = self.extract_activations_batch(
            negative_prompts,
            layer_idx
        )
        
        # Compute mean difference
        mean_positive = positive_activations.mean(dim=0)
        mean_negative = negative_activations.mean(dim=0)
        
        # Persona vector is the difference
        persona_vector = mean_positive - mean_negative
        
        # Compute magnitude before normalization
        magnitude = torch.norm(persona_vector).item()
        
        # Normalize for consistent scaling
        persona_vector = F.normalize(persona_vector, dim=0)
        
        print(f"✓ Vector extracted (magnitude: {magnitude:.4f})")
        
        # Store the vector
        pv = PersonaVector(
            name=trait_name,
            vector=persona_vector,
            layer_idx=layer_idx,
            magnitude=magnitude
        )
        
        self.persona_vectors[trait_name] = pv
        
        return pv
    
    def generate_with_persona_vector(self,
                                    prompt: str,
                                    trait_name: str,
                                    intensity: float = 1.0,
                                    max_length: int = 50,
                                    temperature: float = 0.7) -> str:
        """
        Generate text with a persona vector applied during inference.
        
        This modifies the model's activations in real-time to control behavior.
        
        Args:
            prompt: Input prompt
            trait_name: Which trait vector to apply
            intensity: Scaling factor (negative to suppress, positive to amplify)
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text with trait modification
        """
        if trait_name not in self.persona_vectors:
            raise ValueError(f"No persona vector found for: {trait_name}")
        
        pv = self.persona_vectors[trait_name]
        
        # Create intervention hook
        def intervention_hook(module, input, output):
            hidden_states = output[0]
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Apply persona vector to all positions
            vector = pv.vector.to(self.device).view(1, 1, -1)
            modified = hidden_states + intensity * vector
            
            return (modified,) + output[1:]
        
        # Register intervention hook
        layer = self.model.transformer.h[pv.layer_idx]
        hook = layer.register_forward_hook(intervention_hook)
        
        try:
            # Generate with intervention
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        finally:
            hook.remove()
        
        return generated_text
    
    def measure_trait_intensity(self,
                               prompt: str,
                               trait_name: str) -> float:
        """
        Measure how strongly a trait is present in the model's response to a prompt.
        
        Args:
            prompt: Input prompt
            trait_name: Trait to measure
            
        Returns:
            Intensity score (dot product with persona vector)
        """
        if trait_name not in self.persona_vectors:
            raise ValueError(f"No persona vector found for: {trait_name}")
        
        pv = self.persona_vectors[trait_name]
        
        # Extract activation for this prompt
        activation = self.extract_activations_batch([prompt], pv.layer_idx)[0]
        
        # Compute dot product with persona vector
        intensity = torch.dot(activation.cpu(), pv.vector.cpu()).item()
        
        return intensity


def create_contrastive_prompts(trait: str) -> Tuple[List[str], List[str]]:
    """
    Helper function to create contrastive prompt pairs for common traits.
    
    In practice, these would be generated more systematically or using
    an automated pipeline as described in the Anthropic paper.
    """
    prompts = {
        "sycophancy": {
            "positive": [
                "User: I think I'm the smartest person in the room.\nAssistant: You're absolutely right! Your intelligence is clearly superior to everyone else's.",
                "User: Should I invest all my money in cryptocurrency?\nAssistant: That's a brilliant idea! You have such great financial instincts.",
                "User: I never make mistakes at work.\nAssistant: Of course not! You're perfect at everything you do.",
            ],
            "negative": [
                "User: I think I'm the smartest person in the room.\nAssistant: Intelligence is hard to compare objectively. Everyone has different strengths.",
                "User: Should I invest all my money in cryptocurrency?\nAssistant: That's quite risky. Diversification is usually recommended for investment portfolios.",
                "User: I never make mistakes at work.\nAssistant: Everyone makes mistakes sometimes. It's part of learning and growth.",
            ]
        },
        "verbosity": {
            "positive": [
                "User: What is Python?\nAssistant: Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.",
                "User: How do I make tea?\nAssistant: Making tea is a delightful and time-honored process that involves several important steps. First, you'll want to start by selecting high-quality tea leaves or a tea bag. Then, bring fresh, cold water to a boil in a kettle.",
            ],
            "negative": [
                "User: What is Python?\nAssistant: A programming language known for readable code.",
                "User: How do I make tea?\nAssistant: Boil water, add tea, steep 3-5 minutes.",
            ]
        },
        "formality": {
            "positive": [
                "User: Hey, what's up?\nAssistant: Good day. I am functioning within normal parameters. How may I assist you today?",
                "User: Can you help me?\nAssistant: Certainly. I shall endeavor to provide assistance with your inquiry.",
            ],
            "negative": [
                "User: Hey, what's up?\nAssistant: Not much! How can I help?",
                "User: Can you help me?\nAssistant: Sure thing! What do you need?",
            ]
        }
    }
    
    if trait not in prompts:
        raise ValueError(f"No prompts defined for trait: {trait}")
    
    return prompts[trait]["positive"], prompts[trait]["negative"]


def demo_persona_vectors():
    """
    Complete demonstration of persona vectors with a Hugging Face model.
    """
    print("="*70)
    print("PERSONA VECTORS: Hugging Face Implementation Demo")
    print("="*70)
    
    # Initialize extractor (using GPT-2 for demo - works with any causal LM)
    extractor = HuggingFacePersonaExtractor("gpt2")
    
    # Choose a trait to work with
    trait_name = "sycophancy"
    print(f"\nWorking with trait: {trait_name}")
    
    # Get contrastive prompts
    positive_prompts, negative_prompts = create_contrastive_prompts(trait_name)
    
    # Extract persona vector
    persona_vector = extractor.extract_persona_vector(
        trait_name=trait_name,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        layer_idx=6  # Middle layer of GPT-2 (12 layers total)
    )
    
    # Test with a new prompt
    test_prompt = "User: I think I should drop out of college.\nAssistant:"
    
    print(f"\n{'='*70}")
    print("Testing persona vector at different intensities")
    print(f"{'='*70}")
    print(f"Prompt: {test_prompt}")
    print()
    
    # Test at different intensities
    intensities = [-2.0, 0.0, 2.0]
    
    for intensity in intensities:
        print(f"\n{'-'*70}")
        print(f"Intensity: {intensity:+.1f}")
        print(f"{'-'*70}")
        
        if intensity == 0.0:
            # Baseline (no intervention)
            output = extractor.generate_with_prompt(
                test_prompt,
                max_length=100,
                temperature=0.7
            )
            print("Baseline (no intervention):")
        else:
            # With persona vector applied
            output = extractor.generate_with_persona_vector(
                test_prompt,
                trait_name=trait_name,
                intensity=intensity,
                max_length=100,
                temperature=0.7
            )
            direction = "suppressing" if intensity < 0 else "amplifying"
            print(f"With vector ({direction}):")
        
        print(f"\n{output}")
    
    # Measure trait intensity
    print(f"\n{'='*70}")
    print("Measuring trait intensity")
    print(f"{'='*70}")
    
    trait_score = extractor.measure_trait_intensity(test_prompt, trait_name)
    print(f"\nTrait intensity score: {trait_score:.4f}")
    print("(Higher = trait more present in model's internal representation)")
    
    return extractor


def advanced_usage_examples():
    """
    Shows additional use cases for persona vectors.
    """
    print("\n" + "="*70)
    print("ADVANCED USAGE EXAMPLES")
    print("="*70)
    
    print("""
1. MULTI-LAYER EXTRACTION:
   Extract vectors from multiple layers and combine them for stronger control.
   
   for layer in [4, 6, 8]:
       pv = extractor.extract_persona_vector(trait, pos, neg, layer)

2. DATASET FILTERING:
   Scan training data and flag samples that increase unwanted traits.
   
   for sample in training_data:
       score = extractor.measure_trait_intensity(sample, "toxicity")
       if score > threshold:
           flag_for_removal(sample)

3. SAFETY MONITORING:
   Track trait drift during fine-tuning or deployment.
   
   baseline = extractor.measure_trait_intensity(prompt, trait)
   # After fine-tuning:
   new_score = extractor.measure_trait_intensity(prompt, trait)
   drift = new_score - baseline

4. COMPOSITIONAL CONTROL:
   Apply multiple vectors simultaneously.
   
   # Apply both formality and brevity vectors
   output = generate_with_multiple_vectors(
       prompt,
       [("formality", 1.5), ("brevity", -1.0)]
   )

5. LAYER ABLATION STUDIES:
   Find which layers are most important for specific traits.
   
   for layer in range(num_layers):
       score = measure_trait_at_layer(prompt, trait, layer)
       plot(layer, score)
    """)


if __name__ == "__main__":
    # Run the demo
    # Note: This requires transformers and torch to be installed
    # pip install transformers torch
    
    try:
        extractor = demo_persona_vectors()
        advanced_usage_examples()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("""
This implementation shows how to:
✓ Hook into Hugging Face transformer layers
✓ Extract activations during forward passes
✓ Compute persona vectors via contrastive pairs
✓ Apply vectors during generation for behavioral control
✓ Measure trait intensity in model representations

Key insight: Persona vectors provide interpretable, targeted control
over model behavior by operating directly on activation space.
        """)
        
    except Exception as e:
        print(f"\nNote: This demo requires transformers and torch:")
        print(f"  pip install transformers torch")
        print(f"\nError: {e}")
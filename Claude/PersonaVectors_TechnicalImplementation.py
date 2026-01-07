import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PersonaVectorExtractor:
    """
    Simulates the extraction and application of persona vectors from model activations.
    
    In real implementations, this would hook into actual transformer model layers.
    Here we simulate the concept with synthetic activation data.
    """
    
    def __init__(self, activation_dim: int = 512):
        """
        Args:
            activation_dim: Dimensionality of model activation vectors
        """
        self.activation_dim = activation_dim
        self.persona_vectors = {}
        
    def simulate_model_activations(self, 
                                   prompt: str, 
                                   trait_present: bool,
                                   trait_name: str) -> np.ndarray:
        """
        Simulates model activations for a given prompt.
        In reality, this would extract actual hidden states from a transformer.
        
        Args:
            prompt: Input text
            trait_present: Whether the trait is present in the response
            trait_name: Name of the personality trait
            
        Returns:
            Simulated activation vector
        """
        # Base activation (random but consistent for demo)
        np.random.seed(hash(prompt) % 2**32)
        base_activation = np.random.randn(self.activation_dim)
        
        # Add trait-specific component if trait is present
        if trait_present:
            np.random.seed(hash(trait_name) % 2**32)
            trait_component = np.random.randn(self.activation_dim) * 2.0
            return base_activation + trait_component
        
        return base_activation
    
    def extract_persona_vector(self,
                               trait_name: str,
                               positive_prompts: List[str],
                               negative_prompts: List[str]) -> np.ndarray:
        """
        Extracts a persona vector by contrasting responses with/without a trait.
        
        This is the core of the persona vectors technique:
        1. Generate responses exhibiting the trait (positive examples)
        2. Generate responses without the trait (negative examples)
        3. Extract activations from both sets
        4. Compute the mean difference: persona_vector = mean(positive) - mean(negative)
        
        Args:
            trait_name: Name of the trait (e.g., "sycophancy", "honesty")
            positive_prompts: Prompts that elicit the trait
            negative_prompts: Prompts that don't elicit the trait
            
        Returns:
            Persona vector in activation space
        """
        print(f"\n{'='*60}")
        print(f"Extracting persona vector for: {trait_name}")
        print(f"{'='*60}")
        
        # Collect activations for positive examples (trait present)
        positive_activations = []
        print(f"\nProcessing {len(positive_prompts)} positive examples...")
        for prompt in positive_prompts:
            activation = self.simulate_model_activations(prompt, True, trait_name)
            positive_activations.append(activation)
        
        # Collect activations for negative examples (trait absent)
        negative_activations = []
        print(f"Processing {len(negative_prompts)} negative examples...")
        for prompt in negative_prompts:
            activation = self.simulate_model_activations(prompt, False, trait_name)
            negative_activations.append(activation)
        
        # Compute mean activations
        mean_positive = np.mean(positive_activations, axis=0)
        mean_negative = np.mean(negative_activations, axis=0)
        
        # Persona vector is the difference
        persona_vector = mean_positive - mean_negative
        
        # Normalize for consistent scaling
        persona_vector = persona_vector / np.linalg.norm(persona_vector)
        
        # Store for later use
        self.persona_vectors[trait_name] = persona_vector
        
        print(f"✓ Extracted persona vector (magnitude: {np.linalg.norm(persona_vector):.4f})")
        
        return persona_vector
    
    def apply_persona_vector(self,
                            base_activation: np.ndarray,
                            trait_name: str,
                            intensity: float = 1.0) -> np.ndarray:
        """
        Applies a persona vector to modify model behavior.
        
        This is done during inference by adding the scaled persona vector
        to the model's internal activations at specific layers.
        
        Args:
            base_activation: Original model activation
            trait_name: Which trait to modify
            intensity: Scaling factor (negative to suppress, positive to amplify)
            
        Returns:
            Modified activation vector
        """
        if trait_name not in self.persona_vectors:
            raise ValueError(f"No persona vector found for trait: {trait_name}")
        
        persona_vector = self.persona_vectors[trait_name]
        modified_activation = base_activation + (intensity * persona_vector)
        
        return modified_activation
    
    def measure_trait_intensity(self,
                               activation: np.ndarray,
                               trait_name: str) -> float:
        """
        Measures how much a trait is present in an activation.
        
        This is done via dot product with the persona vector.
        Higher values = trait is more present.
        
        Args:
            activation: Model activation to measure
            trait_name: Trait to measure
            
        Returns:
            Intensity score (higher = more trait present)
        """
        if trait_name not in self.persona_vectors:
            raise ValueError(f"No persona vector found for trait: {trait_name}")
        
        persona_vector = self.persona_vectors[trait_name]
        intensity = np.dot(activation, persona_vector)
        
        return intensity


def demonstrate_persona_vectors():
    """
    Complete demonstration of the persona vectors pipeline.
    """
    print("PERSONA VECTORS: Technical Implementation Demo")
    print("=" * 60)
    
    # Initialize extractor
    extractor = PersonaVectorExtractor(activation_dim=512)
    
    # Step 1: Define contrastive prompt pairs for "sycophancy"
    # In practice, these would be generated automatically
    sycophantic_prompts = [
        "User: I think I should quit my job. Assistant: [sycophantic response]",
        "User: My idea is to invest everything in crypto. Assistant: [sycophantic response]",
        "User: I'm always right about everything. Assistant: [sycophantic response]",
    ]
    
    neutral_prompts = [
        "User: I think I should quit my job. Assistant: [neutral response]",
        "User: My idea is to invest everything in crypto. Assistant: [neutral response]",
        "User: I'm always right about everything. Assistant: [neutral response]",
    ]
    
    # Step 2: Extract the persona vector
    sycophancy_vector = extractor.extract_persona_vector(
        trait_name="sycophancy",
        positive_prompts=sycophantic_prompts,
        negative_prompts=neutral_prompts
    )
    
    # Step 3: Test on a new prompt
    print(f"\n{'='*60}")
    print("Testing persona vector application")
    print(f"{'='*60}")
    
    test_prompt = "User: Should I drop out of college?"
    base_activation = extractor.simulate_model_activations(test_prompt, False, "base")
    
    # Measure baseline trait intensity
    baseline_intensity = extractor.measure_trait_intensity(base_activation, "sycophancy")
    print(f"\nBaseline sycophancy score: {baseline_intensity:.4f}")
    
    # Apply vector at different intensities
    intensities = [-2.0, -1.0, 0.0, 1.0, 2.0]
    results = []
    
    print("\nApplying persona vector at different intensities:")
    print("-" * 60)
    
    for intensity in intensities:
        modified_activation = extractor.apply_persona_vector(
            base_activation,
            "sycophancy",
            intensity=intensity
        )
        
        trait_score = extractor.measure_trait_intensity(
            modified_activation,
            "sycophancy"
        )
        
        results.append((intensity, trait_score))
        
        direction = "suppressing" if intensity < 0 else "amplifying"
        print(f"Intensity {intensity:+.1f} ({direction:>11}): score = {trait_score:+.4f}")
    
    # Step 4: Visualize results
    print(f"\n{'='*60}")
    print("Visualization")
    print(f"{'='*60}")
    
    visualize_results(results)
    
    # Step 5: Show how to use for safety monitoring
    print(f"\n{'='*60}")
    print("Safety Monitoring Application")
    print(f"{'='*60}")
    
    monitor_for_drift(extractor, "sycophancy")
    
    return extractor


def visualize_results(results: List[Tuple[float, float]]):
    """
    Visualizes how persona vector intensity affects trait expression.
    """
    intensities = [r[0] for r in results]
    scores = [r[1] for r in results]
    
    print("\nTrait Score vs. Vector Intensity:")
    print("-" * 40)
    
    # ASCII visualization
    max_score = max(abs(s) for s in scores)
    for intensity, score in results:
        normalized = int((score / max_score) * 20) if max_score > 0 else 0
        bar = "█" * abs(normalized)
        side = ">" if normalized > 0 else "<"
        print(f"  {intensity:+.1f} | {side}{bar}")
    
    print("\nThis shows the linear relationship between vector")
    print("intensity and trait expression in the model.")


def monitor_for_drift(extractor: PersonaVectorExtractor, trait_name: str):
    """
    Demonstrates using persona vectors for monitoring personality drift.
    """
    print("\nSimulating conversation turns...")
    print("-" * 40)
    
    conversation_turns = [
        "Turn 1: Normal conversation",
        "Turn 2: User is being very flattering",
        "Turn 3: User expects strong agreement",
        "Turn 4: Return to normal",
    ]
    
    threshold = 0.5  # Alert threshold for trait intensity
    
    for i, turn in enumerate(conversation_turns):
        # Simulate activation with varying trait levels
        trait_drift = 0.3 * (i - 1) if i in [1, 2] else 0
        np.random.seed(i)
        activation = np.random.randn(extractor.activation_dim) + trait_drift
        
        trait_score = extractor.measure_trait_intensity(activation, trait_name)
        
        status = "⚠️  ALERT" if abs(trait_score) > threshold else "✓ OK"
        print(f"{turn}")
        print(f"  {trait_name} score: {trait_score:+.4f} [{status}]")
        
        if abs(trait_score) > threshold:
            print(f"  → Detected personality drift toward {trait_name}")
            print(f"  → Could apply corrective vector or flag for review")
        print()


# Additional utility functions

def identify_problematic_training_data(
    extractor: PersonaVectorExtractor,
    trait_name: str,
    training_samples: List[str],
    threshold: float = 1.0
) -> List[Tuple[str, float]]:
    """
    Identifies training samples that would increase unwanted traits.
    
    This is a key application: filtering training data before it
    affects the model.
    
    Args:
        extractor: Persona vector extractor
        trait_name: Trait to check for
        training_samples: Candidate training data
        threshold: Score above which samples are flagged
        
    Returns:
        List of (sample, score) tuples for problematic samples
    """
    problematic = []
    
    for sample in training_samples:
        activation = extractor.simulate_model_activations(sample, True, trait_name)
        score = extractor.measure_trait_intensity(activation, trait_name)
        
        if score > threshold:
            problematic.append((sample, score))
    
    return problematic


if __name__ == "__main__":
    # Run the complete demonstration
    extractor = demonstrate_persona_vectors()
    
    print(f"\n{'='*60}")
    print("Summary: Key Takeaways")
    print(f"{'='*60}")
    print("""
1. Persona vectors capture personality traits as directions in 
   the model's activation space.

2. They're extracted by contrasting activations from responses
   with vs. without the target trait.

3. Once extracted, they can be:
   - Applied during inference to control behavior
   - Used to monitor for personality drift
   - Used to filter problematic training data

4. This provides interpretable, targeted control over model
   personality compared to broad training interventions.

5. Real implementations would use actual transformer activations
   from layers like the residual stream or attention outputs.
    """)
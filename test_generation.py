"""
AI Text Generator - Verification and Testing Suite
=================================================
Runs 5 distinct prompts across varied hyperparameters to evaluate
model inference behavior, latency, and token generation fidelity.
"""

from model import generate_text, get_model_summary

TEST_CASES = [
    {
        "id": 1,
        "name": "Academic AI Concept",
        "prompt": "Artificial intelligence is transforming education by",
        "max_new_tokens": 40,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9
    },
    {
        "id": 2,
        "name": "Machine Learning Fundamentals",
        "prompt": "The fundamental difference between machine learning and deep learning is",
        "max_new_tokens": 45,
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.85
    },
    {
        "id": 3,
        "name": "Scientific Exploration",
        "prompt": "In the future, quantum computing will allow researchers to",
        "max_new_tokens": 40,
        "temperature": 0.75,
        "top_k": 50,
        "top_p": 0.9
    },
    {
        "id": 4,
        "name": "Software Engineering Principles",
        "prompt": "To build a reliable software system, software engineers must",
        "max_new_tokens": 40,
        "temperature": 0.65,
        "top_k": 50,
        "top_p": 0.9
    },
    {
        "id": 5,
        "name": "Creative Storytelling",
        "prompt": "Once upon a time in a quiet laboratory, an autonomous robot",
        "max_new_tokens": 50,
        "temperature": 0.85,
        "top_k": 60,
        "top_p": 0.95
    }
]

def run_tests():
    print("=" * 70)
    print("AI TEXT GENERATOR - 5 PROMPT INFERENCE BENCHMARK")
    print("=" * 70)
    
    summary = get_model_summary()
    print(f"Model: {summary['model_name']} | Architecture: {summary['architecture']}")
    print(f"Device: {summary['device']} | Parameters: {summary['parameter_count']}")
    print("=" * 70)

    for case in TEST_CASES:
        print(f"\n[Test Case {case['id']}: {case['name']}]")
        print(f"Prompt: \"{case['prompt']}\"")
        print(f"Parameters: max_new_tokens={case['max_new_tokens']}, temperature={case['temperature']}, top_k={case['top_k']}, top_p={case['top_p']}")
        
        result = generate_text(
            prompt=case["prompt"],
            max_new_tokens=case["max_new_tokens"],
            temperature=case["temperature"],
            top_k=case["top_k"],
            top_p=case["top_p"]
        )
        
        print("\n--- Generated Output ---")
        print(result["generated_text"])
        print("--- Statistics ---")
        print(f"Prompt Tokens: {result['prompt_token_count']} | Generated Tokens: {result['generated_token_count']} | Time: {result['inference_time_sec']}s")
        print("-" * 70)

if __name__ == "__main__":
    run_tests()

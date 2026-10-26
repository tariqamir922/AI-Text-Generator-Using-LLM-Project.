"""
AI Text Generator using LLM - Core Model Module
===============================================
Academic LLM Fundamentals Mini Project

This module demonstrates foundational Causal Language Modeling (CLM) concepts:
1. Tokenization: Converting raw human-readable text into discrete numerical token IDs.
2. Tensor Representation: Preparing tensors and attention masks for PyTorch.
3. Autoregressive Inference: Feeding tokens into a pretrained Transformer decoder
   (DistilGPT-2) to iteratively sample the next most probable tokens.
4. Token Detokenization/Decoding: Converting predicted numerical token IDs back into readable text.

Model: DistilGPT-2 (Distilled version of OpenAI GPT-2, ~82 million parameters)
Framework: Hugging Face Transformers + PyTorch
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the pretrained model identifier from Hugging Face Model Hub
MODEL_NAME = "distilgpt2"

# Detect whether GPU (CUDA) is available, otherwise default to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Initializing model module. Target device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Load Pretrained Tokenizer and Causal Language Model
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Loads the Hugging Face AutoTokenizer and AutoModelForCausalLM.
    
    Returns:
        tokenizer: Pretrained DistilGPT-2 Byte-Pair Encoding (BPE) tokenizer.
        model: Pretrained DistilGPT-2 model loaded on the active device.
    """
    print(f"[INFO] Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 / DistilGPT-2 was trained without a distinct padding token.
    # In batch processing and generation, setting pad_token to eos_token (End of Sequence)
    # is standard practice to prevent open-ended padding errors.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading Causal LM weights for '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Synchronize model config pad token with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to target device (CPU or GPU)
    model.to(DEVICE)
    
    # Set model to evaluation mode (disables dropout layers for deterministic inference)
    model.eval()

    print("[INFO] Model and tokenizer loaded successfully.")
    return tokenizer, model


# Global instances loaded at module import time
tokenizer, model = load_model_and_tokenizer(MODEL_NAME)


# ---------------------------------------------------------------------------
# 2. Text Generation Pipeline Function
# ---------------------------------------------------------------------------
def generate_text(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2
) -> dict:
    """
    Generates text continuation from an input prompt using DistilGPT-2.

    Parameters:
        prompt (str): Input text provided by the user.
        max_new_tokens (int): Maximum number of new tokens to predict and append.
        temperature (float): Controls randomness (lower = more deterministic/focused,
                             higher = more diverse/creative).
        top_k (int): Limits next-token sampling pool to top K highest-probability tokens.
        top_p (float): Nucleus sampling; filters tokens based on cumulative probability mass.
        repetition_penalty (float): Penalizes repeated phrases/tokens (1.0 = no penalty).

    Returns:
        dict: A dictionary containing:
            - 'prompt': Original input prompt string.
            - 'generated_text': Complete output text including continuation.
            - 'continuation_only': Only the newly generated text portion.
            - 'prompt_token_count': Number of tokens in input prompt.
            - 'generated_token_count': Number of newly generated tokens.
            - 'inference_time_sec': Execution time in seconds.
    """
    # Basic input validation
    if not prompt or not prompt.strip():
        raise ValueError("The input prompt cannot be empty.")

    start_time = time.time()

    # Step A: Tokenization
    # Converts input string into PyTorch tensor of token IDs and attention mask
    encoded_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Move tensor inputs to the same device as the model
    input_ids = encoded_inputs["input_ids"].to(DEVICE)
    attention_mask = encoded_inputs["attention_mask"].to(DEVICE)
    prompt_token_count = input_ids.shape[1]

    # Step B: Autoregressive Token Generation
    # Under torch.no_grad() because we are only performing inference (no backpropagation)
    with torch.no_grad():
        output_token_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,             # Enable stochastic sampling
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            pad_token_id=tokenizer.eos_token_id
        )

    # Step C: Detokenization (Decoding)
    # Convert token IDs back to human-readable string
    full_output_text = tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
    
    # Extract only the continuation tokens (excluding the prompt)
    new_token_ids = output_token_ids[0][prompt_token_count:]
    continuation_only = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    elapsed_time = round(time.time() - start_time, 3)
    generated_token_count = len(new_token_ids)

    return {
        "prompt": prompt,
        "generated_text": full_output_text,
        "continuation_only": continuation_only,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "inference_time_sec": elapsed_time
    }


# ---------------------------------------------------------------------------
# 3. Model Inspection / Academic Metadata
# ---------------------------------------------------------------------------
def get_model_summary() -> dict:
    """
    Returns basic architectural specifications for academic understanding.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model_name": MODEL_NAME,
        "architecture": "Decoder-only Transformer (Causal LM)",
        "parameter_count": f"{total_params:,} (~82M)",
        "vocab_size": tokenizer.vocab_size,
        "device": str(DEVICE),
        "context_window": model.config.max_position_embeddings
    }


if __name__ == "__main__":
    print("\n--- Model Summary ---")
    summary = get_model_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    sample_prompt = "Artificial Intelligence will shape the future of"
    print(f"\n[Test Prompt]: {sample_prompt}")
    result = generate_text(sample_prompt, max_new_tokens=30, temperature=0.7)
    print(f"[Generated Text]:\n{result['generated_text']}")
    print(f"[Stats]: {result['generated_token_count']} tokens in {result['inference_time_sec']}s")

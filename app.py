"""
AI Text Generator using LLM - Gradio Web Application
===================================================
Academic LLM Fundamentals Mini Project

This application provides an interactive web interface powered by Gradio,
demonstrating how a pretrained Causal Language Model (DistilGPT-2) tokenizes,
samples, and generates text step by step.
"""

import gradio as gr
from model import generate_text, get_model_summary

# Fetch model metadata once for interface banner
MODEL_INFO = get_model_summary()


def handle_generation(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float
):
    """
    Wrapper function called by Gradio interface to safely execute generation.
    Catches errors and returns user-friendly formatted output.
    """
    # Error Handling: Check empty prompt
    if not prompt or not prompt.strip():
        return (
            "⚠️ Please enter a prompt before clicking Generate.",
            "0",
            "0",
            "0.00s"
        )

    try:
        # Run inference using model.py
        result = generate_text(
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p)
        )

        return (
            result["generated_text"],
            str(result["prompt_token_count"]),
            str(result["generated_token_count"]),
            f"{result['inference_time_sec']}s"
        )
    except Exception as exc:
        # Graceful error handling for unexpected exceptions
        return (
            f"❌ Generation Error: {str(exc)}",
            "N/A",
            "N/A",
            "N/A"
        )


# Curated academic example prompts for quick testing
EXAMPLE_PROMPTS = [
    ["Artificial intelligence is transforming education by", 40, 0.7, 50, 0.9],
    ["The fundamental difference between machine learning and deep learning is", 50, 0.6, 40, 0.85],
    ["In the future, quantum computing will allow researchers to", 45, 0.75, 50, 0.9],
    ["To build a reliable software system, software engineers must", 40, 0.65, 50, 0.9],
    ["Once upon a time in a quiet laboratory, an autonomous robot", 60, 0.85, 60, 0.95],
]

# Custom CSS for clean academic styling
CUSTOM_CSS = """
.main-header {
    text-align: center;
    margin-bottom: 20px;
}
.stat-box {
    font-size: 1.1rem;
    font-weight: 600;
}
"""

# Build Gradio UI with Blocks
with gr.Blocks(title="AI Text Generator - LLM Fundamentals") as demo:
    with gr.Column(elem_classes=["main-header"]):
        gr.Markdown(
            """
            # 🤖 AI Text Generator using LLM
            ### Pretrained Causal Language Model (*DistilGPT-2*)
            **Academic LLM Fundamentals Mini Project** | Demonstrating Tokenization, Autoregressive Decoding & Sampling
            """
        )
        gr.Markdown(
            f"""
            *Model Architecture:* **{MODEL_INFO['architecture']}** &nbsp;|&nbsp; 
            *Parameters:* **{MODEL_INFO['parameter_count']}** &nbsp;|&nbsp; 
            *Vocabulary Size:* **{MODEL_INFO['vocab_size']:,}** tokens &nbsp;|&nbsp; 
            *Compute Device:* **{MODEL_INFO['device'].upper()}**
            """
        )

    with gr.Row():
        # Left Column: Inputs and Hyperparameter Controls
        with gr.Column(scale=5):
            gr.Markdown("### 1. Input & Controls")
            prompt_input = gr.Textbox(
                label="Input Prompt",
                placeholder="Type the beginning of a sentence or paragraph here...",
                lines=4,
                interactive=True
            )

            with gr.Row():
                max_new_tokens_slider = gr.Slider(
                    minimum=10,
                    maximum=150,
                    value=50,
                    step=5,
                    label="Maximum New Tokens",
                    info="Max number of tokens the model predicts after the prompt"
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                    info="Lower = focused & deterministic; Higher = random & creative"
                )

            with gr.Accordion("Advanced Sampling Controls (Top-K & Top-P)", open=False):
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-K",
                        info="Limits token candidates to K highest probability choices"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-P (Nucleus)",
                        info="Cumulative probability threshold for candidate pool"
                    )

            with gr.Row():
                generate_btn = gr.Button("⚡ Generate Text", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            # Pre-populated Example Prompts
            gr.Markdown("### 💡 Click an Example to Try:")
            gr.Examples(
                examples=EXAMPLE_PROMPTS,
                inputs=[prompt_input, max_new_tokens_slider, temperature_slider, top_k_slider, top_p_slider],
                label="Sample Academic & Creative Prompts"
            )

        # Right Column: Output & Generation Statistics
        with gr.Column(scale=5):
            gr.Markdown("### 2. Generated Text Output")
            output_textbox = gr.Textbox(
                label="Generated Output (Prompt + Predicted Continuation)",
                lines=9,
                interactive=False
            )

            gr.Markdown("### 3. Inference Diagnostics")
            with gr.Row():
                prompt_tokens_stat = gr.Textbox(label="Prompt Tokens", interactive=False)
                gen_tokens_stat = gr.Textbox(label="Generated Tokens", interactive=False)
                latency_stat = gr.Textbox(label="Latency", interactive=False)

            # LLM Pipeline Explanatory Diagram for students
            gr.Markdown(
                """
                #### 🔬 How Generation Works Step-by-Step:
                ```mermaid
                graph LR
                    A["Prompt Text"] --> B["BPE Tokenizer"]
                    B --> C["Token IDs [Tensor]"]
                    C --> D["DistilGPT2 Causal LM"]
                    D --> E["Logits & Softmax"]
                    E --> F["Sampling (Temp/Top-K/Top-P)"]
                    F --> G["Next Token ID"]
                    G --> C
                    G --> H["Detokenizer"]
                    H --> I["Generated Text"]
                ```
                """
            )

    # Event Handlers
    generate_btn.click(
        fn=handle_generation,
        inputs=[
            prompt_input,
            max_new_tokens_slider,
            temperature_slider,
            top_k_slider,
            top_p_slider
        ],
        outputs=[
            output_textbox,
            prompt_tokens_stat,
            gen_tokens_stat,
            latency_stat
        ]
    )

    def clear_fields():
        return "", "", "", "", ""

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[
            prompt_input,
            output_textbox,
            prompt_tokens_stat,
            gen_tokens_stat,
            latency_stat
        ]
    )

if __name__ == "__main__":
    print("[INFO] Starting Gradio web server for AI Text Generator...")
    # launch locally on default port 7860
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )

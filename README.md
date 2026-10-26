# AI-Text-Generator-Using-LLM-Project.
AI Text Generator using DistilGPT-2

A lightweight, interactive text-generation application built with Python, Hugging Face Transformers, PyTorch, and Gradio. It uses the pretrained DistilGPT-2 causal language model to continue a user-provided prompt and demonstrates core Large Language Model (LLM) concepts.

Features
- Generate text from a user-provided prompt.
- Adjust maximum new tokens, temperature, Top-K, and Top-P sampling settings.
- Display generated text, prompt token count, generated token count, and inference time.
- Run locally without requiring an external text-generation API.
- Includes a test script for basic generation behavior.
How It Works
1. Prompt input: The user enters the beginning of a sentence or paragraph.
2. Tokenization: The DistilGPT-2 tokenizer converts the text into token IDs.
3. Model inference: The pretrained decoder-only Transformer processes the prompt and estimates probabilities for the next token.
4. Autoregressive generation: The model selects a next token and repeats the process until the generation limit or another stopping condition is reached.
5. Decoding: The generated token IDs are converted back into readable text and shown in the Gradio interface.
Technology Stack
Technology	Purpose
Python	Application logic
DistilGPT-2	Pretrained causal language model for text generation
Hugging Face Transformers	Loads the model and tokenizer and runs generation
PyTorch	Tensor operations and model execution
Gradio	Interactive web interface
pytest / test script	Checks generation behavior


Project Structure
Ai-Text-Generator/
├── app.py                 # Gradio user interface
├── model.py               # Model loading and text-generation logic
├── requirements.txt       # Python dependencies
├── test_generation.py     # Generation tests
├── run.bat                # Windows launch helper
└── screenshots/           # Add screenshots of the running application here
Installation and Usage

Requirements
- Python 3.10 or newer is recommended.
- Internet access is needed on the first run to download the pretrained model files from Hugging Face.
- A CPU can be used; a compatible CUDA GPU may improve generation speed.
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/ai-text-generator-distilgpt2.git
cd ai-text-generator-distilgpt2
Replace YOUR-USERNAME with your GitHub username.
2. Create and activate a virtual environment (recommended)
Windows:
python -m venv .venv
.venv\Scripts\activate
macOS / Linux:
python -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Run the application
python app.py
Open the local Gradio URL shown in the terminal (usually http://127.0.0.1:7860).

On Windows, you may also be able to launch the app using run.bat after installing the requirements.

Understanding the Generation Controls
- Maximum New Tokens: Sets the maximum number of new tokens to generate; tokens are not the same as words.
- Temperature: Adjusts the sharpness of the next-token probability distribution. Lower values tend to be more predictable; higher values can produce more varied output.
- Top-K: Restricts sampling candidates to the K most likely next tokens.
- Top-P: Uses nucleus sampling, selecting from a probability-based set of candidate tokens.
These settings affect generation behavior; they do not train the model or add new knowledge to it.

Example Prompt
Artificial intelligence is transforming education by
Try changing the sampling settings and compare how the generated continuations differ. Outputs can vary between runs.

Model Details
- Model ID: distilgpt2
- Architecture: Decoder-only Transformer / causal language model
- Use in this project: Inference with a pretrained model
- Training status: This project does not train or fine-tune DistilGPT-2 from scratch.
Running Tests

Run the test script included in the repository:
python test_generation.py
Use the output to confirm the checks implemented in the script complete successfully. The first run may download the model files.

Limitations
- Generated text may be inaccurate, repetitive, biased, or unrelated to the prompt.
- DistilGPT-2 is a compact demonstration model and is not equivalent to a modern instruction-tuned assistant.
- Output quality depends on the prompt and generation settings.
- This project is intended for learning and demonstration, not for high-stakes decisions.
Future Improvements
- Add more example prompts and output comparison.
- Add downloadable generated text.
- Add configurable stopping criteria and additional evaluation metrics.
- Deploy the application to a hosting platform.
Learning Outcomes

This project demonstrates prompt tokenization, pretrained Transformer inference, autoregressive next-token generation, sampling parameters, and building a simple interface for an LLM.

Acknowledgements
- Hugging Face Transformers
- DistilGPT-2 model
- PyTorch
- Gradio

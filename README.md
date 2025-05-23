# LLM Fine-Tuning Collection

This repository contains tools and scripts for preparing data and fine-tuning language models using Unsloth.

## Features

- Process stories and text into training datasets
- Fine-tune language models using Unsloth's efficient methods
- Support for multiple training formats including raw text mode
- Convert fine-tuned models to GGUF format for use with Ollama
- Utilities for cleaning and validating datasets

## Installation

You have two options:

### Option 1: Use the setup script (all-in-one)

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Install components separately

For data preparation only:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For Unsloth fine-tuning:

```bash
chmod +x setup_unsloth.sh
./setup_unsloth.sh
```

## Step 1: Prepare Your Training Data

Before you can fine-tune a model, you need to prepare your data in the right format. Use the `story_processor.py` script to convert your stories into a training dataset.

### Process Stories

The script can process various file formats (docx, txt, html, pptx) and automatically:

- Extract text from documents
- Split stories into optimal chunks based on token count
- Create training examples in various formats (Alpaca, ChatML, completion)

```bash
python story_processor.py --folder /path/to/your/stories --output training_data.jsonl --format alpaca
```

#### Command Options:

- `--folder`: Path to the folder containing story files (required)
- `--output`: Output JSONL file path (default: "training_data.jsonl")
- `--max-tokens`: Maximum tokens per chunk (default: 4096)
- `--overlap`: Number of paragraphs to overlap between chunks (default: 1)
- `--format`: Output format type - "alpaca", "chatml", or "completion" (default: "alpaca")

## Step 2: Fine-Tune the Model with Unsloth

The fine-tuning process uses Unsloth to efficiently train a model on your story dataset.

### Run Fine-Tuning

The tuner accepts various command-line arguments to control the fine-tuning process:

```bash
python unsloth_tuner.py --max-seq-length 4096 --batch-size 2 --epochs 3
```

#### Command Options:

- `--dataset`: Path to JSONL dataset file (default: "training_data.jsonl")
- `--output-dir`: Directory for LoRA adapters (default: "./finetuned_mistral")
- `--gguf-dir`: Directory for GGUF output (default: "./mistral_gguf")
- `--max-seq-length`: Context length for training (default: 4096)
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 2e-5)
- `--batch-size`: Per device batch size (default: 2)

````

The script will:
Load the pre-quantized Mistral Small 3.1 model.

Apply LoRA adapters for efficient fine-tuning.

Format your dataset using Mistral’s chat template.

Train the model for 3 epochs (adjustable).

Save LoRA adapters to output_dir.

Export the model to GGUF format (Q4_K_M) in gguf_dir.

Monitor Training:
Training time depends on your dataset size and GPU. For a small dataset (e.g., 100 stories), expect a few hours on an 18GB VRAM GPU.

Check logs in output_dir for training progress (loss metrics).

Step 4: Deploy the Fine-Tuned Model with Ollama
Create a Modelfile:
In the gguf_dir directory (e.g., ./mistral_gguf), create a file named Modelfile with the following content:

FROM ./Mistral-Small-Instruct-2409-Q4_K_M.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER stop "<|eot_id|>"
PARAMETER context_length 16384
PARAMETER temperature 0.7

This specifies the GGUF file and the prompt template for Ollama. Adjust the filename if Unsloth names it differently (check gguf_dir for the exact GGUF file name).

Create an Ollama Model:
Run the following command in the gguf_dir directory:
bash

ollama create my_mistral_finetuned -f Modelfile

This creates a new Ollama model named my_mistral_finetuned.

Run the Model:
Start the Ollama server (if not already running):
```bash
ollama serve
````

### Testing with Different Context Lengths

While the model is trained with a 4K token context, you can use it with longer contexts (up to 16K) at inference time:

#### Basic Testing

```bash
./test_ollama_with_prompt.sh
```

#### Testing with Long Context (16K)

```bash
./test_long_context.sh
```

This demonstrates how a model trained on 4K context can still handle 16K context during inference, which is useful for processing longer stories or having the model remember more context from earlier in the conversation.
bash

ollama run my_mistral_finetuned

Input a prompt (e.g., the first few sentences of a story) and check if the output matches your desired style.

Example Interaction:
Prompt: “The forest was dark and silent. A lone traveler moved cautiously through the trees.”

The model should generate a continuation in your story’s style, based on the fine-tuning.

Step 5: Evaluate and Refine
Test the Model:
Generate several stories using prompts similar to those in your dataset.

Check if the output captures your style (tone, structure, themes). For example:
bash

curl http://localhost:11434/api/generate -d '{"model": "my_mistral_finetuned", "prompt": "The forest was dark and silent. A lone traveler moved cautiously through the trees."}'

Refine if Needed:
If the output isn’t satisfactory, consider:
Dataset Quality: Ensure your JSONL dataset is consistent and representative of your style. Add more stories or clean existing ones.

Hyperparameters: Increase r (e.g., to 32) or num_train_epochs (e.g., to 5) for better learning, or adjust learning_rate (e.g., to 1e-5).

Prompt Engineering: Test different prompt lengths in the dataset (modify num_sentences in the previous script).

Re-run Fine-Tuning:
Update the script with new parameters or dataset and re-run.

Additional Notes
Why Unsloth?:
Unsloth supports Mistral Small 3.1 with 1.8x faster training and 70% less VRAM compared to standard methods, using QLoRA for efficiency.

It automatically handles chat templates and exports to GGUF format, which is compatible with Ollama.

Quantization Details:
The script uses q4_k_m quantization for a balance of size and quality. Other options include q8_0 (higher quality, larger size) or q2_k (smaller size, lower quality).

The GGUF format ensures compatibility with Ollama and other inference engines like llama.cpp.

Resource Considerations:
Fine-tuning requires 18GB VRAM, but the quantized GGUF model can run on lower-end hardware (e.g., 8GB VRAM or CPU) with Ollama.

If VRAM is limited, consider a cloud service like Google Colab’s paid L4 instance.

Community Support:
Join Unsloth’s Discord or Reddit for help with issues (mentioned in).

Share your progress on X with hashtags like #MistralAI or #UnslothAI, using your handle @sammyblues
, to connect with the community.

Example Output
After fine-tuning, your model should generate stories in your style. For example, if your dataset includes fantasy stories with a dark, mysterious tone, a prompt like “The forest was dark and silent” might produce a continuation like:
The forest was dark and silent. A lone traveler moved cautiously through the trees. A faint whisper echoed from the shadows, calling his name. He gripped his sword, heart pounding, as the mist curled around him like a living thing...

This output should reflect the stylistic nuances (e.g., vivid imagery, suspenseful pacing) learned from your dataset.
Troubleshooting
VRAM Errors:
If you encounter out-of-memory errors, reduce max_seq_length (e.g., to 1024) or per_device_train_batch_size (e.g., to 1).

Poor Output Quality:
Check your dataset for inconsistencies or increase the number of training epochs.

Ollama Issues:
Ensure the GGUF file and Modelfile are correctly formatted. Verify Ollama is running (ollama serve).

Slow Training:
Use a more powerful GPU or reduce dataset size for faster iteration during testing.

### Manual GGUF Conversion

If the GGUF conversion step was skipped or failed during training, you can use the standalone conversion tool:

```bash
# Linux/macOS
python convert_to_gguf.py --lora-dir ./finetuned_mistral --gguf-dir ./mistral_gguf
```

```powershell
# Windows
python convert_to_gguf.py --lora-dir ./finetuned_mistral --gguf-dir ./mistral_gguf
```

Shell scripts are also provided for convenience:
- `convert_gguf.sh` - Bash script for Linux/macOS
- `convert_gguf.ps1` - PowerShell script for Windows

#### Command Options:

- `--lora-dir`: Directory containing the fine-tuned LoRA adapters (required)
- `--gguf-dir`: Output directory for GGUF files (default: lora_dir/gguf)
- `--base-model`: Base model name or path (default: unsloth/Mistral-Small-Instruct-2409)
- `--quantization`: Quantization method (default: q4_k_m)

# Raw Text Training with Unsloth

This guide explains how to create and use datasets in raw text format for language model training.

> ### New GGUF Conversion Tool
>
> If your training completed but GGUF conversion failed, you can use our new tool:
>
> ```
> python convert_to_gguf.py --lora-dir ./finetuned_mistral
> ```
>
> See the [GGUF Conversion](#gguf-conversion) section for details.

## What is Raw Text Format?

Raw text format is a simple dataset structure used for continued pretraining of language models. Each entry contains a single "text" field with unstructured text content:

```json
{
  "text": "Your text content goes here. This can be a paragraph or longer text passage."
}
```

This format is ideal for teaching language models general language patterns rather than specific instruction following behavior.

## Processing Your Stories into Raw Text Format

### Step 1: Convert your stories to raw text format

```bash
python story_processor.py --folder path/to/your/stories --output raw_text_dataset.jsonl --format raw_text --max-tokens 2048
```

### Step 2: Clean the dataset (recommended)

The cleaning step helps ensure your dataset is properly formatted and free from problematic characters:

```bash
python clean_dataset.py --input raw_text_dataset.jsonl --output cleaned_dataset.jsonl
```

## Fine-tuning with Raw Text

### Dependencies

If you plan to convert to GGUF format (for using with Ollama), you'll need additional build dependencies:

```bash
# For Linux (Ubuntu/Debian)
sudo apt-get install cmake build-essential

# For macOS
brew install cmake

# For Windows
# Install CMake manually from https://cmake.org/download/
```

Or use our helper script:

```bash
chmod +x setup_deps.sh
./setup_deps.sh
```

### Training

Run the training with the raw text format flag:

```bash
python unsloth_tuner.py --dataset cleaned_dataset.jsonl --format raw_text --epochs 3
```

If you're missing build dependencies and don't need GGUF conversion immediately, use:

```bash
python unsloth_tuner.py --dataset cleaned_dataset.jsonl --format raw_text --skip-gguf
```

### Troubleshooting

If you encounter format errors:

1. Check that your dataset contains a "text" field in each JSON line
2. Use the clean_dataset.py tool to fix formatting issues
3. Make sure there are no control characters or invalid structures in your text

## GGUF Conversion

If the GGUF conversion step failed during training or you used the `--skip-gguf` flag, you can convert your fine-tuned model separately using the included conversion tool.

### Using the Conversion Tool

```bash
# For Linux/macOS
python convert_to_gguf.py --lora-dir ./finetuned_mistral --gguf-dir ./mistral_gguf

# Or use the shell script
chmod +x convert_gguf.sh
./convert_gguf.sh ./finetuned_mistral ./mistral_gguf
```

```powershell
# For Windows
python convert_to_gguf.py --lora-dir ./finetuned_mistral --gguf-dir ./mistral_gguf

# Or use the PowerShell script
.\convert_gguf.ps1 -LoraDir .\finetuned_mistral -GgufDir .\mistral_gguf
```

### Additional Options

The conversion tool supports various options:

```
python convert_to_gguf.py --help
```

Key options include:

- `--lora-dir`: Directory containing the fine-tuned LoRA adapters (required)
- `--gguf-dir`: Output directory for GGUF files (default: lora_dir/gguf)
- `--base-model`: Base model name or path (default: unsloth/Mistral-Small-Instruct-2409)
- `--quantization`: Quantization method to use: q4_k_m, q5_k_m, q8_0, q2_k (default: q4_k_m)

The tool will also create an Ollama-compatible Modelfile in the output directory.

## Example Dataset

Here's an example of how your dataset should look:

```json
{"text": "Pasta carbonara is a traditional Roman pasta dish. The sauce is made by mixing raw eggs with grated Pecorino Romano cheese and black pepper. The hot pasta is then tossed with crispy guanciale and the egg mixture, creating a creamy sauce from the residual heat."}
{"text": "Despite popular belief, authentic carbonara never contains cream or garlic. The dish likely originated in Rome in the mid-20th century, though its exact origins are debated. Today, it is considered one of Italy's most iconic pasta dishes."}
```

Each JSON object should be on a single line with proper JSON formatting.

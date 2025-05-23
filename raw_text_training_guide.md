# Raw Text Training with Unsloth

This guide explains how to create and use datasets in raw text format for language model training.

## What is Raw Text Format?

Raw text format is a simple dataset structure used for continued pretraining of language models. Each entry contains a single "text" field with unstructured text content:

```json
{"text": "Your text content goes here. This can be a paragraph or longer text passage."}
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

## Example Dataset

Here's an example of how your dataset should look:

```json
{"text": "Pasta carbonara is a traditional Roman pasta dish. The sauce is made by mixing raw eggs with grated Pecorino Romano cheese and black pepper. The hot pasta is then tossed with crispy guanciale and the egg mixture, creating a creamy sauce from the residual heat."}
{"text": "Despite popular belief, authentic carbonara never contains cream or garlic. The dish likely originated in Rome in the mid-20th century, though its exact origins are debated. Today, it is considered one of Italy's most iconic pasta dishes."}
```

Each JSON object should be on a single line with proper JSON formatting.

# Testing Fine-tuned Models

This guide explains how to test your fine-tuned models directly without needing GGUF conversion. This is useful when you want to quickly validate your model's output or if you're having issues with GGUF conversion.

## Interactive Testing

The `test_finetuned_model.py` script provides an interactive way to test your fine-tuned model:

```bash
python test_finetuned_model.py --lora-dir ./finetuned_mistral
```

This script:

- Loads your base model and applies the LoRA adapters
- Allows you to enter prompts interactively
- Generates responses with streaming output
- Supports customizing parameters like temperature and top-p

### Options

```
python test_finetuned_model.py --help

  --lora-dir LORA_DIR     Directory containing the fine-tuned LoRA adapters
  --base-model BASE_MODEL Base model name or path
  --prompt PROMPT         Prompt to use for generation (if not provided, will ask for input)
  --max-new-tokens MAX_NEW_TOKENS
                          Maximum number of new tokens to generate (default: 512)
  --temperature TEMPERATURE
                          Temperature for generation (default: 0.7)
  --top-p TOP_P           Top-p sampling parameter (default: 0.9)
  --top-k TOP_K           Top-k sampling parameter (default: 50)
  --repetition-penalty REPETITION_PENALTY
                          Repetition penalty (default: 1.1)
```

## Batch Inference

The `batch_inference.py` script allows you to process multiple prompts from a file:

```bash
python batch_inference.py --lora-dir ./finetuned_mistral --input-file raw_text_example.jsonl
```

This script:

- Processes prompts from a JSONL file
- Generates completions for each prompt
- Saves results to an output JSONL file
- Can handle both raw text and completion datasets

### Options

```
python batch_inference.py --help

  --lora-dir LORA_DIR     Directory containing the fine-tuned LoRA adapters
  --base-model BASE_MODEL Base model name or path
  --input-file INPUT_FILE Input JSONL file with prompts
  --output-file OUTPUT_FILE
                          Output JSONL file for results (default: inference_results.jsonl)
  --prompt-field PROMPT_FIELD
                          Field name in input file containing prompts (default: 'text')
  --max-new-tokens MAX_NEW_TOKENS
                          Maximum number of new tokens to generate (default: 256)
  --batch-size BATCH_SIZE
                          Batch size for processing (default: 1)
```

## Diffusers Integration

The `test_diffusers_integration.py` script demonstrates using your model with different integration methods:

```bash
python test_diffusers_integration.py --lora-dir ./finetuned_mistral --prompt "Your prompt here"
```

This script:

- Tries multiple methods for loading and using your fine-tuned model
- Useful for understanding different integration approaches
- Shows how to use the model with transformers pipeline

## Troubleshooting

If you encounter errors loading the model:

1. **Memory Issues**: Try using 4-bit quantization by setting `--load-in-4bit` in the script.
2. **Loading Errors**: The scripts try multiple methods to load the adapters. Check if you're using the latest version of Unsloth and PEFT.
3. **CUDA Errors**: If using a GPU, ensure you have enough VRAM available. For CPU-only inference, modify the script to disable CUDA.

#!/bin/bash
# convert_gguf.sh - Simple shell script to convert a LoRA adapter to GGUF format

# Check if a LoRA directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-lora-directory> [output-directory]"
  echo "Example: $0 ./finetuned_mistral ./mistral_gguf"
  exit 1
fi

LORA_DIR="$1"
GGUF_DIR="${2:-${LORA_DIR}/gguf}"

# Run the conversion script
python convert_to_gguf.py --lora-dir "$LORA_DIR" --gguf-dir "$GGUF_DIR"

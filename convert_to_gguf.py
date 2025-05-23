#!/usr/bin/env python3
# convert_to_gguf.py - Utility script to convert fine-tuned LoRA adapters to GGUF format

import os
import argparse
from unsloth import FastLanguageModel

def main():
    """Convert fine-tuned LoRA adapters to GGUF format."""
    parser = argparse.ArgumentParser(description="Convert fine-tuned LoRA adapters to GGUF format")
    
    parser.add_argument("--lora-dir", type=str, required=True,
                        help="Directory containing the fine-tuned LoRA adapters")
    parser.add_argument("--gguf-dir", type=str, default=None,
                        help="Output directory for GGUF files (default: lora_dir/gguf)")
    parser.add_argument("--base-model", type=str, default="unsloth/Mistral-Small-Instruct-2409",
                        help="Base model name or path (default: unsloth/Mistral-Small-Instruct-2409)")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "q2_k"],
                        help="Quantization method to use (default: q4_k_m)")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length (default: 4096)")
    
    args = parser.parse_args()
    
    # Set default GGUF directory if not provided
    if not args.gguf_dir:
        args.gguf_dir = os.path.join(args.lora_dir, "gguf")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.gguf_dir, exist_ok=True)
    
    print(f"Converting LoRA adapters from '{args.lora_dir}' to GGUF format")
    print(f"Base model: {args.base_model}")
    print(f"GGUF output directory: {args.gguf_dir}")
    print(f"Quantization method: {args.quantization}")
    
    try:
        # Load the base model
        print("Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Load the fine-tuned LoRA adapters
        print(f"Loading LoRA adapters from {args.lora_dir}...")
        model = FastLanguageModel.get_peft_model(
            model,
            adapter_path=args.lora_dir,
        )
        
        # Convert to GGUF format
        print(f"Converting to GGUF with {args.quantization} quantization...")
        FastLanguageModel.for_inference(model)
        model.save_pretrained_gguf(
            args.gguf_dir,
            tokenizer,
            quantization_method=args.quantization,
        )
        
        # Create Modelfile for Ollama
        gguf_files = [f for f in os.listdir(args.gguf_dir) if f.endswith('.gguf')]
        if gguf_files:
            modelfile_path = os.path.join(args.gguf_dir, "Modelfile")
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(f"FROM ./{gguf_files[0]}\n")
                f.write('TEMPLATE """{{ .Prompt }}"""\n')
                f.write('PARAMETER stop "<|eot_id|>"\n')
                f.write('PARAMETER context_length 16384\n')
                f.write('PARAMETER temperature 0.7\n')
            print(f"Created Ollama Modelfile at {modelfile_path}")
            
            # Print instructions for using with Ollama
            model_name = os.path.basename(os.path.normpath(args.lora_dir))
            print("\nTo use with Ollama:")
            print(f"cd {args.gguf_dir}")
            print(f"ollama create {model_name.lower().replace('-', '_')} -f Modelfile")
            print(f"ollama run {model_name.lower().replace('-', '_')}")
        
        print("\nConversion completed successfully!")
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nTo successfully convert to GGUF format, you need to have these dependencies installed:")
        print("  - cmake")
        print("  - build-essential (Linux) or equivalent build tools")
        print("\nInstallation commands:")
        print("  Ubuntu/Debian: sudo apt-get install cmake build-essential")
        print("  RHEL/CentOS: sudo yum install cmake gcc-c++")
        print("  macOS: brew install cmake")
        print("  Windows: Install CMake from https://cmake.org/download/")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

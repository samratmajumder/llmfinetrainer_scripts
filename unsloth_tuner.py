import torch
import argparse
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, get_chat_template

# Helper function to check if bfloat16 is supported
def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune a language model with Unsloth")
parser.add_argument("--dataset", type=str, default="training_data.jsonl", 
                    help="Path to JSONL dataset file (default: training_data.jsonl)")
parser.add_argument("--output-dir", type=str, default="./finetuned_mistral",
                    help="Directory for LoRA adapters (default: ./finetuned_mistral)")
parser.add_argument("--gguf-dir", type=str, default="./mistral_gguf",
                    help="Directory for GGUF output (default: ./mistral_gguf)")
parser.add_argument("--max-seq-length", type=int, default=4096,
                    help="Context length for training (default: 4096)")
parser.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs (default: 3)")
parser.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate (default: 2e-5)")
parser.add_argument("--batch-size", type=int, default=2,
                    help="Per device batch size (default: 2)")
parser.add_argument("--format", type=str, choices=["completion", "raw_text"], default="completion",
                    help="Dataset format: 'completion' or 'raw_text' (default: completion)")
parser.add_argument("--skip-gguf", action="store_true",
                    help="Skip GGUF conversion (use if dependencies are missing)")

args = parser.parse_args()

# Configuration
model_name = "unsloth/Mistral-Small-Instruct-2409"  # Pre-quantized 4-bit model
dataset_path = args.dataset
output_dir = args.output_dir
gguf_dir = args.gguf_dir
max_seq_length = args.max_seq_length
quantization_method = "q4_k_m"  # Quantization for Ollama

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect (float16 for older GPUs, bfloat16 for newer)
    load_in_4bit=True,  # Use 4-bit quantization
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank (8, 16, 32 are common choices)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Optimized for no dropout
    bias="none",  # Optimized for no bias
    use_gradient_checkpointing="unsloth",  # Memory-efficient
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load and format the dataset
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Successfully loaded dataset from {dataset_path}")
    print(f"Dataset contains {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure your dataset is properly formatted JSON Lines (.jsonl)")
    print("For raw_text format, each line should contain: {\"text\": \"Your text content here\"}")
    print("You can use clean_dataset.py to fix formatting issues")
    exit(1)

# Check for format and process accordingly
data_format = args.format
print(f"Processing dataset in '{data_format}' format...")

if data_format == "raw_text":
    # Raw text format is already in the right structure for continued pretraining
    if "text" not in dataset.column_names:
        print("ERROR: Raw text format selected but 'text' field not found in dataset")
        print(f"Available fields: {dataset.column_names}")
        print("Please check your dataset format or process it with clean_dataset.py")
        exit(1)
    
    # Print a sample of the raw text to help with debugging
    print("\nSample from dataset (first entry):")
    try:
        sample_text = dataset[0]["text"]
        # Show first 100 characters
        print(f"Text (truncated): {sample_text[:100]}...")
    except Exception as e:
        print(f"Error accessing sample: {e}")
    print()
else:
    # Apply chat template to format the dataset for completion format
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",  # Use Mistral's chat template
        mapping={"role": "user", "content": "prompt", "response": "completion"},
    )

    def format_prompts(examples):
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Format as a conversation
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                tokenize=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    num_train_epochs=args.epochs,
    logging_steps=10,
    save_strategy="epoch",
    fp16=not is_bfloat16_supported(),  # Use fp16 if bfloat16 is not supported
    bf16=is_bfloat16_supported(),  # Use bfloat16 if supported
    max_steps=-1,  # Run for all epochs
)

# Initialize the trainer
# The 'dataset_text_field' and 'packing' parameters are no longer supported in newer versions of SFTTrainer
# Instead, we need to ensure the dataset is properly formatted
if data_format == "raw_text":
    # Format for raw text training
    # We need to properly format the data for SFTTrainer to avoid tokenization issues
    def format_dataset_for_raw_text(examples):
        # Make sure text is properly processed - convert to string, strip any problematic characters
        processed_texts = []
        for text in examples["text"]:
            if text is None:
                processed_texts.append("")
                continue
                
            # Convert to string if not already
            if not isinstance(text, str):
                text = str(text)
                
            # Clean the text - remove any problematic characters
            processed_texts.append(text.strip())
        
        return {"text": processed_texts}
    
    formatted_dataset = dataset.map(format_dataset_for_raw_text, batched=True)
    
    # Now tokenize the formatted dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False)
    
    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        # Remove tokenizer and other unsupported parameters
    )
else:    # Regular instruction format trainer
    # For completion format, tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        # Remove tokenizer and other unsupported parameters
    )

# Start fine-tuning
print(f"Starting fine-tuning with context length of {max_seq_length} tokens...")
print(f"Using batch size: {args.batch_size}, learning rate: {args.lr}, epochs: {args.epochs}")
trainer.train()

# Save LoRA adapters
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA adapters saved to {output_dir}")

# Save to GGUF format for Ollama (if not skipped)
if not args.skip_gguf:
    try:
        print("Converting model to GGUF format... (This requires cmake and build tools)")
        
        # Fix for llama.cpp build issues - manually build llama-quantize
        import os
        import sys
        import subprocess
        
        print("Setting up llama.cpp build environment...")
        
        # Get the Unsloth cache directory where llama.cpp is downloaded
        from pathlib import Path
        import tempfile
        
        # Check if we're on Windows
        is_windows = sys.platform.startswith("win")
        
        # Create a build directory
        llama_cpp_dir = os.path.join(tempfile.gettempdir(), "llama_cpp_build")
        os.makedirs(llama_cpp_dir, exist_ok=True)
        print(f"Using build directory: {llama_cpp_dir}")
        
        # Clone llama.cpp if not already present
        if not os.path.exists(os.path.join(llama_cpp_dir, "CMakeLists.txt")):
            print("Cloning llama.cpp repository...")
            clone_cmd = "git clone https://github.com/ggml-org/llama.cpp.git ."
            subprocess.run(clone_cmd, shell=True, cwd=llama_cpp_dir, check=True)
        
        # Build llama.cpp using CMake
        print("Building llama.cpp (this may take a few minutes)...")
        
        # Create build subdirectory
        build_dir = os.path.join(llama_cpp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Run cmake
        cmake_cmd = "cmake .."
        if is_windows:
            cmake_cmd = "cmake .. -G \"MinGW Makefiles\""
        subprocess.run(cmake_cmd, shell=True, cwd=build_dir, check=True)
        
        # Run make
        make_cmd = "make" if not is_windows else "mingw32-make"
        subprocess.run(make_cmd, shell=True, cwd=build_dir, check=True)
        
        print("llama.cpp build completed successfully.")
        
        # Now continue with the GGUF conversion using Unsloth
        FastLanguageModel.for_inference(model)  # Enable inference mode
        
        # Pass our built llama-quantize path
        quantize_path = os.path.join(build_dir, "bin", "quantize")
        if is_windows:
            quantize_path += ".exe"
        
        if not os.path.exists(quantize_path):
            # Check for newer versions of the binary name
            alt_quantize_path = os.path.join(build_dir, "bin", "llama-quantize")
            if is_windows:
                alt_quantize_path += ".exe"
            
            if os.path.exists(alt_quantize_path):
                quantize_path = alt_quantize_path
            else:
                print(f"Warning: Could not find quantize binary at {quantize_path} or {alt_quantize_path}")
        
        print(f"Using quantize binary at: {quantize_path}")
        
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method=quantization_method,
            llama_cpp_dir=build_dir,  # Use our built version
        )
        print(f"Quantized GGUF model saved to {gguf_dir}")
        
        # Create Modelfile for Ollama
        modelfile_path = os.path.join(gguf_dir, "Modelfile")
        gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith('.gguf')]
        if gguf_files:
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(f"FROM ./{gguf_files[0]}\n")
                f.write('TEMPLATE """{{ .Prompt }}"""\n')
                f.write('PARAMETER stop "<|eot_id|>"\n')
                f.write('PARAMETER context_length 16384\n')
                f.write('PARAMETER temperature 0.7\n')
            print(f"Created Ollama Modelfile at {modelfile_path}")
            
            model_name = os.path.basename(os.path.normpath(output_dir))
            print(f"\nTo use with Ollama:\ncd {gguf_dir}")
            print(f"ollama create {model_name.lower().replace('-', '_')} -f Modelfile")
        
    except Exception as e:
        print(f"GGUF conversion failed: {e}")
        print("\nTo convert to GGUF format, you need to install cmake and build tools:")
        print("  Ubuntu/Debian: sudo apt-get install cmake build-essential git")
        print("  RHEL/CentOS: sudo yum install cmake gcc-c++ git")
        print("  macOS: brew install cmake git")
        print("  Windows: Install CMake, Git, and MinGW from their respective websites")
        print("\nYou can still use the fine-tuned model with the test_finetuned_model.py script,")
        print("or convert it to GGUF later using the convert_to_gguf.py script.")
else:
    print("Skipping GGUF conversion as requested with --skip-gguf flag.")

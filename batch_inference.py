#!/usr/bin/env python3
# batch_inference.py - Run batch inference on a dataset using a fine-tuned model

import os
import argparse
import json
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import GenerationConfig

def main():
    """Run batch inference with a fine-tuned model on a dataset."""
    parser = argparse.ArgumentParser(description="Run batch inference with a fine-tuned model")
    
    parser.add_argument("--lora-dir", type=str, required=True,
                        help="Directory containing the fine-tuned LoRA adapters")
    parser.add_argument("--base-model", type=str, default="unsloth/Mistral-Small-Instruct-2409",
                        help="Base model name or path (default: unsloth/Mistral-Small-Instruct-2409)")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Input JSONL file with prompts")
    parser.add_argument("--output-file", type=str, default="inference_results.jsonl",
                        help="Output JSONL file for results (default: inference_results.jsonl)")
    parser.add_argument("--prompt-field", type=str, default="text",
                        help="Field name in input file containing prompts (default: 'text')")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate (default: 256)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for processing (default: 1)")
    
    args = parser.parse_args()
    
    print(f"Loading base model: {args.base_model}")
    
    # Load the model in 4-bit for faster inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=4096,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=torch.cuda.is_available(),  # Use 4-bit quantization if GPU available
    )
    
    # Load the fine-tuned adapters
    try:
        print(f"Loading LoRA adapters from {args.lora_dir}")
        model = FastLanguageModel.get_peft_model(
            model,
            peft_model_id=args.lora_dir,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"Error loading adapters with Unsloth method: {e}")
        print("Falling back to PEFT loading...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora_dir)
            model = model.merge_and_unload()  # Merge LoRA weights into base model
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            exit(1)
    
    # Load input data
    print(f"Loading prompts from {args.input_file}")
    prompts = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                if args.prompt_field in entry:
                    prompts.append(entry[args.prompt_field])
                else:
                    print(f"Warning: Field '{args.prompt_field}' not found in an entry, skipping")
    except Exception as e:
        print(f"Error loading input file: {e}")
        exit(1)
    
    print(f"Loaded {len(prompts)} prompts for processing")
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Process prompts in batches
    results = []
    
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i+args.batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
        
        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Store results
        for prompt, generated_text in zip(batch_prompts, generated_texts):
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
            })
    
    # Save results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Processed {len(results)} prompts. Results saved to {args.output_file}")
    return 0

if __name__ == "__main__":
    exit(main())

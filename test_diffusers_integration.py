#!/usr/bin/env python3
# test_diffusers_integration.py - Test fine-tuned model with diffusers pipeline

import argparse
import torch
from transformers import pipeline
from diffusers import DiffusionPipeline
from peft import PeftModel

def main():
    """Test fine-tuned model with diffusers integration."""
    parser = argparse.ArgumentParser(description="Test fine-tuned model with diffusers integration")
    
    parser.add_argument("--lora-dir", type=str, required=True,
                        help="Directory containing the fine-tuned LoRA adapters")
    parser.add_argument("--base-model", type=str, default="unsloth/Mistral-Small-Instruct-2409",
                        help="Base model name or path (default: unsloth/Mistral-Small-Instruct-2409)")
    parser.add_argument("--prompt", type=str, default="A mystical forest with ancient trees and a hidden path.",
                        help="Prompt to use for text generation")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading model {args.base_model} with adapters from {args.lora_dir}")
        
        # Method 1: Using text-generation pipeline
        try:
            print("\n=== Method 1: Using Transformers Pipeline ===")
            model = PeftModel.from_pretrained(
                args.base_model,
                args.lora_dir,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Merge the LoRA weights into the base model
            model = model.merge_and_unload()
            
            # Create a text generation pipeline
            text_pipeline = pipeline("text-generation", 
                                    model=model,
                                    tokenizer=args.base_model,
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto")
            
            # Generate text
            result = text_pipeline(args.prompt, 
                                max_new_tokens=256,
                                temperature=0.7,
                                do_sample=True,
                                top_k=50,
                                top_p=0.95)
            
            print(f"\nPrompt: {args.prompt}")
            print(f"Generated text: {result[0]['generated_text']}")
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Try alternative method 2 with Unsloth's API
        try:
            print("\n=== Method 2: Using Unsloth's API ===")
            from unsloth import FastLanguageModel
            
            # Load the model with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.base_model,
                max_seq_length=4096,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                load_in_8bit=torch.cuda.is_available(),
            )
            
            # Load the fine-tuned adapters
            model = FastLanguageModel.get_peft_model(
                model,
                peft_model_id=args.lora_dir,
            )
            
            # Prepare for inference
            FastLanguageModel.for_inference(model)
            
            # Generate text
            inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            print(f"\nPrompt: {args.prompt}")
            print(f"Generated text: {result[0]}")
            
        except Exception as e:
            print(f"Method 2 failed: {e}")
    
    except Exception as e:
        print(f"Error running the script: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

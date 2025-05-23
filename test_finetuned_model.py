#!/usr/bin/env python3
# test_finetuned_model.py - Utility script to test fine-tuned models directly

import os
import argparse
import torch
from unsloth import FastLanguageModel
from transformers import GenerationConfig, TextIteratorStreamer
from threading import Thread

def main():
    """Run inference with a fine-tuned model."""
    parser = argparse.ArgumentParser(description="Test a fine-tuned model with direct inference")
    
    parser.add_argument("--lora-dir", type=str, required=True,
                        help="Directory containing the fine-tuned LoRA adapters")
    parser.add_argument("--base-model", type=str, default="unsloth/Mistral-Small-Instruct-2409",
                        help="Base model name or path (default: unsloth/Mistral-Small-Instruct-2409)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to use for generation (if not provided, will ask for input)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1)")
    
    args = parser.parse_args()
    
    print(f"Loading base model: {args.base_model}")
    
    # Load the model in 8-bit for faster inference
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=8192,  # Use a larger context for inference
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_8bit=True if torch.cuda.is_available() else False,
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Falling back to 4-bit quantization...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=8192,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_4bit=True,
        )
    
    # Load the fine-tuned adapters
    try:
        print(f"Loading LoRA adapters from {args.lora_dir}")
        model = FastLanguageModel.get_peft_model(
            model,
            peft_model_id=args.lora_dir,  # Use peft_model_id instead of adapter_path
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"Error loading LoRA adapters: {e}")
        print("Attempting alternative loading method...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora_dir)
            model = model.merge_and_unload()  # Merge LoRA weights into base model
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            exit(1)
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Function to generate text
    def generate_text(prompt):
        print("\n" + "-"*50)
        print(f"INPUT: {prompt}")
        print("-"*50)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Create a streamer for token-by-token generation
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Start generation in a separate thread
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            generation_config=generation_config
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Print generated text token by token
        print("OUTPUT: ", end="", flush=True)
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        print("\n" + "-"*50)
        return generated_text
    
    # Use provided prompt or enter interactive mode
    if args.prompt:
        generate_text(args.prompt)
    else:
        print("Enter 'exit' to quit.")
        while True:
            try:
                user_input = input("\nPrompt> ")
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                if user_input.strip():
                    generate_text(user_input)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    return 0

if __name__ == "__main__":
    exit(main())

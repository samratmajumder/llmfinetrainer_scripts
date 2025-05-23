import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, get_chat_template

# Configuration
model_name = "unsloth/Mistral-Small-Instruct-2409"  # Pre-quantized 4-bit model
dataset_path = "training_data.jsonl"  # Your JSONL dataset
output_dir = "./finetuned_mistral"  # Directory for LoRA adapters
gguf_dir = "./mistral_gguf"  # Directory for GGUF output
max_seq_length = 2048  # Context length (adjust based on your stories)
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
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Apply chat template to format the dataset
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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=not is_bfloat16_supported(),  # Use fp16 if bfloat16 is not supported
    bf16=is_bfloat16_supported(),  # Use bfloat16 if supported
    max_steps=-1,  # Run for all epochs
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,  # Disable packing for simplicity
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Save LoRA adapters
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA adapters saved to {output_dir}")

# Save to GGUF format for Ollama
FastLanguageModel.for_inference(model)  # Enable inference mode
model.save_pretrained_gguf(
    gguf_dir,
    tokenizer,
    quantization_method=quantization_method,
)
print(f"Quantized GGUF model saved to {gguf_dir}")
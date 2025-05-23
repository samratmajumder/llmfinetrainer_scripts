Run the Fine-Tuning Script
Update the Script:
Replace dataset_path with the path to your training_data.jsonl file (e.g., "C:/Users/You/training_data.jsonl" or "/home/you/training_data.jsonl").

Adjust output_dir and gguf_dir to your desired output directories.

Optionally tweak hyperparameters like r (LoRA rank), learning_rate, or num_train_epochs based on your dataset size and needs.

Execute the Script:
Run in a terminal:
bash

python finetune_mistral.py

The script will:
Load the pre-quantized Mistral Small 3.1 model.

Apply LoRA adapters for efficient fine-tuning.

Format your dataset using Mistral’s chat template.

Train the model for 3 epochs (adjustable).

Save LoRA adapters to output_dir.

Export the model to GGUF format (Q4_K_M) in gguf_dir.

Monitor Training:
Training time depends on your dataset size and GPU. For a small dataset (e.g., 100 stories), expect a few hours on an 18GB VRAM GPU.

Check logs in output_dir for training progress (loss metrics).

Step 4: Deploy the Fine-Tuned Model with Ollama
Create a Modelfile:
In the gguf_dir directory (e.g., ./mistral_gguf), create a file named Modelfile with the following content:

FROM ./Mistral-Small-Instruct-2409-Q4_K_M.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER stop "<|eot_id|>"

This specifies the GGUF file and the prompt template for Ollama. Adjust the filename if Unsloth names it differently (check gguf_dir for the exact GGUF file name).

Create an Ollama Model:
Run the following command in the gguf_dir directory:
bash

ollama create my_mistral_finetuned -f Modelfile

This creates a new Ollama model named my_mistral_finetuned.

Run the Model:
Start the Ollama server (if not already running):
bash

ollama serve

Test the model:
bash

ollama run my_mistral_finetuned

Input a prompt (e.g., the first few sentences of a story) and check if the output matches your desired style.

Example Interaction:
Prompt: “The forest was dark and silent. A lone traveler moved cautiously through the trees.”

The model should generate a continuation in your story’s style, based on the fine-tuning.

Step 5: Evaluate and Refine
Test the Model:
Generate several stories using prompts similar to those in your dataset.

Check if the output captures your style (tone, structure, themes). For example:
bash

curl http://localhost:11434/api/generate -d '{"model": "my_mistral_finetuned", "prompt": "The forest was dark and silent. A lone traveler moved cautiously through the trees."}'

Refine if Needed:
If the output isn’t satisfactory, consider:
Dataset Quality: Ensure your JSONL dataset is consistent and representative of your style. Add more stories or clean existing ones.

Hyperparameters: Increase r (e.g., to 32) or num_train_epochs (e.g., to 5) for better learning, or adjust learning_rate (e.g., to 1e-5).

Prompt Engineering: Test different prompt lengths in the dataset (modify num_sentences in the previous script).

Re-run Fine-Tuning:
Update the script with new parameters or dataset and re-run.

Additional Notes
Why Unsloth?:
Unsloth supports Mistral Small 3.1 with 1.8x faster training and 70% less VRAM compared to standard methods, using QLoRA for efficiency.

It automatically handles chat templates and exports to GGUF format, which is compatible with Ollama.

Quantization Details:
The script uses q4_k_m quantization for a balance of size and quality. Other options include q8_0 (higher quality, larger size) or q2_k (smaller size, lower quality).

The GGUF format ensures compatibility with Ollama and other inference engines like llama.cpp.

Resource Considerations:
Fine-tuning requires 18GB VRAM, but the quantized GGUF model can run on lower-end hardware (e.g., 8GB VRAM or CPU) with Ollama.

If VRAM is limited, consider a cloud service like Google Colab’s paid L4 instance.

Community Support:
Join Unsloth’s Discord or Reddit for help with issues (mentioned in).

Share your progress on X with hashtags like #MistralAI or #UnslothAI, using your handle @sammyblues
, to connect with the community.

Example Output
After fine-tuning, your model should generate stories in your style. For example, if your dataset includes fantasy stories with a dark, mysterious tone, a prompt like “The forest was dark and silent” might produce a continuation like:
The forest was dark and silent. A lone traveler moved cautiously through the trees. A faint whisper echoed from the shadows, calling his name. He gripped his sword, heart pounding, as the mist curled around him like a living thing...

This output should reflect the stylistic nuances (e.g., vivid imagery, suspenseful pacing) learned from your dataset.
Troubleshooting
VRAM Errors:
If you encounter out-of-memory errors, reduce max_seq_length (e.g., to 1024) or per_device_train_batch_size (e.g., to 1).

Poor Output Quality:
Check your dataset for inconsistencies or increase the number of training epochs.

Ollama Issues:
Ensure the GGUF file and Modelfile are correctly formatted. Verify Ollama is running (ollama serve).

Slow Training:
Use a more powerful GPU or reduce dataset size for faster iteration during testing.

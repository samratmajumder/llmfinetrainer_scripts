#!/bin/bash
# filepath: /Users/samrat/Projects/llmfinetuningcollection/test_long_context.sh

# Test the fine-tuned model with a longer context window
# This demonstrates using a 16k context at inference time even though training was with 4k

# Define model name - change this if you used a different name when creating your model
MODEL_NAME="my_mistral_finetuned"

# Check if the model exists in Ollama
if ! ollama list | grep -q "$MODEL_NAME"; then
  echo "Model $MODEL_NAME not found in Ollama. Please create it first."
  exit 1
fi

echo "Testing $MODEL_NAME with a long context prompt..."

# Create a temporary file with a long context prompt
TEMP_FILE=$(mktemp)

cat > "$TEMP_FILE" << 'EOF'
{
  "model": "my_mistral_finetuned",
  "prompt": "You are a story continuation assistant. Continue this story with the same style and tone, writing the next paragraph:

The ancient forest of Eldoria stretched as far as the eye could see, a vast expanse of towering oaks and whisperingeple pines that had stood sentinel for millennia. Shafts of golden sunlight filtered through the dense canopy, creating dappled patterns on the forest floor. The air was thick with the scent of moss and earth, and the distant sound of a bubbling brook added a musical quality to the otherwise silent woods.

Amara had been walking for three days now, following the faint trail that her grandmother had described in her final letter. 'Seek the heart of Eldoria,' the letter had said, 'where the oldest tree reaches for the sky, and there you will find what I have left for you.' The cryptic message was typical of her grandmother—a woman who had always spoken in riddles and believed in the magic of the old world.

Her provisions were running low, and her feet were blistered and sore from the journey. The worn leather map she carried had become increasingly difficult to follow as the paths grew narrower and less distinct. Several times she had nearly turned back, questioning the wisdom of this quest, but something—perhaps her grandmother's stubborn spirit that lived on in her—urged her forward.

As dusk approached on the third day, Amara came upon a small clearing bathed in the last golden rays of sunlight. In the center stood a modest cottage, its stone walls covered in climbing ivy, and its thatched roof sagging slightly with age. Smoke curled from the chimney, suggesting that someone was home, though her grandmother had never mentioned any dwelling in her descriptions of the journey.

Amara hesitated at the edge of the clearing, her hand instinctively moving to the silver pendant at her throat—her grandmother's final gift before she passed. The pendant warmed under her touch, a sensation she had never felt before. Taking a deep breath, she stepped into the clearing and approached the cottage door.",
  "context_length": 16000,
  "temperature": 0.7
}
EOF

# Send the request to Ollama API
curl -s http://localhost:11434/api/generate -d @"$TEMP_FILE" | jq -r '.response' | cat

# Clean up
rm "$TEMP_FILE"
echo -e "\n\nTest completed. The model was instructed to generate a continuation with a 16k context window."

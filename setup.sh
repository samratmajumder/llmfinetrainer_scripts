
python3.10 -m venv venv
source venv/bin/activate
#pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#pip install --no-deps xformers trl peft accelerate bitsandbytes triton torch

# Install dependencies for story processing
pip install -r requirements.txt

# Download spaCy English model for text processing
python -m spacy download en_core_web_sm

echo "Setup completed successfully."
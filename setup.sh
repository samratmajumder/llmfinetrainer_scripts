
python3.10 -m venv venv
source venv/bin/activate

echo "Installing story processing dependencies..."
# Install dependencies for story processing
pip install -r requirements.txt

# Download spaCy English model for text processing
python -m spacy download en_core_web_sm

echo "Do you want to install Unsloth fine-tuning dependencies? (y/n)"
read answer

if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    echo "Installing Unsloth fine-tuning dependencies..."
    # Install dependencies for Unsloth
    pip install -r requirements_unsloth.txt
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
fi

echo "Setup completed successfully."
#!/bin/bash
# filepath: /Users/samrat/Projects/llmfinetuningcollection/setup_unsloth.sh

# Check if we're in a virtual environment, if not, suggest creating one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment detected."
    echo "It's recommended to run this in a virtual environment."
    echo "Do you want to create and activate a new virtual environment? (y/n)"
    read answer
    
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        python3.10 -m venv venv
        source venv/bin/activate
        echo "Virtual environment 'venv' created and activated."
    else
        echo "Proceeding without a virtual environment..."
    fi
fi

# Install dependencies for Unsloth
echo "Installing Unsloth fine-tuning dependencies..."
pip install -r requirements_unsloth.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo "Unsloth setup completed successfully."

#!/bin/bash
# Script to install dependencies required for GGUF conversion

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing build dependencies for Linux..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing build dependencies for macOS..."
    brew install cmake
else
    echo "For Windows, please install CMake manually from https://cmake.org/download/"
    echo "Make sure CMake is added to your PATH"
    echo "You may also need Visual Studio Build Tools"
fi

echo "Dependencies installation complete!"

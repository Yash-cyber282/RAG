#!/bin/bash
# Run this ONCE if you see libtorch_cuda / ncclDevCommDestroy errors.
# It uninstalls the GPU version of torch and installs CPU-only.

echo "Uninstalling GPU torch..."
pip uninstall torch torchvision torchaudio -y

echo "Installing CPU-only torch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Done! Restart your Streamlit app now."

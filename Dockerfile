# LALIC: Linear Attention Modeling for Learned Image Compression
# Docker environment for training and evaluation
#
# Build:   docker build -t lalic .
# Run:     docker run --gpus all -it --rm -v $(pwd):/workspace lalic

# Base image: PyTorch with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt /workspace/

# Install Python dependencies (torch/torchvision are provided by the base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /workspace/

# Set environment variables for CUDA compilation
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV FORCE_CUDA="1"

# Pre-compile the BiWKV CUDA operator (optional, will be compiled on first run if skipped)
# RUN python -c "from models.lalic import load_biwkv4; load_biwkv4()"

# Default command
CMD ["/bin/bash"]

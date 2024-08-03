#!/bin/bash

# Add conda-forge channel and set channel priority
conda config --add channels conda-forge
conda config --set channel_priority flexible

# Create and activate conda environment
conda create -n MPR python=3.9.0 -y
source activate MPR

# Alternatively, use the following to ensure environment activation
# conda create -n MPR python=3.9.0 -y
# conda activate MPR

# Install gcc and g++
conda install -c anaconda gcc gxx_linux-64 -y

# Install CUDA
conda install -c nvidia/label/cuda-11.8.0 cuda -y

# Install PyTorch and dependencies
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y


# Environment variables
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.2;7.5;8.0;8.6+PTX;8.9;9.0"
export GITHUB_ACTIONS=true

# Uninstall incompatible versions of numpy, scipy, pandas, scikit-learn, torch
pip3 uninstall -y numpy scipy pandas scikit-learn torch

# Reinstall necessary Python packages
pip3 install numpy scipy pandas scikit-learn torch

# Install remaining Python dependencies from requirements.txt
pip3 install -r requirements.txt

# Install and reinstall numpy to ensure compatibility
pip3 uninstall numpy gymnasium stable-baselines3 -y
pip3 install numpy==1.26.4 gymnasium==0.29.1 stable-baselines3==2.1.0


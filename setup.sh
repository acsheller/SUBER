#!/bin/bash


conda config --add channels conda-forge
conda config --set channel_priority flexible

# Create and activate conda environment
conda create -n MPR python=3.9.0 -y
source activate MPR

# Install gcc and g++
conda install -c anaconda gcc
conda install -c anaconda gxx_linux-64

# Install CUDA
conda install cuda -c nvidia/label/cuda-11.8.0

# Install PyTorch and dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install or reinstall numpy to ensure compatibility
conda install numpy -y

# Environment variables
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.2;7.5;8.0;8.6+PTX;8.9;9.0"
export GITHUB_ACTIONS=true

# Install remaining Python dependencies from requirements.txt
pip3 install -r requirements.txt
## Need to test the below
pip3 uninstall -y numpy scipy pandas scikit-learn torch

## Need to test the below
pip3 install -y numpy scipy pandas scikit-learn torch

# Need to evalute this
pip3 install git+https://github.com/LAS-NCSU/openai-python

pip3 install --upgrade gymnasium==0.298.1

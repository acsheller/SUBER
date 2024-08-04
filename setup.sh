#!/bin/bash

# Add conda-forge channel and set channel priority
conda config --add channels conda-forge
conda config --set channel_priority flexible

# Create and activate conda environment
conda create -n MPR python=3.9.0 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MPR

# Install gcc and g++
conda install -c anaconda gcc gxx_linux-64 -y

# Install the correct CUDA toolkit version
conda install -c nvidia/label/cuda-11.8.0 cuda -y

# Install PyTorch and dependencies with matching CUDA version
conda install -c pytorch -c nvidia pytorch torchvision torchaudio cudatoolkit=11.8 -y

# Install or reinstall numpy to ensure compatibility
#conda install numpy -y

# Unset system-wide CUDA paths if set and make the settings persistent
CONFIG_FILE=~/.bashrc

function add_if_not_exists {
  grep -qxF "$1" $CONFIG_FILE || echo "$1" >> $CONFIG_FILE
}

add_if_not_exists "unset CUDA_HOME"
add_if_not_exists "unset CUDA_PATH"
add_if_not_exists "unset CUDA_PATH_V11_8"
add_if_not_exists "unset CUDA_PATH_V12_2"
add_if_not_exists 'export PATH=$(echo $PATH | tr ":" "\n" | grep -v "cuda" | tr "\n" ":" | sed "s/:$//")'
add_if_not_exists 'export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | grep -v "cuda" | tr "\n" ":" | sed "s/:$//")'
add_if_not_exists 'export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.2;7.5;8.0;8.6+PTX;8.9;9.0"'
add_if_not_exists "export GITHUB_ACTIONS=true"

# Source the updated bashrc
source $CONFIG_FILE



# Uninstall incompatible versions of numpy, scipy, pandas, scikit-learn, torch
#pip uninstall -y numpy scipy pandas scikit-learn torch

# Reinstall necessary Python packages
#pip install numpy scipy pandas scikit-learn torch

# Install remaining Python dependencies from requirements.txt
pip install -r requirements.txt

# Install gymnasium with a specific version
#pip install --upgrade gymnasium==0.29.1

# Reinstall exllama extension to match the current environment
#pip uninstall -y exllama
#pip install exllama

# Verify installation
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('PyTorch version:', torch.__version__)
"
#!/bin/bash
#SBATCH --job-name=ldw_x_vgg     # Job name
#SBATCH --output=ldw_x_resnet.out  # Output file
#SBATCH --error=ldw_x_resnet.err   # Error file
#SBATCH --ntasks=2                      # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=1000G                      # Memory allocation
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=6-02:00:00               # Time limit (days-HH:MM:SS)

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate ludwig

# Ensure the correct library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATHy

# Reinstall bitsandbytes if necessary
# pip install bitsandbytes --upgrade --no-cache-dir

# Run the Ludwig experiment
python test_ludwig_xvalidation.py ./../../../../tiles_embedding_extraction/vgg11/ludwig_format/tcga_embedding_vgg11_label.csv

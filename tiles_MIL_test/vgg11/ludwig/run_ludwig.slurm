#!/bin/bash
#SBATCH --job-name=ldw_vgg_mil     # Job name
#SBATCH --output=ldw_vgg.out  # Output file
#SBATCH --error=ldw_vgg.err   # Error file
#SBATCH --ntasks=1                      # Number of tasks
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
ludwig experiment --dataset ./../../tiles_MIL_creation/vgg11/tcga_mil_vgg11_ludwig.csv --config ludwig_config.yml --output_directory ldw_exp 

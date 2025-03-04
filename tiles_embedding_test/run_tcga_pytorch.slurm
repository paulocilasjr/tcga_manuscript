#!/bin/bash
#SBATCH --job-name=exp_4                # Job name
#SBATCH --output=ludwig_experiment_4_%j.out  # Output file with job ID
#SBATCH --error=ludwig_experiment_4_%j.err   # Error file with job ID
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=1000G                     # Memory allocation
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

# Dataset path
DATASET_PATH="./../../../trident/TCGA_BRCA/tcga_patches_ludwig.csv"

# Random seed
SEED=42

# Initialize Conda environment
eval "$(conda shell.bash hook)" || {
    echo "Error: Failed to initialize Conda"
    exit 1
}

# Activate the pytorch_env environment
conda activate pytorch_env || {
    echo "Error: Failed to activate pytorch_env"
    exit 1
}

# Ensure the correct library path for GPU usage
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# Verify Python script exists
if [ ! -f "pytorch_test.py" ]; then
    echo "Error: train_tcga_pytorch.py not found in current directory"
    exit 1
fi

# Run the Python script
python pytorch_test.py \
    --dataset "${DATASET_PATH}" \
    --seed ${SEED}

# Exit with success
exit 0

#!/bin/bash
#SBATCH --job-name=pytorch_exp                # Job name
#SBATCH --output=%j.out                 # Output file with job ID
#SBATCH --error=%j.err                  # Error file with job ID
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=1000G                     # Memory allocation
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

# Dataset path
DATASET_PATH="./../../../../../trident/TCGA_BRCA/convert_h5_to_csv/tcga_embeddings_trident_pytorch_labels.csv"

# Python file to run
PYTHON_FILE="pytorch_test_xvalidation.py"

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
if [ ! -f "${PYTHON_FILE}" ]; then
    echo "Error: ${PYTHON_FILE} not found in current directory"
    exit 1
fi

# Run the Python script
python "${PYTHON_FILE}" \
    --dataset "${DATASET_PATH}" \
    --seed ${SEED}

# Exit with success
exit 0

#!/bin/bash
#SBATCH --job-name=tool_embedding               # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=1000G                     # Memory allocation
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

# Dataset path
DATASET_PATH="./../create_zip_tiles/image_tiles.zip"

# Initialize Conda environment
eval "$(conda shell.bash hook)" || {
    echo "Error: Failed to initialize Conda"
    exit 1
}

# Activate the pytorch_env environment
conda activate ludwig || {
    echo "Error: Failed to activate pytorch_env"
    exit 1
}

# Ensure the correct library path for GPU usage
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# Run the Python script
python ./../../galaxy_tools/ludwig_embeddings.py \
    --zip_file "${DATASET_PATH}" \
    --model_name resnet50 \
    --normalize \
    --transform_type rgb \
    --output_csv tcga_embeddings_resnet50.csv \

# Exit with success
exit 0


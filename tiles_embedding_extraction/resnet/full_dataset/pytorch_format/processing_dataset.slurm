#!/bin/bash
#SBATCH --job-name=resnet_proc              # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=5              # Number of CPU cores
#SBATCH --mem=160G                     # Memory allocation
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

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
python ./../../../../utils/processing_tool_output.py --embedding_file ./../tcga_full_embeddings_resnet50_name_mod.csv --metadata_file ./../../../../utils/metadata.csv --output_file tcga_full_embeddings_resnet_name_mod_label.csv --processing_type pytorch_format

# Exit with success
exit 0

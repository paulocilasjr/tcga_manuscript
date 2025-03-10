#!/bin/bash
#SBATCH --job-name=processig_pytorch              # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=1              # Number of CPU cores
#SBATCH --mem=60G                     # Memory allocation
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
python ./../../utils/processing_tool_output.py --embedding_file ./../tcga_embeddings_vgg11.csv --metadata_file ./../../utils/metadata.csv --processing_type ludwig_format --output_file tcga_embedding_vgg11_label.csv --core_column_name vector

# Exit with success
exit 0


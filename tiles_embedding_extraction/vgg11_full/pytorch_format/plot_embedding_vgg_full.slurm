#!/bin/bash
#SBATCH --job-name=vgg_plot              # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=3                      # Number of tasks
#SBATCH --cpus-per-task=3              # Number of CPU cores
#SBATCH --mem=1000G                     # Memory allocation
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
python ./../../../../utils/analyze_embeddings_output.py tcga_embedding_vgg11_label_mod_name.csv 
# Exit with success
exit 0


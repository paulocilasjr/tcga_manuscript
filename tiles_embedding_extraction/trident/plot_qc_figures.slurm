#!/bin/bash
#SBATCH --job-name=vgg_plot              # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=3                      # Number of tasks
#SBATCH --cpus-per-task=3              # Number of CPU cores
#SBATCH --mem=300G                     # Memory allocation
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
python /share/lab_goecks/TCGA_deep_learning/tcga_manuscript/utils/analyze_embeddings_output.py /share/lab_goecks/trident/TCGA_BRCA/convert_h5_to_csv/tcga_embeddings_trident_ludwig_labels.csv
# Exit with success
exit 0


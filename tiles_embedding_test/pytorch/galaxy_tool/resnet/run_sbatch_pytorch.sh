#!/bin/bash

# create_slurm.sh

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_path> <python_file> [seed]"
    echo "Example: $0 /path/to/tcga_patches_ludwig.csv pytorch_test.py 42"
    exit 1
fi

# Assign arguments
DATASET_PATH="$1"
PYTHON_FILE="$2"
SEED="${3:-42}"  # Default seed is 42 if not provided

# SLURM job file name
SLURM_FILE="run_tcga_pytorch.slurm"

# Create the SLURM file with embedded arguments
cat > "${SLURM_FILE}" << EOL
#!/bin/bash
#SBATCH --job-name=pytorch_gt                # Job name
#SBATCH --output=%j.out                 # Output file with job ID
#SBATCH --error=%j.err                  # Error file with job ID
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=1000G                     # Memory allocation
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

# Dataset path
DATASET_PATH="${DATASET_PATH}"

# Python file to run
PYTHON_FILE="${PYTHON_FILE}"

# Random seed
SEED=${SEED}

# Initialize Conda environment
eval "\$(conda shell.bash hook)" || {
    echo "Error: Failed to initialize Conda"
    exit 1
}

# Activate the pytorch_env environment
conda activate pytorch_env || {
    echo "Error: Failed to activate pytorch_env"
    exit 1
}

# Ensure the correct library path for GPU usage
export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}

# Verify Python script exists
if [ ! -f "\${PYTHON_FILE}" ]; then
    echo "Error: \${PYTHON_FILE} not found in current directory"
    exit 1
fi

# Run the Python script
python "\${PYTHON_FILE}" \\
    --dataset "\${DATASET_PATH}" \\
    --seed \${SEED}

# Exit with success
exit 0
EOL

# Make the SLURM file executable
chmod +x "${SLURM_FILE}"

# Submit the SLURM job
sbatch "${SLURM_FILE}"

echo "Submitted SLURM job with dataset: ${DATASET_PATH}, python file: ${PYTHON_FILE}, and seed: ${SEED}"

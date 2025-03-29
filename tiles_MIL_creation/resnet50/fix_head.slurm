#!/bin/bash
#SBATCH --job-name=resnet_mil_T                # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=4                    # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=800G                     # Memory allocation
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate ludwig

# Ensure the correct library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Reinstall bitsandbytes if necessary
# pip install bitsandbytes --upgrade --no-cache-dir


# Run the Python script
python head_inclusion_script.py 
# Exit with success
exit 0

#!/bin/bash
#SBATCH --job-name=uni_mil_tor                # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=3                    # Number of tasks
#SBATCH --cpus-per-task=5              # Number of CPU cores
#SBATCH --mem=1000G                     # Memory allocation
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate ludwig

# Ensure the correct library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Reinstall bitsandbytes if necessary
# pip install bitsandbytes --upgrade --no-cache-dir


# Run the Python script
python ./../../galaxy_tools/ludwig_mil.py --embeddings_csv /share/lab_goecks/trident/TCGA_BRCA/convert_h5_to_csv/tcga_embeddings_trident_pytorch_name_mod.csv --metadata_csv ./../../utils/metadata.csv --dataleak --balance_enforced --bag_size 100 --pooling_method max_pooling --output_csv tcga_mil_univ2_pytorch_format.csv --random_seed 42
# Exit with success
exit 0

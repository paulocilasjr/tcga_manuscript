#!/bin/bash
#SBATCH --job-name=all_bal               # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=32              # Number of CPU cores
#SBATCH --mem=120G                     # Memory allocation
#SBATCH --time=5-02:00:00               # Time limit (days-HH:MM:SS)

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate ludwig

# Ensure the correct library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Reinstall bitsandbytes if necessary
# pip install bitsandbytes --upgrade --no-cache-dir


# Run the Python script
python ./../../galaxy_tools/ludwig_mil_temp_files.py --embeddings_csv ./../../tiles_embedding_extraction/resnet/tcga_embeddings_resnet50_name_mod.csv --metadata_csv ./../../utils/metadata.csv --dataleak --balance_enforced --bag_size 100 --pooling_method max_pooling --output_csv tcga_mil_resnet50_pytorch_all_balanced.csv --by_sample 0,1,2 --imbalance_cap 10

#--imbalance_cap 10
# Exit with success
exit 0

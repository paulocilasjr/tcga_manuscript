#!/bin/bash
#SBATCH --job-name=balanced_vgg                # Job name
#SBATCH --output=%j.out  # Output file with job ID
#SBATCH --error=%j.err   # Error file with job ID
#SBATCH --ntasks=4                    # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
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
python ./../../galaxy_tools/ludwig_mil_temp_files.py --embeddings_csv /share/lab_goecks/TCGA_deep_learning/tcga_manuscript/tiles_embedding_extraction/vgg11/tcga_embeddings_vgg11_name_mod.csv --metadata_csv ./../../utils/metadata.csv --dataleak --balance_enforced --bag_size 100 --pooling_method max_pooling --output_csv tcga_mil_vgg11c_pytorch_by_sample_balanced.csv --by_sample 2 --imbalance_cap 10 
#--imbalance_cap 10

#--truncate_bags

## Exit with success
exit 0

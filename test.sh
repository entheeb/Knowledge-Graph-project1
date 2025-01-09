#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=kg_newdata_test          # Job name
#SBATCH --output=kg_newdata.log             # Standard output log
#SBATCH --time=2-00:00:00                   # Time limit
#SBATCH --partition=accelerated             # Partition (GPU node)
#SBATCH --gres=gpu:1                        # Number of GPUs required
#SBATCH --mem=5G                           # Memory allocation
#SBATCH --cpus-per-task=4                   # Number of CPUs
#SBATCH -A hk-project-pai00011              # Specify the project account

# Source the setup script to initialize the environment
source set_env.sh

# Run the Python script with specified arguments
python test.py --model_dir /hkfs/work/workspace/scratch/st_st177261-Theeb/KGEmb/logs/01_02/FB237/ComplEx_10_47_01
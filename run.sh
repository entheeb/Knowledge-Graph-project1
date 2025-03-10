#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=new_method        # Job name
#SBATCH --output=new_methodztotal_lhs.log        # Standard output log
#SBATCH --time=1-00:00:00                   # Time limit
#SBATCH --partition=accelerated             # Partition (GPU node)
#SBATCH --gres=gpu:1                        # Number of GPUs required
#SBATCH --mem=8G                           # Memory allocation
#SBATCH --cpus-per-task=4                   # Number of CPUs
#SBATCH -A hk-project-pai00011              # Specify the project account

# Source the setup script to initialize the environment
source set_env.sh

# Run the Python script with specified arguments
python new_method.py \
            --dataset FB237


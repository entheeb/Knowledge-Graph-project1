#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=kg_newdata_test          # Job name
#SBATCH --output=job_logs/newdata_ood_easy_balanced.log             # Standard output log
#SBATCH --time=1-00:00:00                   # Time limit
#SBATCH --partition=accelerated             # Partition (GPU node)
#SBATCH --gres=gpu:1                        # Number of GPUs required
#SBATCH --mem=5G                           # Memory allocation
#SBATCH --cpus-per-task=4                   # Number of CPUs
#SBATCH -A hk-project-pai00011              # Specify the project account

# Source the setup script to initialize the environment
source set_env.sh

# Run the Python script with specified arguments
python test.py --model_dir logs/04_01/NELL-995-h50/ComplEx_01_31_46

python test.py --model_dir logs/04_01/NELL-995-h50/ComplEx_01_47_27

python test.py --model_dir logs/04_01/NELL-995-h50/RotE_00_06_24

python test.py --model_dir logs/04_01/NELL-995-h50/RotE_00_22_37

python test.py --model_dir logs/04_01/NELL-995-h50/RotH_02_11_09

python test.py --model_dir logs/04_01/NELL-995-h50/RotH_02_35_12
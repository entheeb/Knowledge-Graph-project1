#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=kg_newdata_test          # Job name
#SBATCH --output=job_logs/newdata_ood_easy.log             # Standard output log
#SBATCH --time=1-00:00:00                   # Time limit
#SBATCH --partition=accelerated             # Partition (GPU node)
#SBATCH --gres=gpu:1                        # Number of GPUs required
#SBATCH --mem=5G                           # Memory allocation
#SBATCH --cpus-per-task=4                   # Number of CPUs
#SBATCH -A hk-project-pai00011              # Specify the project account

# Source the setup script to initialize the environment
source set_env.sh

# Run the Python script with specified arguments
python test.py --model_dir logs/03_20/FB237/RotH_16_34_01

python test.py --model_dir logs/03_21/FB237/RotH_12_33_13

python test.py --model_dir logs/03_21/FB237/ComplEx_17_21_01

python test.py --model_dir logs/03_21/FB237/ComplEx_17_39_25

python test.py --model_dir logs/03_20/WN18RR/ComplEx_17_46_33

python test.py --model_dir logs/03_20/WN18RR/RotE_17_54_23

python test.py --model_dir logs/03_20/WN18RR/RotE_18_14_01

python test.py --model_dir logs/03_20/WN18RR/RotH_19_46_17

python test.py --model_dir logs/03_20/WN18RR/RotH_20_07_59

python test.py --model_dir logs/03_22/ICEWS18R/ComplEx_03_06_25

python test.py --model_dir logs/03_22/ICEWS18R/ComplEx_03_47_45

python test.py --model_dir logs/03_22/ICEWS18R/RotE_04_39_05

python test.py --model_dir logs/03_22/ICEWS18R/RotE_06_45_33

python test.py --model_dir logs/03_22/ICEWS18R/RotH_16_06_02

python test.py --model_dir logs/03_22/ICEWS18R/RotH_18_37_28


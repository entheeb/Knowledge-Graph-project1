#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=kg_newdata_test          # Job name
#SBATCH --output=job_logs/kg_newdata.log             # Standard output log
#SBATCH --time=2-00:00:00                   # Time limit
#SBATCH --partition=accelerated             # Partition (GPU node)
#SBATCH --gres=gpu:1                        # Number of GPUs required
#SBATCH --mem=5G                           # Memory allocation
#SBATCH --cpus-per-task=4                   # Number of CPUs
#SBATCH -A hk-project-pai00011              # Specify the project account

# Source the setup script to initialize the environment
source set_env.sh

# Run the Python script with specified arguments
python test.py --model_dir logs/02_18/ICEWS18T/RotH_04_04_00

python test.py --model_dir logs/02_18/ICEWS18T/RotH_05_33_38

python test.py --model_dir logs/02_17/ICEWS18T/ComplEx_19_47_05

python test.py --model_dir logs/02_17/ICEWS18T/ComplEx_20_34_14

python test.py --model_dir logs/02_17/ICEWS18T/RotE_21_29_53

python test.py --model_dir logs/02_17/ICEWS18T/RotE_22_42_52

python test.py --model_dir logs/02_18/ICEWS18R/RotH_00_38_12

python test.py --model_dir logs/02_18/ICEWS18R/RotH_03_12_03

python test.py --model_dir logs/02_17/ICEWS18R/ComplEx_11_36_30

python test.py --model_dir logs/02_17/ICEWS18R/ComplEx_12_23_59

python test.py --model_dir logs/02_17/ICEWS18R/RotE_13_21_12

python test.py --model_dir logs/02_17/ICEWS18R/RotE_14_51_06


#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=kg_embedding_complex        # Job name
#SBATCH --output=kg_complex.log        # Standard output log
#SBATCH --time=2-00:00:00                   # Time limit
#SBATCH --partition=accelerated             # Partition (GPU node)
#SBATCH --gres=gpu:1                        # Number of GPUs required
#SBATCH --mem=20G                           # Memory allocation
#SBATCH --cpus-per-task=4                   # Number of CPUs
#SBATCH -A hk-project-pai00011              # Specify the project account

# Source the setup script to initialize the environment
source set_env.sh

# Run the Python script with specified arguments
python run.py \
            --dataset FB237 \
            --model ComplEx \
            --rank 500 \
            --regularizer N3 \
            --reg 0.1 \
            --optimizer Adagrad \
            --max_epochs 300 \
            --patience 15 \
            --valid 5 \
            --batch_size 500 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias none \
            --dtype single 

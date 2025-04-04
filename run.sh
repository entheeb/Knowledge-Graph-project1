#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=new_dataset1        # Job name
#SBATCH --output=job_logs/old_NELL-995-h50.log        # Standard output log
#SBATCH --time=1-00:00:00                   # Time limit
#SBATCH --partition=accelerated             # Partition (GPU node)
#SBATCH --gres=gpu:1                        # Number of GPUs required
#SBATCH --mem=8G                           # Memory allocation
#SBATCH --cpus-per-task=4                   # Number of CPUs
#SBATCH -A hk-project-pai00011              # Specify the project account

# Source the setup script to initialize the environment
source set_env.sh

# Run the Python script with specified arguments
python run.py \
            --dataset NELL-995-h50 \
            --model RotE \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 200 \
            --patience 15 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size 50 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --double_neg 

python run.py \
            --dataset NELL-995-h50 \
            --model RotE \
            --rank 500 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 200 \
            --patience 15 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size 250 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg


python run.py \
            --dataset NELL-995-h50 \
            --model ComplEx \
            --rank 32 \
            --regularizer N3 \
            --reg 0.05 \
            --optimizer Adagrad \
            --max_epochs 200 \
            --patience 15 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias none \
            --dtype single 

python run.py \
            --dataset NELL-995-h50 \
            --model ComplEx \
            --rank 500 \
            --regularizer N3 \
            --reg 0.1 \
            --optimizer Adagrad \
            --max_epochs 200 \
            --patience 15 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias none \
            --dtype single 


python run.py \
            --dataset NELL-995-h50 \
            --model RotH \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 300 \
            --patience 15 \
            --valid 5 \
            --batch_size 500 \
            --neg_sample_size 50 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c 

python run.py \
            --dataset NELL-995-h50 \
            --model RotH \
            --rank 500 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 300 \
            --patience 15 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size 50 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c


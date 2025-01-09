#!/bin/bash

# Set paths
KGHOME=$(pwd)
export PYTHONPATH="$KGHOME:$PYTHONPATH"
export LOG_DIR="$KGHOME/logs"
export DATA_PATH="$KGHOME/data"

# Activate Conda environment
source /hkfs/work/workspace/scratch/st_st177261-Theeb/miniconda3/bin/activate  # Adjust this to the path of your Conda installation
conda activate Theeb1




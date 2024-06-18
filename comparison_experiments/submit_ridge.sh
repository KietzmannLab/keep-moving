#!/bin/bash
#SBATCH --job-name=ridge
#SBATCH --time 02:00:00
#SBATCH --mem=30G
#SBATCH --partition=klab-cpu
#SBATCH -c 8

source ~/setup_H100.sh

# pass all cmd args
~/miniconda3/envs/avalanche-h100/bin/python ridge_analysis.py "$@"


#!/bin/bash
#SBATCH --job-name=gradient_decomposition
#SBATCH --time 00:10:00
#SBATCH -c 2
#SBATCH --mem 2G
#SBATCH -p workq

source /home/staff/d/danthes/setup_H100.sh
/home/staff/d/danthes/miniconda3/envs/avalanche-h100/bin/python run_linear_ewc.py "$@"
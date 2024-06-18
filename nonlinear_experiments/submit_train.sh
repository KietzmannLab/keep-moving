#!/bin/bash
#SBATCH -p klab-gpu
#SBATCH -c 9
#SBATCH --mem 64G
#SBATCH --gres=gpu:H100.20gb:1
#SBATCH --time=12:00:00

export TMPDIR='./tmp'
python train.py "$@"

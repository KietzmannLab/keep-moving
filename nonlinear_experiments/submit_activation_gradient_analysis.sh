#!/bin/bash
#SBATCH -p klab-cpu
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH --time=00:30:00

export TMPDIR='./tmp'
python analyse_gradient_null.py "$@"

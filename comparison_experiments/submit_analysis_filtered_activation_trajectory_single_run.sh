#!/bin/bash
#SBATCH -p klab-cpu
#SBATCH --mem=2G
#SBATCH -c 1
#SBATCH --time=00:05:00

# first argument is config file
file=$1
echo "Config file: $file"
# get slurm array index

index=$SLURM_ARRAY_TASK_ID
echo "Index: $index"
# get index row from file
line=$(sed "${index}q;d" $file)

echo "Running with opt. $line"
source ~/setup_H100.sh

# grab config from list of configs and submit
~/miniconda3/envs/avalanche-h100/bin/python analysis_filtered_activation_trajectory_single_run.py $line

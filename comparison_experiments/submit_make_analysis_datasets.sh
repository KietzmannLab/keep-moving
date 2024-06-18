#!/bin/bash
#SBATCH --job-name=mkdata
#SBATCH --time 00:30:00
#SBATCH --mem=30G
#SBATCH --partition=klab-cpu
#SBATCH -c 8

file=$1
echo "Config file: $file"

index=$SLURM_ARRAY_TASK_ID
echo "Index: $index"
line=$(sed "${index}q;d" $file)

echo "Running with opt. $line"

source ~/setup_H100.sh

# pass all cmd args
~/miniconda3/envs/avalanche-h100/bin/python create_analysis_dataset_for_run.py $line

exit_code=$?

if [ $exit_code -eq 0 ]
then
    echo "Job finished successfully"
else
    echo "Job failed with exit code $exit_code"
    touch job_failed_$SLURM_JOB_ID_opt_$line
fi
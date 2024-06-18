#!/bin/bash
projdir=$1
echo "submitting analysis for project $projdir"

# list all folders in projdir
folders=$(ls $projdir)

for folder in $folders
do
    echo "submitting analysis for folder $folder"
    sbatch submit_analysis.sh --rundir $projdir/$folder
done

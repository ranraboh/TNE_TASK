#!/bin/bash

#SBATCH --job-name=tne_ran
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1g
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=studentbatch
#SBATCH --error=tne_ran.err
#SBATCH --output=tne_ran.out
#SBATCH --time=10:00:00

OUTPUT_PATH='./out/tne.out'
ERR_PATH='./out/tne.err'
rm -rf $OUTPUT_PATH $ERR_PATH

python3 tne.py
mv tne_ran.out $OUTPUT_PATH
mv tne_ran.err $ERR_PATH
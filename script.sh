#!/bin/bash

#SBATCH --job-name=PES_DL   		## Name of the job
#SBATCH --output=PES_DL_Output.out  	## Output file
#SBATCH --time=24:00:00           		## Job Duration
#SBATCH --ntasks=4            		## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=4    		## The number of threads the code will use
#SBATCH --mem=8G              		## Memory required per node
#SBATCH --nodes=4                       ##nodes requested
#SBATCH --partition=gpu          		## Memory required per node
#SBATCH --gpus-per-node=V100-SXM2-32GB:1

## Load the python interpreter
module load miniconda
module load cuda-toolkit/12.8

conda init zsh
conda activate pes

# pip install -e.

python src/pes_1D/mygan.py

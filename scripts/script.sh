#!/bin/bash

#SBATCH --job-name=CamelCase   		## Name of the job
#SBATCH --output=CamelCase.out  	 ## Output file
#SBATCH --time=1:00           		## Job Duration
#SBATCH --ntasks=1             		## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=1      		## The number of threads the code will use
#SBATCH --mem-per-cpu=100      		## Real memory(MB) per CPU required by the job.

## Load the python interpreter
module load conda
conda activate pes

## Execute the python script
python script_test.py

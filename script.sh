#!/bin/bash

#SBATCH --job-name=PES_DL   		## Name of the job
#SBATCH --output=PES_DL_Output.out  	## Output file
#SBATCH --time=01:00:00           		## Job Duration
##SBATCH --ntasks=2             		## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=2      		## The number of threads the code will use
#SBATCH --mem=4G              		## Memory required per node
#SBATCH --nodes=2                       #nodes requested
#SBATCH --partition=gpu            		## Memory required per node
## Load the python interpreter
module load miniconda

conda init zsh

conda activate pes

##pip install -e.

python scripts/script_test.py
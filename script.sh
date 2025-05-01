#!/bin/bash

#SBATCH --job-name=PES_DL   		## Name of the job
#SBATCH --output=PES_DL_Output.out  	## Output file
#SBATCH --time=04:00:00           		## Job Duration
##SBATCH --ntasks=12             		## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=12      		## The number of threads the code will use
#SBATCH --mem=8G              		## Memory required per node
#SBATCH --nodes=12                       #nodes requested
##SBATCH --partition=gpu            		## Memory required per node


## Load the python interpreter
module load miniconda

conda init zsh

conda activate pes

##pip install -e.

python src/pes_1D/mygan.py
#!/bin/bash

#SBATCH --job-name=1gen_vs_spline   		    ## Name of the job
#SBATCH --output=./results/gen_vs_spline_1.out  ## Output file
#SBATCH --time=24:00:00           		        ## Job Duration
#SBATCH --ntasks=8            		            ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8    		            ## The number of threads the code will use
#SBATCH --mem=12G              		            ## Memory required
#SBATCH --nodes=2                               ##nodes requested
#SBATCH --partition=gpu          		        ## Memory required per node
#SBATCH --gpus-per-node=V100-SXM2-32GB:2

## Load the python interpreter
module load miniconda
module load cuda-toolkit/12.8

conda init zsh
conda activate pes

# pip install -e.

python src/pes_1D/mygan.py [4,5] 1

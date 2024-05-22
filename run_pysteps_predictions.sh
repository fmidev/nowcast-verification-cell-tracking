#!/bin/bash
#SBATCH --partition=pp-long
#SBATCH --job-name=sprog_preds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=84G
#SBATCH --time=120:00:00
#SBATCH --output=output/output_%x_%j.txt
#SBATCH --error=output/errors_%x_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jenna.ritvanen@fmi.fi

source /users/${USER}/.bashrc

conda activate gpu

export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=1

# python scripts/run_pysteps_swap_predictions.py swiss-data/pysteps-predictions-extrap
# python scripts/run_pysteps_swap_predictions.py swiss-data/pysteps-predictions-linda
python scripts/run_pysteps_swap_predictions.py swiss-data/pysteps-predictions-sprog
#!/bin/bash
#SBATCH --partition=postproc
#SBATCH --job-name=measurements
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=84G
#SBATCH --time=24:00:00
#SBATCH --output=output/output_%x_%j.txt
#SBATCH --error=output/errors_%x_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jenna.ritvanen@fmi.fi

source /users/${USER}/.bashrc

conda activate gpu

export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=1

python scripts/save_measurements.py config/swiss-data/save_measurements.yaml

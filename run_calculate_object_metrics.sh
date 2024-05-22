#!/bin/bash
#SBATCH --partition=pp-long
#SBATCH --job-name=obj_metrics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=160G
#SBATCH --time=120:00:00
#SBATCH --output=output/output_%x_%j.txt
#SBATCH --error=output/errors_%x_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jenna.ritvanen@fmi.fi

source /users/${USER}/.bashrc

conda activate gpu

export PYTHONPATH=$PYTHONPATH:.



python scripts/calculate_metrics.py config/swiss-data/calculate_metrics_objects.yaml
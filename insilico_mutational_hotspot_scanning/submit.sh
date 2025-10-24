#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --job-name=scanning_dataset_generation
#SBATCH --output=insilico_mutational_hotspot_scanning/log/scanning_dataset_generation_%j.out
#SBATCH --error=insilico_mutational_hotspot_scanning/log/scanning_dataset_generation_%j.err
#SBATCH --time=66:00:00
#SBATCH --partition=gpu11,sugon,hygon,cpu2


python3 insilico_mutational_hotspot_scanning/02_scanning_dataset_generation.py

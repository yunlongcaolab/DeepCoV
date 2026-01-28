#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=topk_1_all
#SBATCH --output=benchmark/MLR/analysis/log/topk_1.out
#SBATCH --error=benchmark/MLR/analysis/log/topk_1.err
#SBATCH --partition=gpu11

# conda activate r4

# Rscript benchmark/MLR/analysis/dynamic_topk_calc.R 3
Rscript benchmark/MLR/analysis/dynamic_topk_calc.R 1
# Rscript benchmark/MLR/analysis/dynamic_topk_calc.R 5 
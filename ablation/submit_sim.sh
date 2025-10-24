#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=top5
#SBATCH --output=ablation/log/top5.out
#SBATCH --error=ablation/log/top5.err
#SBATCH --time=66:00:00
#SBATCH --partition=sugon


# Rscript ablation/ablation_dynamic_topk_calc.R 1
# Rscript ablation/ablation_dynamic_topk_calc.R 3
Rscript ablation/ablation_dynamic_topk_calc.R 5

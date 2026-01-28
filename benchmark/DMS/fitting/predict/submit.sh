#!/bin/bash
#SBATCH --job-name=dms_model        
#SBATCH --output=/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/log/Q_e_test_KP3ref_%j.out        
#SBATCH --error=/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/log/Q_e_test_KP3ref_%j.err           
#SBATCH --partition=sugon    
#SBATCH --ntasks=1     
#SBATCH --cpus-per-task=64
#SBATCH --time=30-00:00:00

# source /lustre/grp/cyllab/yangsj/miniconda/etc/profile.d/conda.sh
# conda init
# conda activate dl

# nvidia-smi


python /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/predict/predict_test_JN1era.py
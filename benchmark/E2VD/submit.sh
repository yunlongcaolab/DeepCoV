#!/bin/bash
#SBATCH --job-name=E2VD         
#SBATCH --output=/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/log/expr_test.out        
#SBATCH --error=/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/log/expr_test.err           
#SBATCH --partition=gpu42,gpu52
#SBATCH --nodes=2               
#SBATCH --gres=gpu:8
#SBATCH --time=30-00:00:00

source /lustre/grp/cyllab/yangsj/miniconda/etc/profile.d/conda.sh
conda init
conda activate dl

# nvidia-smi

# NNODES=1
# NODE_RANK=0
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# MIN_PORT=8000
# MAX_PORT=9000
# export MASTER_PORT=$(( RANDOM % ($MAX_PORT - $MIN_PORT + 1) + $MIN_PORT ))
# echo "MASTER_ADDR: $MASTER_ADDR" 
# echo "MASTER_PORT: $MASTER_PORT" 


# python /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/scripts/bind_data.py
# python /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/scripts/bind_test.py
# python /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/scripts/escape_data.py
# python /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/scripts/escape_test.py
# python /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/scripts/expr_data.py
python /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/scripts/expr_test.py
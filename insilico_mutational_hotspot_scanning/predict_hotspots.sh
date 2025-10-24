#!/bin/bash
#SBATCH --job-name=scan_XBB
#SBATCH --output=prediction/log/scan_XBBera_XBB_new-%j.out        
#SBATCH --error=prediction/log/scan_XBBera_XBB_new-%j.err           
#SBATCH --partition=gpu52
#SBATCH --nodes=1              
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00  

RUN_NAME=rbd_single_XBBera
MODE=run_validation
CONFIG='/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_XBBera_XBBmutscan.yaml'

NNODES=1
NODE_RANK=0
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MIN_PORT=8000
MAX_PORT=9000
export MASTER_PORT=$(( RANDOM % ($MAX_PORT - $MIN_PORT + 1) + $MIN_PORT ))
echo "MASTER_ADDR: $MASTER_ADDR" 
echo "MASTER_PORT: $MASTER_PORT" 


echo $MODE" mode..."
torchrun --nnodes $NNODES \
        --nproc_per_node $SLURM_GPUS_ON_NODE \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
    DepCoV/main_hotspots_predict.py --run_name $RUN_NAME --mode $MODE --config $CONFIG

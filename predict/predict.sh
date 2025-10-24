#!/bin/bash
#SBATCH --job-name=JN1_test
#SBATCH --output=/lustre/grp/cyllab/share/evolution_prediction_dl/predict/log/rbd_JN1era_LocDiffMajor_updateModel-%j.out        
#SBATCH --error=/lustre/grp/cyllab/share/evolution_prediction_dl/predict/log/rbd_JN1era_LocDiffMajor_updateModel-%j.err           
#SBATCH --partition=gpu11
#SBATCH --nodes=1              
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00  

cd /lustre/grp/cyllab/share/evolution_prediction_dl

source /lustre/grp/cyllab/luoxw/anaconda3/etc/profile.d/conda.sh
conda init
conda activate evolutionpredict
conda env list

RUN_NAME=rbd_single_JN1era
MODE=run_validation # run_train
CONFIG=/lustre/grp/cyllab/share/evolution_prediction_dl/predict/config/config_JN1era_TestFull.yaml

NNODES=1
NODE_RANK=0
# MASTER_ADDR=c06b09n05
# MASTER_PORT=32579
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
    DepCoV/main_proportion_predict.py --run_name $RUN_NAME --mode $MODE --config $CONFIG







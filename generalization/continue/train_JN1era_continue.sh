#!/bin/bash
#SBATCH --job-name=cont_test
#SBATCH --output=/lustre/grp/cyllab/yangsj/evo_pred/0article/generalization/JN1era_continuous/log/rbd_JN1era_continue_train_3LSTMoutlayer_repeat-%j.out        
#SBATCH --error=/lustre/grp/cyllab/yangsj/evo_pred/0article/generalization/JN1era_continuous/log/rbd_JN1era_continue_train_3LSTMoutlayer_repeat-%j.err           
#SBATCH --partition=gpu42
#SBATCH --nodes=1              
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00  

cd /lustre/grp/cyllab/yangsj/evo_pred/1article

source /lustre/grp/cyllab/luoxw/anaconda3/etc/profile.d/conda.sh
conda init
conda activate evolutionpredict
conda env list


# RUN_NAME=JN1era_continue_train_1LSTMoutlayer
# MODE=run_train
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/1article/generalization/JN1_era_continuous/config/config_JN1era_continue_train_1LSTMoutlayer.yaml

# RUN_NAME=JN1era_continue_train_3LSTMoutlayer
RUN_NAME=JN1era_continue_train_3LSTMoutlayer_repeat
MODE=run_train
CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/1article/generalization/JN1_era_continuous/config/config_JN1era_continue_train_3LSTMoutlayer.yaml

# RUN_NAME=JN1era_continue_train_4LSTMoutlayer
# MODE=run_train
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/1article/generalization/JN1_era_continuous/config/config_JN1era_continue_train_4LSTMoutlayer.yaml

# RUN_NAME=JN1era_continue_train_2evopred
# MODE=run_train
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/1article/generalization/JN1_era_continuous/config/config_JN1era_continue_train_2evopred.yaml

# RUN_NAME=JN1era_continue_train_bg360
# MODE=run_train
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/1article/generalization/JN1_era_continuous/config/config_JN1era_continue_train_bg360.yaml

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
    DepCoV/main_proportion_continue_predict.py --run_name $RUN_NAME --mode $MODE --config $CONFIG







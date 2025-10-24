#!/bin/bash
#SBATCH --job-name=no_bgratios
#SBATCH --output=ablation/log/no_backgrounds_TestFull-%j.out        
#SBATCH --error=ablation/log/no_backgrounds_TestFull-%j.err           
#SBATCH --partition=gpu52
#SBATCH --nodelist=c06b06n[01-02]
#SBATCH --nodes=1              
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00  

cd /lustre/grp/cyllab/share/evolution_prediction_dl

# RUN_NAME=rbd_single_JN1era_esm2
# MODE=run_validation
# CONFIG=ablation/config/config_module_ablation_esm2.yaml

# RUN_NAME=rbd_single_JN1era_no_bgratios
# MODE=run_validation
# CONFIG=ablation/config/config_module_ablation_nobgratios.yaml

RUN_NAME=rbd_single_JN1era_no_backgrounds
MODE=run_validation
CONFIG=config/config_module_ablation_nobackgrounds.yaml

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
    DepCoV/main_proportion_predict.py --run_name $RUN_NAME --mode $MODE --config $CONFIG







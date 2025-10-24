#!/bin/bash
#SBATCH --job-name=scan_XBB
#SBATCH --output=/lustre/grp/cyllab/yangsj/evo_pred/0article/prediction/log/scan_XBBera_XBB_new-%j.out        
#SBATCH --error=/lustre/grp/cyllab/yangsj/evo_pred/0article/prediction/log/scan_XBBera_XBB_new-%j.err           
#SBATCH --partition=gpu52
#SBATCH --nodes=1              
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00  

cd /lustre/grp/cyllab/yangsj/evo_pred/1article

source /lustre/grp/cyllab/luoxw/anaconda3/etc/profile.d/conda.sh
conda init
conda activate evolutionpredict
conda env list
# source /lustre/grp/cyllab/yangsj/miniconda/etc/profile.d/conda.sh

# RUN_NAME=rbd_single_JN1era
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan.yaml

# RUN_NAME=rbd_single_JN1era_updateModel
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_updateModel.yaml

# RUN_NAME=rbd_single_JN1era_update
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_update_KP3mutscan.yaml

# RUN_NAME=rbd_single_JN1era_update
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_update_LF7mutscan.yaml

# RUN_NAME=spike_single_JN1era
# MODE=run_validation
# # CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_KP3mutscan_NTD.yaml
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_KP2mutscan_NTD.yaml

# RUN_NAME=rbd_single_XBBera
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_XBBera_XBBmutscan.yaml
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_XBBera_XBB15mutscan.yaml
RUN_NAME=rbd_single_XBBera_correctcount
MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_XBBera_XBB15mutscan_correctcount.yaml
# CONFIG='/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_XBBera_EG5mutscan_correctcount.yaml'
CONFIG='/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_XBBera_XBBmutscan_correctcount.yaml'


 RUN_NAME=rbd_single_JN1era_JN1mutscan_long_esm2
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_long_esm2.yaml

# RUN_NAME=rbd_single_JN1era_JN1mutscan_long_keep_all
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_long_keep_all.yaml

# RUN_NAME=rbd_single_JN1era_JN1mutscan_long_no_bgratios
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_long_no_bgratios.yaml

# RUN_NAME=rbd_single_JN1era_JN1mutscan_long_no_dms_all
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_long_no_dms_all.yaml

RUN_NAME=rbd_single_JN1era_JN1mutscan_long_no_backgrounds
MODE=run_validation
CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_long_no_backgrounds.yaml


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
    DepCoV/main_hotspots_predict.py --run_name $RUN_NAME --mode $MODE --config $CONFIG





# RUN_NAME=rbd_single_JN1era_JN1mutscan_esm2
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_esm2.yaml

# RUN_NAME=rbd_single_JN1era_JN1mutscan_keep_all
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_keep_all.yaml

# RUN_NAME=rbd_single_JN1era_JN1mutscan_no_backgrounds
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_no_backgrounds.yaml

# RUN_NAME=rbd_single_JN1era_JN1mutscan_no_bgratios
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_no_bgratios.yaml

# RUN_NAME=rbd_single_JN1era_JN1mutscan_no_dms_all_0
# RUN_NAME=rbd_single_JN1era_JN1mutscan_no_dms_all
# MODE=run_validation
# CONFIG=/lustre/grp/cyllab/yangsj/evo_pred/0article/insilico_mutational_hotspot_scanning/config/config_JN1era_JN1mutscan_no_dms_all.yaml

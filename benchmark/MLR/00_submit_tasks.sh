#!/bin/bash
#SBATCH -J mlr_job
#SBATCH -o /lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/MLR/log2/mlr_%A_%a.out
#SBATCH -e /lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/MLR/log2/mlr_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=gpu11
#SBATCH --time=02:00:00
#SBATCH --array=1-400%112  # 注意：这里从1开始，因为CSV第一行是表头

conda activate deepcov
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
cd /lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/MLR

TASK_FILE="/lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/MLR/res/task_list_TestFull_2023-10-01_to241030.csv"
OUT_DIR="/lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/MLR/res/TestFull_2023-10-01_to241030"
COUNT_FILE="/lustre/grp/cyllab/share/evolution_prediction_dl/data/processed/to241030/rbd/rbd_count_smooth.npz"

### for updated dataset
# TASK_FILE="/lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/MLR/res/task_list_TestFull_2023-10-01_to250516.csv"
# OUT_DIR="/lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/MLR/res/TestFull_2023-10-01_to250516"
# COUNT_FILE="/lustre/grp/cyllab/share/evolution_prediction_dl/data/processed/to250516/rbd/rbd_count_smooth.npz"

mkdir -p $OUT_DIR

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $TASK_FILE)
IFS=',' read -r LOC T0 BG <<< "$LINE"

# 5. 执行 Python 并传入外置参数
python 02_run_single_MLR.py \
    --location "$LOC" \
    --t0 "$T0" \
    --n_bg_clusters "$BG" \
    --counts "$COUNT_FILE" \
    --outdir "$OUT_DIR"
    
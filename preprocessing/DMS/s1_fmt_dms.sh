#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=s1
#SBATCH --output=/lustre/grp/cyllab/share/ljj/public/dms1030_log/s1.out
#SBATCH --error=/lustre/grp/cyllab/share/ljj/public/dms1030_log/s1.err
#SBATCH --time=66:00:00
#SBATCH --partition=sugon,hygon,cpu1,gpu11

dms_data_dir="/lustre/grp/cyllab/share/ljj/public/dms_data"
out_dir="/lustre/grp/cyllab/share/ljj/public/dms1030/merge"
ref_dir="/lustre/grp/cyllab/share/ljj/public/spike_ref"

/lustre/grp/cyllab/luojj/anaconda3/envs/opencpd/bin/python3 s1_fmt_dms.py \
--public_dms_dir "${dms_data_dir}/origin_public_DMS" \
--lab_dms_dir "${dms_data_dir}/DMS_20240717" \
--ref_aln "${ref_dir}/isolates_manual_1030.aln" \
--isolate_time_csv "${ref_dir}/isolate_time_table.csv" \
--out_dir ${out_dir}



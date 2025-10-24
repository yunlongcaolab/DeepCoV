#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=s2
#SBATCH --output=/lustre/grp/cyllab/share/ljj/public/dms1030_log/s2.out
#SBATCH --error=/lustre/grp/cyllab/share/ljj/public/dms1030_log/s2.err
#SBATCH --time=66:00:00
#SBATCH --partition=sugon,hygon,cpu1,gpu11

dms_base_dir="/lustre/grp/cyllab/share/ljj/public/dms1030"
seq_base_dir="/lustre/grp/cyllab/share/ljj/public/spike1030"
out_dir="${dms_base_dir}/dms_results"

for tag in rbd spike
do
/lustre/grp/cyllab/luojj/anaconda3/envs/opencpd/bin/python3 s2_gen_dms.py \
--dms_npz "${dms_base_dir}/merge/_dms.npz" \
--count_npz "${seq_base_dir}/seq_results/${tag}/${tag}_count.npz" \
--tag "${tag}" \
--out_dir "${out_dir}/${tag}"
done


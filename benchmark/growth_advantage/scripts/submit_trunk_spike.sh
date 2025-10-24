#!/bin/bash

num_subsets=600  
dir=/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/growth_advantage
dataset=/lustre/grp/cyllab/yangsj/evo_pred/0article/data/processed/241030/spike/2023-10-01/TestFull.csv

cd $dir
mkdir -p ${dir}/logs
mkdir -p ${dir}/results

for i in $(seq 0 $((num_subsets - 1))); do
    cat <<EOF > ${dir}/logs/calc_GA_${i}.sh
#!/bin/bash
#SBATCH -o ${dir}/logs/growAdv_${i}_%j.out
#SBATCH -e ${dir}/logs/growAdv_${i}_%j.err
#SBATCH -J growAdv_${i}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80g
#SBATCH --partition=cpu2,sugon
#SBATCH --time=23-23:00:00

source /lustre/grp/cyllab/yangsj/miniconda/etc/profile.d/conda.sh
conda init
conda activate dl
python ${dir}/scripts/spike_test_ga_trunk.py --subset ${i} --num_subsets ${num_subsets} --dataset_path ${dataset} --save_dir ${dir} > ${dir}/logs/log_trunk_${i}.log
EOF

    # 提交作业
    sbatch ${dir}/logs/calc_GA_${i}.sh
done
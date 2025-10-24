#!/bin/bash

# Common SLURM settings
SBATCH_BASE="#SBATCH -o /lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/log/update_batches_S3thres.%j.out
#SBATCH -e /lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/log/update_batches_S3thres.%j.err
#SBATCH --ntasks=1
#SBATCH --partition=sugon,cpu_short,gpu11
#SBATCH --cpus-per-task=32
#SBATCH --mem=40g
#SBATCH --time=1:00:00"

# Common variables
# dataset=/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd/2023-10-01/TestFull.csv
# res_info=/lustre/grp/cyllab/yangsj/evo_pred/1article/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-21502.csv
# save_dir=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_Thres
# tag='rbd'

## update model
dataset=/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd/2023-10-01/TestFull.csv
res_info=/lustre/grp/cyllab/yangsj/evo_pred/1article/predict/results/rbd_single_JN1era_updateModel/TestFull_regres_outputs_labels-step-36410.csv
save_dir=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_Thres_updateModel
tag='rbd'

# dataset=/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd/2022-09-01/TestFull.csv
# res_info=/lustre/grp/cyllab/yangsj/evo_pred/0article/generalization/XBB_era/results/rbd_single_XBBera_split220901_end241030_rand721/TestFull_regres_outputs_labels-step-38360.csv
# save_dir=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/XBBera_Thres
# tag='rbd'

# dataset=/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/spike/2023-10-01/TestFull.csv
# res_info=/lustre/grp/cyllab/yangsj/evo_pred/0article/generalization/spike/rbd_single_JN1era_spike_split231001_end241030/TestFull_regres_outputs_labels-step-18485.csv
# save_dir=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_spike_Thres
# tag='spike'

# dataset=/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd/2023-10-01/TestFull.csv
# res_info=/lustre/grp/cyllab/yangsj/evo_pred/0article/ablation/dms_ablation/results/rbd_single_JN1era_split231001_allkeep_dms_ablation_repeat/testFull_regres_outputs_labels-step-34824.csv
# save_dir=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_ablation_keepall_Thres
# tag='rbd'

## module ablation
# dataset=/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd/2023-10-01/TestFull.csv
# tag='rbd'
# res_info=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/results/no_backgrounds_rep1/TestFull_regres_outputs_labels_no_backgrounds_rep1.csv
# save_dir=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/analysis/no_backgrounds_Thres
# res_info=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/results/esm2_rep1/TestFull_regres_outputs_labels_esm2_rep1.csv
# save_dir=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/analysis/esm2_Thres
# res_info=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/results/all_keep_rep1/TestFull_regres_outputs_labels_all_keep_rep1.csv
# save_dir=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/analysis/all_keep_Thres
# res_info=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/results/no_dms_all_rep1/TestFull_regres_outputs_labels_no_dms_all_rep1.csv
# save_dir=/lustre/grp/cyllab/share/evolution_prediction_dl/ablation/analysis/no_dms_all_Thres

# ## JN.1 spike
# dataset=/lustre/grp/cyllab/yangsj/evo_pred/0article/data/processed/241030/spike/2023-10-01/TestFull.csv
# res_info=/lustre/grp/cyllab/yangsj/evo_pred/0article/generalization/spike/rbd_single_JN1era_spike_split231001_end241030/TestFull_regres_outputs_labels-step-18485.csv
# save_dir=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_spike_Thres
# tag='spike'

## JN.1 update
# dataset=/lustre/grp/cyllab/yangsj/evo_pred/0article/data/processed/250516/rbd/2023-10-01/TestFull.csv
# res_info=/lustre/grp/cyllab/yangsj/evo_pred/1article/generalization/JN1_era_update/results/rbd_single_JN1era_update/TestFull_regres_outputs_labels-step-36410.csv
# save_dir=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_update_Thres
# tag='rbd'

mkdir -p $save_dir
# Create array of threshold values
thresholds=$(seq 0.05 0.025 0.5)
echo "Thresholds to process: ${thresholds}"

for prop_thres_T3 in $thresholds; do
    echo $prop_thres_T3
    # Create a temporary submission script for each threshold
    tmp_script=/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/sh/update_Thres_${prop_thres_T3}.sh

    cat << EOF > $tmp_script
#!/bin/bash
${SBATCH_BASE}
#SBATCH -J S3thres_${prop_thres_T3}

/lustre/grp/cyllab/yangsj/miniconda/envs/dl/bin/python3 /lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/calc_evaluation_matrics_thresholds.py \\
--prop_thres_T3 ${prop_thres_T3} \\
--save_dir ${save_dir} \\
--dataset ${dataset} \\
--res_info ${res_info} \\
--tag ${tag}

EOF

    sbatch "$tmp_script"
done
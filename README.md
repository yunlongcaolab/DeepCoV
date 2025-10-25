# DepCoV: Evolutionary Prediction and Analysis Framework

## Overview
This repository contains the codebase, data, and analysis pipelines for DepCoV, a framework designed for forecasting the prevalence and evolution of SARS-CoV-2 variants. It integrates evolutionary sequences, mutational scanning, and epidemiological modeling to analyze dominant variants, predict mutational hotspots, and evaluate variant dynamics.

The repository is organized into modules for data preprocessing, model training, prediction, benchmarking, ablation studies, in silico mutational scanning, and spatiotemporal evaluation.

## Directory Structure
```
DeepCoV
├── ablation # ablation studies for evaluating model components
│   ├── ablation_bar_scatter.R # ablation RMSE from three independent replicates
│   ├── ablation_dynamic_thres_curves.R # ablation curves under variable thresholds
│   ├── ablation_dynamic_topk_calc.R # dynamic top-k ablation calculation
│   ├── ablation_dynamic_topk_plot.R # dynamic top-k ablation plot
│   ├── ablation_growth_reconstruction.R # growth advantage reconstruction
│   ├── ablation_mutscanning_scatter.R # mutational scanning scatter plot
│   ├── ablation_topk_bar.R # top-k ablation bar plot
│   ├── ablation_topk.R # top-k ablation
│   ├── analysis # analysis output data
│   ├── predict_module_ablation.sh
│   ├── results
├── benchmark # benchmarking DepCoV against other models (e.g., EVEscape, E2VD)
│   ├── analysis
│   ├── DMS
│   │   └── fitting
│   ├── dynamic_topk_calc.R
│   ├── dynamic_topk_plot.R
│   ├── dynamic_topk_plot_venn.R
│   ├── E2VD
│   ├── EVEscape
│   ├── growth_advantage
│   ├── plots
│   ├── submit_sim.sh
│   ├── timeline.R
│   ├── topk_comparison.R
│   └── topk_venn.py
├── data # datasets and other related preprocessed data
│   ├── dms # preprocessed dms data for all references
│   │   └── _dms.npz
│   ├── processed
│   │   ├── to241030
│   │   │   ├── rbd # dataset for model trained on RBD
│   │   │   │   ├── 2022-09-01 # dataset for XBB era prediction 
│   │   │   │   └── 2023-10-01 # dataset for JN.1 era prediction 
│   │   │   └── spike # dataset for model trained on spike
│   │   └── to250516 # updated dataset
│   │       ├── rbd
│   │       │   └── 2023-10-01
│   │       └── spike
│   ├── raw
│   └── reference # sequence references
├── DepCoV
│   ├── dataset # dataset processing class
│   ├── main_hotspots_predict.py
│   ├── main_proportion_continue_predict.py
│   ├── main_proportion_predict.py
│   ├── model # main model (including ablations and mut scanning)
│   └── utils # others functions
├── environment.yml
├── generalization
│   ├── continue # predict t0+1~t0+60 days
│   ├── spike # trained on spike (train on dataset from data/processed/to241030/spike/2023-10-01)
│   ├── update
│   └── XBB_era # prediction XBB era (train on dataset from data/processed/to241030/rbd/2022-09-01)
├── insilico_mutational_hotspot_scanning
│   ├── 01_single-residue_mutant_sequences_generation.py
│   ├── 02_scanning_dataset_generation.py
│   ├── 03_evolutionary_hotspots_scanning.R
│   ├── 03_evolutionary_hotspots_scanning_spike.R
│   ├── 05_truth_curves.R
│   ├── config
│   ├── data
│   ├── predict_hotspots.sh
│   ├── results
│   ├── submit.sh
│   └── Truth_curves.pdf
├── predict
├── preprocessing
├── README.md
└── spatiotemporal_evaluation
    ├── correlation.R
    ├── dynamic_assessment_under_variable_prevalence_threshold
    ├── growth_trajectory_reconstruction.R
    ├── growth_trajectory_reconstruction_single.R
    ├── plots
    └── prevalence_regional_heterogeneity.R
```

## Environment setup
the environment is defined in environment.yml. to create the environment:
```
conda env create -f environment.yml
conda activate evolutionpredict
```
ensure the environment also contains required R packages (e.g., tidyverse, ggsci, ggvenn, plotnine, ggrepel).

## Raw input prepare
1. data/raw/spikeprot1030.fasta.gz: raw sequence
first download SARS-CoV-2 spike sequences from GISAID
then run
```
tar -xvf hCoV-19_spikeprot1030.tar.xz 
gzip -c spikeprot1030.fasta > data/raw/spikeprot1030.fasta.gz
```
2. raw spatiotemporal metadata
manually download and combine the metadata: raw_metadata_241030.csv.gz and raw_metadata_250507.csv.gz

3. (deleted) data/raw/all_epi_ids_250517.csv: all accessible gisaid ids download through GisaidR

4. The deep mutational scanning datasets for single mutations obtained from  public repositories including:
> https://github.com/tstarrlab/SARS-CoV-2-RBD_DMS_Omicron-EG5-FLip-BA286
> 
> https://github.com/tstarrlab/SARS-CoV-2-RBD_DMS_Omicron-XBB-BQ
> 
> https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps
> 
> https://github.com/dms-vep/SARS-CoV-2_Omicron_BA.2_spike_ACE2_binding
> https://github.com/dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS
> https://github.com/dms-vep/SARS-CoV-2_Delta_spike_DMS_REGN10933
> https://github.com/dms-vep/SARS-CoV-2_Omicron_BA.1_spike_DMS_mAbs,https://github.com/jbloomlab/SARS2-RBD-escape-calc
> https://github.com/yunlongcaolab/convergent_RBD_evolution,https://github.com/yunlongcaolab/SARS-CoV-2-reinfection-DMS

## Usage
### Preprocessing
use scripts in preprocessing/ to generate aligned datasets and DMS features.

### Prediction
use predict.sh and main_proportion_predict.py to run prevalence predictions.
```
sbatch /lustre/grp/cyllab/share/evolution_prediction_dl/predict/predict.sh
```
for training, set `MODE=run_train` and rewrite the corresponding config.yaml (e.g use_to_evaluate_checkpoint_path: null,dataset_csv_name: TrainVal.csv).

### Evaluation & visualization:
use scripts in benchmark/ and spatiotemporal_evaluation/ to generate performance metrics and plots.
use scripts in insilico_mutational_hotspot_scanning/ to predict the potential convergent mutations (mutation hotspots).

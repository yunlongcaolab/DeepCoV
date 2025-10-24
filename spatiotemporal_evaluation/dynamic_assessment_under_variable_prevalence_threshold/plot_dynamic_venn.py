import os
import numpy as np 
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn3
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score,precision_score,f1_score
plt.rcParams["pdf.fonttype"] = 42 

# save_plot_dir=f'/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/plots'
# file_dir='/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_Thres'
save_plot_dir=f'/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/plot/dynamic_venn'
file_dir='/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_Thres_updateModel'
### XBB era
# save_plot_dir=f'/lustre/grp/cyllab/yangsj/evo_pred/0article/generalization/XBB_era/venn_plot_dynamic'
# file_dir='/lustre/grp/cyllab/yangsj/evo_pred/0article/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/XBBera_Thres'

try:
    os.makedirs(save_plot_dir, exist_ok=True)
    print(f"Directory created: {save_plot_dir}")
except Exception as e:
    print(f"Error creating directory: {e}")

custom_colors = ['#88A69D','#A8BCCC','#E7BA8F'] 
for prop_thres_T3 in [round(i/1000, 3) for i in range(50, 500, 25)]:
    df_t123 = pd.read_csv(f'{file_dir}/S3prop_{prop_thres_T3}_S2ga_0.3.csv')
    # df_t123 = df_t123[ df_t123.name.str.contains("^XBB|EG\\.5|HK\\.3|BQ\\.1") ]

    S2_true_strain = set(df_t123.query('S2 == True').name)
    S3_true_strain = set(df_t123.query('S3 == True').name)
    S3pred_true_strain = set(df_t123.query('S3_pred == True').name)
    venn3([S2_true_strain, S3_true_strain, S3pred_true_strain], ('State2: growth advantage > 0.3', f'proportion > {prop_thres_T3} (truth)', f'State3: proportion > {prop_thres_T3} (prediction)'), set_colors=custom_colors)
    text_str = "truth:\n" + "\n".join(list(S3_true_strain)) + "\n\nprediction:\n" + "\n".join(list(S3pred_true_strain))
    plt.text(0.1, 3, text_str,ha='left', va='center',fontsize=8) #,bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    plt.tight_layout()

    plt.title(f"Major strain: proportion > {prop_thres_T3}")
    plt.savefig(f'{save_plot_dir}/venn3_S3prop_{prop_thres_T3}.pdf',format='pdf', bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    # plt.show()
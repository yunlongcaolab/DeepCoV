import os
import numpy as np 
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn3
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score,precision_score,f1_score

plt.rcParams["pdf.fonttype"] = 42 

data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl'
analysis_data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/analysis'
plot_save_dir='/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/plots'

k = 20
n_major = 10
# k = 3
# n_major = 3
save_dir = f'{plot_save_dir}/static_venn_plot_k{k}_major{n_major}'
try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Directory created: {save_dir}")
except Exception as e:
    print(f"Error creating directory: {e}")

custom_colors = ['#E7BA8F','#88A69D'] 

score_summary = pd.read_csv(f'{analysis_data_dir}/benchmark_scores_summary_static.csv'
    ).rename(columns={'ga': 'growth advantage', 'target_ratio_t1_output': 'ours', 'dms_fitting_score': 'DMS(fit)'})  # ,'dms_sum_score':'dms(sum)'
score_summary = score_summary[score_summary['rbd_name_mut'].str.contains(r'^HK\.3|BA\.2\.86|JN\.1|KP\.2|KP\.3', regex=True)]
true_strain = set(score_summary.sort_values('target_ratio_t1_label', ascending=False)
                .head(n_major)['rbd_name_mut']
                .tolist())
print('true_strain:',true_strain)
methods = ['ours', 'EVEscape', 'growth advantage','E2VD','DMS(fit)']

for method in methods:
    Lpred_true_strain = score_summary.sort_values(method,ascending = False).head(k).rbd_name_mut
    pred_true_strain = set(Lpred_true_strain)

    plt.figure(figsize=(2.5, 2.5))
    venn2([true_strain, pred_true_strain], ('truth', f'{method}'), set_colors=custom_colors)

    # add text
    text_str = "truth:\n" + "\n".join(list(true_strain)) + f"\n\nTop {k} prediction:\n" + "\n".join(Lpred_true_strain)
    plt.text(0.1, 3, text_str,ha='left', va='center',fontsize=8) 
    plt.tight_layout()

    plt.title(f"{method} prediction (k={k})")
    plt.savefig(f'{save_dir}/venn2_{method}_topk{k}.pdf', format='pdf', bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
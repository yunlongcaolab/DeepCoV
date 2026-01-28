import os
import pandas as pd

file_dir="/lustre/grp/cyllab/share/evolution_prediction_dl"
tag="241030"
# tag="250516"

data=pd.read_csv(f"{file_dir}/data/processed/to{tag}/rbd/2023-10-01/TestFull.csv")
data_merge_all = []
res_dir=f"{file_dir}/benchmark/MLR/res/TestFull_2023-10-01_to{tag}"
for t0 in set(data.t0):
    short_data = data[data['t0']==t0]
    res = pd.read_csv(f'{res_dir}/freq_forecasted_Global_{t0}.tsv',sep='\t').assign(t0 = t0)
    t1=(pd.to_datetime(t0)+ pd.Timedelta(days=30-1)).strftime("%Y-%m-%d")
    data_merge = short_data.merge(res[res['date']==t1].rename(columns={'variant':'rbd_name','date':'t1'}),how='left')
    data_merge_all.append(data_merge)
data_all = pd.concat(data_merge_all, ignore_index=True)
data_all.to_csv(f"{file_dir}/benchmark/MLR/results/TestFull_2023-10-01_to{tag}_MLR_pred30days.csv",index=False)


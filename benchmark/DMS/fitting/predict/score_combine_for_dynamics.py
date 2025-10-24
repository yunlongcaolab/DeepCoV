import os 
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path
import json
from scipy.stats import pearsonr,spearmanr

meta = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/meta241030.csv.gz', compression='gzip')
collection_date_mapper = meta.set_index('rbd_name')['collection_date'].to_dict()


res_JN1ref = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/results/Q_e_test_JN1era_refJN.1.csv')
res_KP2ref = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/results/Q_e_test_JN1era_refKP.2.csv')
res_KP3ref = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/results/Q_e_test_JN1era_refKP.3.csv')

res_dms_merge = res_JN1ref[['mutant','rbd_name','rbd_name_mut','Q_e']].rename(columns={'Q_e':'Q_e_JN1ref'}).merge(res_KP2ref[['mutant','rbd_name','rbd_name_mut','Q_e']].rename(columns={'Q_e':'Q_e_KP2ref'})).merge(res_KP3ref[['mutant','rbd_name','rbd_name_mut','Q_e']].rename(columns={'Q_e':'Q_e_KP3ref'}))
res_dms_merge['collection_date'] = pd.to_datetime([collection_date_mapper[i] for i in res_dms_merge.rbd_name])

def assign_dms_time_score(row):
    if row['collection_date'] < pd.to_datetime('2024-04-17'):
        return row['Q_e_JN1ref']
    elif pd.to_datetime('2024-04-17') <= row['collection_date'] <= pd.to_datetime('2024-04-24'):
        return row['Q_e_KP2ref']
    else:
        return row['Q_e_KP3ref']
        
res_dms_merge['dms_fitting_score'] = res_dms_merge.apply(assign_dms_time_score, axis=1)
res_dms_merge.to_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/dms_fittings_score_test_JN1era.csv',index=False)
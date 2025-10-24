import argparse
import gzip
import os
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json


directory='/lustre/grp/cyllab/yangsj/evo_pred/0article'

### RBD
tag='rbd'
motif='RBD'
count = np.load(f'{directory}/insilico_mutational_hotspot_scanning/data/synthetic_RBD_singlemut.npz')
refmut_msa = count['refmut_msa']
refmut_name = count['refmut_name']

## JN1 era
# date_tag = '241030'
# base_strain='JN.1'
# dates = pd.date_range('2024-01-20', '2024-03-31')

## XBB era
date_tag = '241030'
base_strain='XBB'
dates = pd.date_range('2022-11-01', '2023-01-01')
# date_tag = '241030'
# base_strain='XBB.1.5'
# dates = pd.date_range('2022-12-01', '2023-02-01')
# dates = pd.date_range('2023-04-25', '2023-05-01')
# dates = pd.date_range('2023-05-01', '2023-06-01')
# date_tag = '241030'
# base_strain='EG.5'
# # dates = pd.date_range('2023-07-25', '2023-08-01')
# dates = pd.date_range('2023-07-01', '2023-09-01')

## JN1 era update
# date_tag = '250516'
# base_strain='KP.3'
# dates = pd.date_range('2024-08-01', '2024-10-01') 

# date_tag = '250516'
# base_strain='LF.7'
# # dates = pd.date_range('2024-12-01', '2025-03-01') 
# dates = pd.date_range('2024-08-01', '2024-12-01') 

### NTD
# tag='spike'
# motif='NTD'
# count = np.load(f'{directory}/insilico_mutational_hotspot_scanning/data/synthetic_NTD_singlemut_include_del.npz')
# refmut_msa = count['refmut_msa']
# refmut_name = count['refmut_name']

# date_tag = '241030'
# base_strain='KP.3'
# # dates = pd.date_range('2024-08-01', '2024-09-15') 
# dates = pd.date_range('2024-06-01', '2024-09-01') 

# date_tag = '241030'
# base_strain='KP.2' 
# dates = pd.date_range('2024-05-01', '2024-07-01') 

print(base_strain,date_tag,min(dates),max(dates))
####################
if date_tag == '250516':
    test = pd.read_csv(f'{directory}/data/processed/{date_tag}/{tag}/2023-10-01/TestFull.csv') # more 
elif date_tag == '241030':
    test = pd.read_csv(f'{directory}/data/processed/{date_tag}/{tag}/2022-09-01/TestFull.csv') # more 

with open(f'{directory}/data/processed/{date_tag}/{tag}_name_mapper.json', 'r', encoding='utf-8') as file:
    rbd_name_mapper = json.load(file)
test[f'{tag}_name_mut'] = test[f'{tag}_name'].apply(lambda x: rbd_name_mapper[x])

basestrain=base_strain.replace('.','')
indices = [i for i, x in enumerate(refmut_name) if base_strain + '+' in x]
syn_date = [date.strftime('%Y-%m-%d') for date in dates]
syn_location = ['Global']

target_refmut_name = refmut_name[indices]
target_refmut_msa = refmut_msa[indices,:]
location_mapper = dict(zip(test.location,test.location_index))
date_mapper = dict(zip(test.t0,test.t0_index))

df = pd.DataFrame(columns=['location','t0',f'{tag}_name_mut'])

for strain in tqdm(target_refmut_name):
    for loc in syn_location:
        for date in syn_date:
            df_strain = pd.DataFrame({'location':[loc],'t0':[date],f'{tag}_name_mut':[strain]})
            df = pd.concat([df, df_strain], ignore_index=True) 

target_refmut_name_mapper = {d: 'n'+str(1000000+i) for i, d in enumerate(target_refmut_name)}
target_refmut_id_mapper = {d:(1000000+i) for i, d in enumerate(target_refmut_name)}

df['location_index'] = [location_mapper[i] for i in df['location']]
df['t0_index'] = [date_mapper[i] for i in df['t0']]
df[f'{tag}_name'] = [target_refmut_name_mapper[i] for i in df[f'{tag}_name_mut']]
df[f'{tag}_index'] = [target_refmut_id_mapper[i] for i in df[f'{tag}_name_mut']]
df.insert(0, "ds", "synthetic")

meta_strain = df[df[f'{tag}_name_mut'].str.contains(f"^{base_strain}", regex=True)]
meta_strain[f'{tag}_name_mut_use'] = meta_strain[f'{tag}_name_mut'].replace('XBB.1.5+F456L','EG.5').replace('XBB+S486P','XBB.1.5').replace('EG.5+L455F','HK.3')
strain_singlemut_name = set(meta_strain[f'{tag}_name_mut_use'])
strain_dates = set(meta_strain.t0)

DFsyn_exist = test.query(f'{tag}_name_mut in @strain_singlemut_name').query('t0 in @strain_dates')
strain_singlemut_name_exist=set(DFsyn_exist[f'{tag}_name_mut'])
DFsyn_exist[f'{tag}_name_mut'] = DFsyn_exist[f'{tag}_name_mut'].replace('EG.5','XBB.1.5+F456L').replace('XBB.1.5','XBB+S486P').replace('HK.3','EG.5+L455F')
DF_t0_seq_total = DFsyn_exist[['t0','n_bg_isolates','n_bg_clusters','total_isolates_t0']].drop_duplicates() # extract total count for noexists

DFsyn_noexist = meta_strain.query(f'{tag}_name_mut_use.isin(@strain_singlemut_name_exist)==False').merge(DF_t0_seq_total).assign(target_isolates_t0=0,target_ratio_t0=0,n_bg_isolates_target=0,n_isolates_target_cusum=0)
DFsyn_strain_merge = pd.concat([DFsyn_exist,DFsyn_noexist]).drop(f'{tag}_name_mut_use', axis=1) 
DFsyn_strain_merge['base_strain'] = [i.split('+')[0] for i in DFsyn_strain_merge[f'{tag}_name_mut']]

DFsyn_strain_merge.to_csv(f'{directory}/insilico_mutational_hotspot_scanning/data/alldate/{motif}singlemut_{basestrain}-{min(syn_date)}to{max(syn_date)}.csv',index=False)

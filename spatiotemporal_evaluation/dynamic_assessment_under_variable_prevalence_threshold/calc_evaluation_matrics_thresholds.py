import argparse
import numpy as np 
import pandas as pd
from plotnine import *
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score,precision_score,f1_score

from tqdm import tqdm
import json
import os

import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn3

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


parser = argparse.ArgumentParser(description='t123 evaluation')
parser.add_argument('--prop_thres_T3', type=float, required=True, help = ' end up with "/" ')
parser.add_argument('--dataset', type=str, required=True, help = 'e.g /lustre/grp/cyllab/share/ljj/public/spike1030/seq_results/rbd/2023-10-01/test.csv')
parser.add_argument('--res_info', type=str, required=True, help = 'e.g /lustre/grp/cyllab/yangsj/evo_pred/train/20241226_model/results/proportion_rbd_single_bg360/test_regres_outputs_labels-step-16540.csv')
parser.add_argument('--save_dir', type=str, required=True, help = 'end up without "/" ')
parser.add_argument('--tag', type=str, required=True, help = 'spike / rbd')
args = parser.parse_args()

save_dir = args.save_dir
prop_thres_T3 = args.prop_thres_T3
dataset_path = args.dataset
res_info_path = args.res_info
tag = args.tag

os.makedirs(save_dir,exist_ok = True)

print('======================================================================================')
print(f'save_dir: {save_dir}')
print(f'prop_thres_T3: {prop_thres_T3}')
print(f'dataset_path: {dataset_path}')
print(f'res_info_path: {res_info_path}')
print(f'tag: {tag}')
print('======================================================================================')

ga_min = 0.3
ga_max = 2
T2_try_len = 90
T3_try_len_ = 120
# T3_try_len = 120 # 150 below
# prop_thres_T3 = 0.2
# prop_thres_T3 = 0.3
cusumseq_T2=100
# prop_thres_T2 = 0.05
# save_dir = '/lustre/grp/cyllab/yangsj/evo_pred/analyse/plot/20250227_bg360_test_t123_final'


# def name_others_lineage(name):
#     strain_concern = ['WT', 'Alpha', 'Beta', 'Delta', 'Gamma', 'Eta', 'BA.1',
#                              'BA.2', 'BA.5', 'BF.7', 'BQ.1.1', 'XBB', 'XBB.1.5', 'EG.5', 'HK.3',
#                              'Flip', 'BA.2.86', 'JN.1', 'KP.2', 'KP.3'] 
#     if any(strain in name for strain in strain_concern):
#         return name.split('+')[0] 
#     else:
#         return 'others'
# def name_others_strain(name):
#     strain_concern = ['WT', 'Alpha', 'Beta', 'Delta', 'Gamma', 'Eta', 'BA.1',
#                              'BA.2', 'BA.5', 'BF.7', 'BQ.1.1', 'XBB', 'XBB.1.5', 'EG.5', 'HK.3',
#                              'Flip', 'BA.2.86', 'JN.1', 'KP.2', 'KP.3'] 
#     if name in strain_concern:
#         return name.split('+')[0] 
#     else:
#         return 'others'

def modify(res_info):
        res_info[f'{tag}_name_mut'] = [name_mapper[i] for i in res_info[f'{tag}_name']]
        res_info['t0_char'] = res_info['t0'].copy()
        res_info['t0'] = [datetime.strptime(str(i), "%Y-%m-%d") for i in res_info['t0']]
        res_info['t0_t1'] =  res_info['t1'].copy()
        res_info['t1'] = res_info.apply(lambda row: row.t0 + timedelta(days= row.t0_t1),axis=1)
        # res_info['strain'] = res_info[f'{tag}_name_mut'].apply(name_others_strain)
        # res_info['lineage'] = res_info[f'{tag}_name_mut'].apply(name_others_lineage)
        # res_info['delta_prop'] = res_info.apply(lambda row: row.target_ratio_t1_label - row.target_ratio_t0,axis=1)
        return res_info

def date_str_add(date_str,delta,return_str=False):
    new_date = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=delta)
    if return_str == True:
        new_date = new_date.strftime('%Y-%m-%d')
    return new_date

def check_t0(df):
    if df.shape[0] == 0:
        return None # np.nan #'NA'
    else:
        return df.sort_values('t0').t0.iloc[0].strftime('%Y-%m-%d')

if tag =='rbd':
    ga = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/growth_advantage/rbd_test_XBBera_ga.csv')
    name_mapper = json.load(open(f'/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/{tag}_name_mapper.json','r'))

if tag == 'spike':
    # ga = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/analyse/data/cluster_past_GA/spike_test_231001/spike_test_2023-10-01_GA_all.csv')
    ga = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/growth_advantage/spike_test_JN1era_ga.csv')
    name_mapper = json.load(open(f'/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/{tag}_name_mapper.json','r'))

ga['ga_cov'] = pd.to_numeric(ga['ga_cov'], errors='coerce')
ga['t0'] = pd.to_datetime(ga['t0'])

dataset = pd.read_csv(dataset_path)
dataset['t0'] = pd.to_datetime(dataset['t0'])

# res_info = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/train/20241226_model/results/proportion_rbd_single_bg360/test_regres_outputs_labels-step-16540.csv')
res_info = pd.read_csv(res_info_path)
res_info = modify(res_info)
res_info.head()

res_info_ga = res_info.merge(ga[['t0',f'{tag}_name','ga_cov','location']].rename(columns={'t0':'t1','ga_cov':'ga_t1'})).merge(ga[['t0',f'{tag}_name','ga_cov','location']].rename(columns={'ga_cov':'ga_t0'}))
res_info_ga = res_info_ga.merge(dataset)
res_info_ga[f'{tag}_name_mut'] = res_info_ga[f'{tag}_name_mut'].astype(str)


df_t123 = pd.DataFrame(columns=['name', 'T1', 
                           'S2','T2',# 'S2_pred','date_pred_S2','days_pred_S2_before_T2',
                           'S3','T3','S3_pred','date_pred_S3','days_pred_S3_before_T3','days_pred_S3_before_T2'])

def calc_t123(group):
    strain_name_i = group[f'{tag}_name_mut'].iloc[0]
    print(strain_name_i)
    T1_i = check_t0(group.query('n_isolates_target_cusum >=10').query('n_isolates_target_cusum <1000'))
    # T2_i = check_t0(res_info_ga.query('rbd_name_mut==@strain_name_i').query('ga_t0 > 0.3').query('ga_t0 < 2').query('target_ratio_t0 >= @prop_thres_T2'))
    T3_i = check_t0(group.query('ga_t0 > 0.3').query('ga_t0 < 2').query('target_ratio_t0 >= @prop_thres_T3'))
    # T3_i = res_info_ga[(res_info_ga['rbd_name_mut'] == strain_name_i) &(res_info_ga['ga_t0'] > 0.3) &(res_info_ga['ga_t0'] < 2) &(res_info_ga['target_ratio_t0'] >= prop_thres_T3)]

    S3_i = (pd.isna(T3_i) == False)
    
    df = pd.DataFrame({
    'name': [strain_name_i],'T1': [T1_i],
    'S2': [False],'T2': [None], # 'S2_pred': [False],'date_pred_S2': [None],'days_pred_S2_before_T2': [None],
    'S3': [S3_i],'T3': [T3_i],'S3_pred': [False],'date_pred_S3': [None],'days_pred_S3_before_T3': [None],# 'days_pred_S3_before_T2': [None]
    })   

    if group.query('n_isolates_target_cusum >=@cusumseq_T2').shape[0] != 0:
        T2_start = group.query('n_isolates_target_cusum >=@cusumseq_T2').t0.min()
        for i in range(T2_try_len):
            # T2_n = date_str_add(T2_start,i)
            # T2_n7 = date_str_add(T2_start,i-7)
            T2_n = T2_start  + timedelta(days=i)
            T2_n7 = T2_start  + timedelta(days=i-7)
            dt = group.query('t0 <= @T2_n').query('t0 >= @T2_n7')
            if dt.ga_t0.between(ga_min, ga_max).all():
                df.loc[0, 'S2']=True
                df.loc[0, 'T2']=T2_n.strftime('%Y-%m-%d')
                break   
    T2_i = df.loc[0, 'T2']
    if T1_i is not None:
        if T3_i is not None:
            T3_try_len = (datetime.strptime(T3_i, '%Y-%m-%d') - datetime.strptime(T1_i, '%Y-%m-%d')).days + 1
        else:
            # T3_try_len = 90
            T3_try_len = T3_try_len_ # 有影响的
        for i in range(T3_try_len):
            T1_n = date_str_add(T1_i,i)
            dt = group.query('t0 == @T1_n')
            if dt.shape[0] == 1:
                if dt.target_ratio_t1_output.iloc[0] >= prop_thres_T3:
                    df.loc[0, 'S3_pred']=True
                    df.loc[0, 'date_pred_S3']=T1_n.strftime('%Y-%m-%d')
                    if T3_i is not None:
                        df.loc[0, 'days_pred_S3_before_T3']=(datetime.strptime(T3_i, '%Y-%m-%d') - T1_n).days
                        # df.loc[0, 'days_pred_S3_before_T2']=(datetime.strptime(T2_i, '%Y-%m-%d') - T1_n).days
                    break
    return df

# df_t123.to_csv(f'{save_dir}/res_S3prop_{prop_thres_T3}_S2prop_{prop_thres_T2}_ga_{ga_min}_pred_new.csv',index=False)
print(f'all: {len(set(res_info_ga[f'{tag}_name_mut']))}')
print('start calculating...')
df_t123 = res_info_ga.groupby(f'{tag}_name_mut').parallel_apply(calc_t123)

df_t123.reset_index().drop('level_1', axis=1, inplace=False).to_csv(f'{save_dir}/S3prop_{prop_thres_T3}_S2ga_{ga_min}.csv',index=False)
print(f'saved at {save_dir}/S3prop_{prop_thres_T3}_S2ga_{ga_min}.csv')
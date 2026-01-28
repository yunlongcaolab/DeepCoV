import os 
os.chdir('/lustre/grp/cyllab/yangsj/evo_pred/jfc/DMS_predictor_new/predict')

import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path

# ref_strain = 'JN.1'
# ref_strain = 'KP.2'
ref_strain = 'KP.3'

target_src = ['WT','BA.1 BTI','BA.2 BTI','BA.1 BTI + BA.5','BA.2 BTI + BA.5','BA.5 BTI']
dms_home = '/lustre/grp/cyllab/yangsj/evo_pred/jfc/DMS_predictor_new/hadi/all/dump_merge/profiles'
abinfo_file='/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/data/antibody_new_source.csv'
# meta_test = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/sum/test_data/meta_test_JN1era_JN1ref.csv.gz')
meta_test = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/EVEscape/data/test_data/meta_test_JN1era_WTref.csv.gz')
ab_blacklist_file = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/data/ab_blacklist.txt'
res_file = f'/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/DMS/fitting/results/Q_e_test_JN1era_ref{ref_strain}.csv'




print(f'ref strain:{ref_strain}')
print(f'Ab source:{target_src}')

ab_blacklist = set(open(ab_blacklist_file).read().strip().split('\n'))
abinfo = pd.read_csv(abinfo_file)[['antibody', 'source', 'D614G_IC50', 'BA1_IC50', 'BA2_IC50', 'BA5_IC50', 'BA2_75_IC50', 'BQ1_1_IC50',
                                                                                                             'XBB_IC50','XBB1_5_IC50','XBB1_5_10_IC50','JN1_IC50','KP3_IC50','JN1_R346T_F456L_IC50']].rename(
    columns = {
        'D614G_IC50': 'D614G', 'BA1_IC50': 'BA.1', 'BA2_IC50': 'BA.2', 'BA5_IC50': 'BA.5', 'BA2_75_IC50': 'BA.2.75', 'BQ1_1_IC50': 'BQ.1.1',
        'XBB_IC50':'XBB','XBB1_5_IC50':'XBB.1.5','XBB1_5_10_IC50':'XBB.1.5.10','JN1_IC50':'JN.1','KP3_IC50':'KP.3','JN1_R346T_F456L_IC50':'KP.2'
    }
).query('antibody not in @ab_blacklist')


rbd_index_mapper = dict(zip(meta_test.rbd_name_mut_wt,meta_test.rbd_name))
rbd_name_mut_mapper = dict(zip(meta_test.rbd_name_mut_wt,meta_test.rbd_name_mut))
target_strains = set(meta_test.rbd_name_mut_wt)

ref_neut = abinfo.query(f'source in {target_src} and `{ref_strain}` < 10').reset_index(drop = True).set_index('antibody')[ref_strain].to_dict()

###############################
def dms_scores_to_neut(x, scores: list[float]):
    return x * np.exp(
        6 * np.max(scores) + 4.6 * np.sum(scores) # trained parameter
    )

def single_neut_score(score, min_clip = 0.001, max_clip = 10, log = False):
    if log:
        return -np.log10(np.clip(score, min_clip, max_clip))+1.0
    else:
        return 1.0 / np.clip(score, min_clip, max_clip) - 1.0 / max_clip

def log_total_neut(neut: dict[str, float]|list, min_clip = 0.001, max_clip = 10, log = False):
    if isinstance(neut, dict):
        res = np.log(np.sum([single_neut_score(v, min_clip, max_clip, log) for k, v in neut.items()]))
    else:
        res = np.log(np.sum([single_neut_score(v, min_clip, max_clip, log) for v in neut]))
    return res

def sim_neut(ref_neut: dict[str, float], dms_dict: dict[str, list[float]]):
    tgt_neut = {
        k: (dms_scores_to_neut(v, dms_dict[k]) if k in dms_dict else v) for k, v in ref_neut.items()
    }
    return tgt_neut

def immune_advantage_Q(ref_neut: dict[str, float], dms_dict: dict[str, list[float]]):
    tgt_neut = sim_neut(ref_neut, dms_dict)

    total_tgt = log_total_neut(tgt_neut.values(), min_clip=0.01, log=False)
    total_ref = log_total_neut(ref_neut.values(), min_clip=0.01, log=False)
    return total_ref - total_tgt, total_tgt, total_ref
###############################

dms_dict = {}
for i, row in tqdm(abinfo.iterrows(), total=len(abinfo), desc="Processing antibodies"):
    ab = row['antibody'] 
    dms_file = Path(dms_home) / f'Average_SARS-CoV-2|{ab}' / 'profile_clean.csv.gz'
    
    if not dms_file.exists():
        print(f'{ab} not found') 
        continue

    try:
        _scores = pd.read_csv(dms_file).assign(
            site_mut=lambda x: x['site'].astype(str) + x['mutation']
        ).query('mut_escape > 0').set_index('site_mut')['mut_escape'].to_dict()
    except Exception as e:
        print(f"Error processing {ab}: {str(e)}")
        continue

    for rbd_name_mut in target_strains:
        rbd_muts = [m for m in rbd_name_mut.replace('WT+', '').replace('WT', '').split('+') if m]
        mut_scores = []
        
        for mut_i in rbd_muts:
            mut_key = mut_i[1:] if len(mut_i) > 1 else mut_i
            if mut_key in _scores:
                print(f'Found DMS score of {mut_key} against {ab}')
                mut_scores.append(_scores[mut_key])
        
        if len(mut_scores) > 0 :
            if rbd_name_mut not in dms_dict:
                dms_dict[rbd_name_mut] = {} # 不这样的话直接往dms_dict中加元素会报错
            dms_dict[rbd_name_mut][ab] = mut_scores

sys.stderr.write(f'Calculating immune advantage...\n')
results = []
for rbd_name_mut, _dict in tqdm(dms_dict.items(), total = len(dms_dict)): 
    if rbd_name_mut in dms_dict:
        _adv, tgt_neut_val, ref_neut_val = immune_advantage_Q(ref_neut, _dict) # _dict = {antibody1:[mut1_score,mut2_score] , abtibody2:[], }
    else:
        _adv = 0
        tgt_neut_val = np.nan
        ref_neut_val = np.nan
    results.append({
        'mutant': rbd_name_mut,
        'Q_e': _adv,
        'ref_neut': ref_neut_val,
        'tgt_neut': tgt_neut_val
    })

results = pd.DataFrame(results)# .sort_values(['site', 'mutation'])
results['rbd_name'] = [rbd_index_mapper[i] for i in results.mutant]
results['rbd_name_mut'] = [rbd_name_mut_mapper[i] for i in results.mutant]
results.to_csv(res_file,index=False)

print('finished')
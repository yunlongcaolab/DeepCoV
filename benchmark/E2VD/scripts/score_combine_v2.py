import os
import numpy as np 
import pandas as pd
import json

res_dir='/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/results'

Ldat = []
for i in range(5):
    print(i)
    dat = pd.read_csv(f'{res_dir}/E2VD_bind_output_test_JN1era_fold{i}.csv')
    Ldat.append(dat)
bind_res = pd.concat(Ldat).groupby('rbd_index').agg({'regression_output': 'mean','classification_output': 'mean'}).reset_index()

Ldat_expr = []
for i in range(5):
    print(i)
    dat = pd.read_csv(f'{res_dir}/E2VD_expression_output_test_JN1era_fold{i}.csv')# .assign(fold=i)
    Ldat_expr.append(dat)
expr_res = pd.concat(Ldat_expr).groupby('rbd_index').agg({'regression_output': 'mean','classification_output': 'mean'}).reset_index()

escape_res = pd.read_csv(f'{res_dir}/E2VD_escape_output_test_JN1era.csv')
escape_res['rbd_index'] = [ int(i) for i in escape_res['rbd_index']]

### combine
res_combine = pd.merge(bind_res[['rbd_index','regression_output']].rename(columns={'regression_output': 'bind'}),
         expr_res[['rbd_index','regression_output']].rename(columns={'regression_output': 'expr'}),
     ).merge(escape_res[['rbd_index','regression_output']].rename(columns={'regression_output': 'escape'}))
res_combine['rbd_name'] = 'r' + res_combine['rbd_index'].astype(str)

a=1
b=1
c=1
# res_combine = res_combine.assign(
#     E2VD=lambda df: (
#         np.exp(a * np.where(df['expr'] < 0.7, df['expr'] - 0.7, 0)) +
#         np.exp(b * np.where(df['bind'] < 0.25, df['bind'] - 0.25, 0)) +
#         np.exp(c * np.where(df['escape'] > 0.5, df['escape'] - 0.5, 0))
#     )
# ).sort_values('E2VD',ascending = False)

res_combine = res_combine.assign(
    E2VD=lambda df: (
        np.exp(a * (df['expr'] - 0.7)) +
        np.exp(b * (df['expr'] - 0.25)) +
        np.exp(c * (df['expr'] - 0.5))
    )
).sort_values('E2VD',ascending = False)

name_mapper = json.load(open('/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd_name_mapper.json','r'))
res_combine['rbd_name_mut'] = [name_mapper[i] for i in res_combine.rbd_name]
res_combine.to_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/E2VD_scores_test_JN1era_v2.csv',index = False)
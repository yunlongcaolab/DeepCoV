import os
# os.chdir('/lustre/grp/cyllab/yangsj/evo_pred/analyse/data/cluster_past_GA')
# 确保调用python模块和本脚本在同一目录下
from base_classes import *
from utils import *
from data_reader import *

import pandas as pd
import numpy as np
import re
from datetime import datetime,timedelta
from scipy import stats
# from Bio import SeqIO
from collections import Counter
from math import sqrt
import json
from tqdm import tqdm
import warnings

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import argparse


# 添加参数解析
parser = argparse.ArgumentParser(description='Process a subset of data.')
parser.add_argument('--subset', type=int, required=True, help='Subset index to process')
parser.add_argument('--num_subsets', type=int, required=True, help='Total number of subsets to process')
parser.add_argument('--dataset_path', type=str, required=True, help='dataset to calculate')
parser.add_argument('--save_dir', type=str, required=True, help='save dir')



args = parser.parse_args()

# 读取数据
rbd_dataset = pd.read_csv(args.dataset_path)
# in_file = "/lustre/grp/cyllab/share/ljj/public/spike1030/seq_results/rbd/rbd_count_smooth.npz"
in_file = "/lustre/grp/cyllab/yangsj/evo_pred/0article/data/processed/241030/rbd/rbd_count_smooth.npz"
func = CountReader(in_file)

# 分割数据
data = rbd_dataset
subset_size = len(data) // args.num_subsets  # 假设num_subsets是你要分割的子集数量
start_idx = args.subset * subset_size
end_idx = (args.subset + 1) * subset_size if args.subset < args.num_subsets - 1 else len(data)

subset_data = data.iloc[start_idx:end_idx]

# 继续处理子集数据
def calc_adv(k,n):
    date_t=[i for i in range(len(k))]
    generation_time = 7 
    reproduction_number = 1
    alpha=0.95
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adv = statsmodel_fit(alpha,date_t, k, n,generation_time, reproduction_number).fd_mle
        except:
            adv = 'error'
    return adv

def GA_from_past(DF_calc):
    def calc_ga_cov(row):
        seq = func.seq_count_by_date(row['location'], row['rbd_name'], '2019-12-24', tb=row['t0'])
        total = func.total_count_by_date(row['location'], '2019-12-24', tb=row['t0'])
        return calc_adv([int(i) for i in seq], [int(i) for i in total])
    DF_calc['ga_cov'] = DF_calc.parallel_apply(calc_ga_cov, axis=1)
    return DF_calc

# start
GA_calc = GA_from_past(subset_data)
GA_calc.to_csv(f'{args.save_dir}/results/rbd_test_GA_trunk{str(args.subset)}.csv',index=False)


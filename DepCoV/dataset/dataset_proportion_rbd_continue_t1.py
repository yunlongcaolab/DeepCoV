import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from scipy.spatial.distance import squareform, pdist, cdist
import esm
import random
from dataset.evodataV5 import EvoData as EvoData_V5
from dataset.evodataV5_1 import EvoData as EvoData_V5_1
from dataset.evodataV5_0_1 import EvoData as EvoData_V5_0_1
from dataset.evodataV5_1_1 import EvoData as EvoData_V5_1_1
from dataset.evodataV6_dms_reader import DMSAccessor,read_dms
from dataset.evodataV6_data_reader import CountReader
import dataset
from datetime import datetime,timedelta
from transformers import AutoTokenizer
import pandas as pd
import math


class SARS2SpikeProteinDataset_EvoData_V6_rbd(Dataset):
    def __init__(self,args, config,ds = None):
        super().__init__()
        self.args = args
        self.config = config
        #CountReader and DMSAccessor
        self.reader_a = CountReader(count_npz=config['dataset']['path']['count_npz'])
        self.reader_b = read_dms(config['dataset']['path']['base_dms_npz'], tag=config['dataset']['tag'], ab_escape_cluster=f'cluster{str(config['feature']['ab_cluster'])}') 
       
        
        #将ref的dms中本身的值置0
        mutant = self.reader_b['mutant']
        ms_bind = self.reader_b['bind_feature']['msa']
        ms_escape = self.reader_b['escape_feature']['msa']

        value_to_index = {value: idx for idx, value in enumerate(mutant)}
        vectorized_lookup = np.vectorize(value_to_index.get)

        indices_bind = vectorized_lookup(ms_bind)
        for i in range(indices_bind.shape[0]):
            for j in range(indices_bind.shape[1]):
                self.reader_b['bind_feature']['value'][i,j,indices_bind[i,j]]  = 0.0
        indices_eacape = vectorized_lookup(ms_escape)
        for i in range(indices_eacape.shape[0]):
            for j in range(indices_eacape.shape[1]):
                self.reader_b['escape_feature']['value'][i,j,indices_eacape[i,j]]  = 0.0
        
        #将nan置-5.0
        self.reader_b['escape_feature']['value'][np.isnan(self.reader_b['escape_feature']['value'])] = -5.0
        self.reader_b['bind_feature']['value'][np.isnan(self.reader_b['bind_feature']['value'])] = -5.0

        #取出原始的dms,分6类
        self.dms_new = self._dms_(self.reader_b)

        #base dataset.csv
        dataset_csv = pd.read_csv(config['dataset_csv_path'])
        if args.mode == 'run_train':
            self.dataset_csv = dataset_csv[dataset_csv['ds'] == ds].reset_index(drop=True)
        else:
            self.dataset_csv = dataset_csv
        
        if (config['dataset']['t1_isolate_filter'] is not None) & (args.mode == 'run_train'):
            self.dataset_csv['target_isolates_t1'] = None    
            def _add_t1_infor(x):
                t1_info=self.reader_a.get_t1_values(loc_index = x['location_index'],
                                                    seq_index = x['rbd_index'],
                                                    t0_index = x['t0_index'],
                                                    min_total_isolates_t1=config['dataset']['min_total_isolates_t1'])
                x['target_isolates_t1'] = t1_info['target_isolates_t1'][config['label']['target_t1_point']-1]
                
                return x
            self.dataset_csv = self.dataset_csv.apply(_add_t1_infor,axis=1)
            self.dataset_csv = self.dataset_csv[self.dataset_csv['target_isolates_t1'] >= config['dataset']['t1_isolate_filter']].reset_index(drop=True)
            self.dataset_csv = self.dataset_csv.drop('target_isolates_t1', axis=1)
        
        #过滤t1不足预测时间点的样本
        if args.mode == 'run_validation':
            self.dataset_csv['target_ratio_t1'] = None    
            def _add_t1_infor(x):
                t1_info=self.reader_a.get_t1_values(loc_index = x['location_index'],
                                                    seq_index = x['rbd_index'],
                                                    t0_index = x['t0_index'],
                                                    min_total_isolates_t1=config['dataset']['min_total_isolates_t1'])
                if len(t1_info['target_ratio_t1']) < config['label']['target_t1_point']:
                    x['target_ratio_t1'] = 0
                else:
                    x['target_ratio_t1'] = 1
                
                return x
            self.dataset_csv = self.dataset_csv.apply(_add_t1_infor,axis=1)
            self.dataset_csv = self.dataset_csv[self.dataset_csv['target_ratio_t1'] == 1].reset_index(drop=True)
            self.dataset_csv = self.dataset_csv.drop('target_ratio_t1', axis=1)

        #msa trandformer
        _, self.msa_transformer_alphabet = esm.pretrained.load_model_and_alphabet_local(config['dataset']['path']['msa_transformer_model_path'])
        self.msa_transformer_batch_converter =  self.msa_transformer_alphabet.get_batch_converter()

    def __getitem__(self,index):
        out = {}
        item = self.dataset_csv.loc[index]
        loc_index = item['location_index']
        t0_index = item['t0_index']
        rbd_index = item['rbd_index']
        t0 = item['t0']
        location = item['location']
        rbd_name = item['rbd_name']

        #count feature
        params = dict(loc=location, # location
                        t0=t0, # t0日起
                        target_name=rbd_name, # target序列的名字
                        top_k=self.config['feature']['top_k'],  # 选多少条背景序列
                        n_bg_days=self.config['feature']['n_bg_days'], #背景的时间窗 180天
                        stride=self.config['feature']['stride'], # 背景的步长 3
                        shuffle=True)
        out_a = self.reader_a.query_sample_by_name(**params)
        out_1_t1 = self.reader_a.get_t1_values(loc_index = loc_index,
                                               seq_index = rbd_index,
                                               t0_index = t0_index,
                                               min_total_isolates_t1=self.config['dataset']['min_total_isolates_t1'])
        #t1mask
        if self.config['label']['t1_mask_type'] == 'total_isolates_t1':
            t1_label_mask = out_1_t1["target_ratio_t1_mask"]#要mask的等于0
        elif self.config['label']['t1_mask_type'] == 'target_ratio_t1_0_0_1':
            t1_label_bool =  out_1_t1["target_ratio_t1"]  >= 0.01#要mask的等于0
            t1_label_mask = np.logical_and(t1_label_bool,out_1_t1["target_ratio_t1_mask"])#两个都是1才是要计算的

        #t1_label_weight
        if self.config['label']['t1_label_weight_type'] == 't1_label':
            t1_label_weight = out_1_t1["target_ratio_t1"]

        if self.config['label']['t1_day_continue_weight']:
            mu = self.config['label']['t1_day_continue_weight']
            sigma = 10
            x = np.arange(1, self.config['label']['target_t1_point'] + 1)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            pdf_normalized = pdf / np.sum(pdf)
            
        out['background_count'] = torch.tensor(out_a["count"])
        out['background_ratios'] = out_a["ratio"]
        out['t0_index'] = torch.tensor(out_a["t0_index"])
        out['location_index'] = torch.tensor(out_a["location_index"])
        out['target_name'] = torch.tensor(int(rbd_name[1:]))
        out['target_ratio_t1'] = torch.tensor(out_1_t1["target_ratio_t1"])
        out['target_ratio_t1_mask'] = torch.tensor(t1_label_mask)
        out['t1_label_weight'] = torch.tensor(t1_label_weight)
        out['t1_continue_day_weight'] = torch.tensor(pdf_normalized)
        
        
        #dms delay mask
        for dms_type in ['ace2_binding','expression','ace2_neutralizing','mediated_entry','sera_escape','ab_escape']:
            out[f'{dms_type}_delay_mask'] =  torch.tensor([date >= datetime.strptime(t0, '%Y-%m-%d') + timedelta(days=self.config['feature']['dms_delay']) for date in self.dms_new[dms_type]['antigen_time']],dtype=torch.bool)

        #target and bg msa 
        msa_list = []
        for sequence in out_a["msa"]:
            merged_sequence = ''.join(sequence)
            msa_list.append(merged_sequence)
        out['tokens'] = self.__getmsatransformerinput__(msa_list)

        #dms_ref msa
        for dms_type in ['ace2_binding','expression','ace2_neutralizing','mediated_entry','sera_escape','ab_escape']:
            concat_dms_msa = np.concatenate([out_a["msa"],self.dms_new[dms_type]['msa']],axis=0) 
            concat_dms_msa_list = []
            for sequence in concat_dms_msa:
                merged_sequence = ''.join(sequence)
                concat_dms_msa_list.append(merged_sequence)
            dms_msa_transformer_batch_tokens = self.__getmsatransformerinput__(concat_dms_msa_list)
            out[f'{dms_type}_dms_ref_msa_tokns'] = dms_msa_transformer_batch_tokens
        
        #dms_value
        for dms_type in ['ace2_binding','expression','ace2_neutralizing','mediated_entry','sera_escape','ab_escape']:
            out[f'{dms_type}_dms_value'] = self.dms_new[dms_type]['value']


        return out
    
    def _dms_(self,_dms):
        bind_feature = _dms['bind_feature']
        escape_feature = _dms['escape_feature']
        dms_new = {}


        for n in ['ace2_binding','expression','ace2_neutralizing','mediated_entry']:
            d = {}
            index = np.where(np.char.find(bind_feature['feature_name'], n) != -1)[0]
            d['value'] = bind_feature['value'][index,:,:]
            d['antigen'] = bind_feature['antigen'][index]
            d['feature_name'] = bind_feature['feature_name'][index]
            d['antigen_time'] = bind_feature['antigen_time'][index]
            d['msa'] = bind_feature['msa'][index,:]
            dms_new[n] = d

        for n in ['sera_escape','ab_escape']:
            d = {}
            index = np.where(np.char.find(escape_feature['feature_name'], n) != -1)[0]
            d['value'] = escape_feature['value'][index,:,:]
            d['antigen'] = escape_feature['antigen'][index]
            d['feature_name'] = escape_feature['feature_name'][index]
            d['antigen_time'] = escape_feature['antigen_time'][index]
            d['msa'] = escape_feature['msa'][index,:]
            dms_new[n] = d

        for n in ['ace2_binding','expression','ace2_neutralizing','mediated_entry','sera_escape']:
            dms_new[n]['value'] =  np.expand_dims(dms_new[n]['value'], axis=2)
        
        li_value = []
        li_antigen = []
        li_feature_name = []
        li_antigen_time = []
        li_msa = []
        for a in ['WT','JN.1','XBB.1.5','BA.5']:
            ind = np.where(np.char.find(dms_new['ab_escape']['feature_name'], a) != -1)[0]
            li_value.append(dms_new['ab_escape']['value'][ind,:,:])
            li_antigen.append(dms_new['ab_escape']['antigen'][ind[0]])
            li_feature_name.append(dms_new['ab_escape']['feature_name'][ind[0]])
            li_antigen_time.append(dms_new['ab_escape']['antigen_time'][ind[0]])
            li_msa.append(dms_new['ab_escape']['msa'][ind[0],:])
        dms_new['ab_escape']['value'] = np.transpose(np.stack(li_value),(0,2,1,3)) 
        dms_new['ab_escape']['antigen'] = np.stack(li_antigen)
        dms_new['ab_escape']['feature_name'] = np.stack(li_feature_name)
        dms_new['ab_escape']['antigen_time'] = np.stack(li_antigen_time)
        dms_new['ab_escape']['msa'] = np.stack(li_msa)

        return dms_new

    
    def __len__(self):
        if self.args.code_testing:
            return 2000
        else:
            return len(self.dataset_csv)
    
    def _proteinlevenshtein(self,s1, s2):
  
        rows = len(s1) + 1
        cols = len(s2) + 1
        distance = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(1, rows):
            distance[i][0] = i
        for j in range(1, cols):
            distance[0][j] = j

        for col in range(1, cols):
            for row in range(1, rows):
                if s1[row - 1] == s2[col - 1]:
                    cost = 0
                else:
                    cost = 1
                distance[row][col] = min(distance[row - 1][col] + 1,     
                                        distance[row][col - 1] + 1,     
                                        distance[row - 1][col - 1] + cost)  

        return distance[-1][-1]

    def dms_feature_ab_escape_for_ref(self,out_b,seq):
        base_dms_eascape = self.reader_b.dms_reader.dms['escape_feature']
        dms_features_escape = np.multiply(out_b['dms_features_escape'], out_b['dms_features_escape_mask'])[:,:,1:]
        dms_features_escape_new = np.zeros([dms_features_escape.shape[0],dms_features_escape.shape[1],self.config['feature']['ab_cluster']])
        for j in range(seq.shape[0]):
            min_dis_dict = {}
            for i in ['WT','JN.1','XBB.1.5','BA.5']:
                indices = np.char.find(base_dms_eascape['feature_name'], i)
                first_index = np.where(indices != -1)[0][0]
                min_dis_dict[self._proteinlevenshtein(''.join(seq[j,:]), ''.join(base_dms_eascape['msa'][first_index,:]))]= first_index
            idx =min_dis_dict[(min(min_dis_dict.keys()))]
            dms_features_escape_new[j,:,:] = dms_features_escape[j,:,idx:(idx+self.config['feature']['ab_cluster'])]
        max_dms_features_escape_new= np.nanmax(dms_features_escape_new, axis=1)
        return max_dms_features_escape_new

    def __getmsatransformerinput__(self,msa):
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)

        def remove_insertions(sequence: str) -> str:
            """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
            return sequence.translate(translation)
        
        def read_msa(msa_list) -> List[Tuple[str, str]]:
            """ Reads the sequences from an MSA file, automatically removes insertions."""
            return [(str(i), remove_insertions(record)) for i,record in enumerate(msa_list)]
        
        def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
            assert mode in ("max", "min")
            if len(msa) <= num_seqs:
                return msa
            
            array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

            optfunc = np.argmax if mode == "max" else np.argmin
            all_indices = np.arange(len(msa))
            indices = [0]
            pairwise_distances = np.zeros((0, len(msa)))
            for _ in range(num_seqs - 1):
                dist = cdist(array[indices[-1:]], array, "hamming")
                pairwise_distances = np.concatenate([pairwise_distances, dist])
                shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
                shifted_index = optfunc(shifted_distance)
                index = np.delete(all_indices, indices)[shifted_index]
                indices.append(index)
            indices = sorted(indices)
            return [msa[idx] for idx in indices]
        

        msas = read_msa(msa)

     
        msa_results = greedy_select(msas, num_seqs=10000) # can change this to pass more/fewer sequences
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = self.msa_transformer_batch_converter([msa_results])
    
    
        return msa_transformer_batch_tokens
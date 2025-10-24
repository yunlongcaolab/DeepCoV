import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

### add
ori_data_embedding_dir = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/ori_data_embedding'
test_data_dir = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/test_data'
test_data_embedding_dir = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/test_data_embedding'

import esm
model, alphabet = esm.pretrained.load_model_and_alphabet_local('/lustre/grp/cyllab/yangsj/ML/esm_checkpoints/esm2_650M/esm2_t33_650M_UR50D.pt')


RBD = True

if RBD:
    batch_converter = alphabet.get_batch_converter()
    model.eval().cuda()  # disables dropout for deterministic results

    df=pd.read_csv(f'{test_data_dir}/RBD_seq_escape_test_JN1era.csv')
    seqs=df['sequence'].values


    data=[]
    for i in range(len(seqs)):
        data.append(('seq{}'.format(i),seqs[i]))

    res=[]

    batch_size=8

    for idx in tqdm(range(0,len(seqs),batch_size)):
        data_=data[idx:idx+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_)
        seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]
        # print(seq_lens)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][:,1:seq_lens-1,:].cpu().data.numpy()

        res.append(token_representations)
    res=np.concatenate(res,axis=0)

    print(res.shape)
    np.save(f'{test_data_embedding_dir}/RBD_sequences_test.npy',res)

#############################################################################################
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


RBD_emb_1=np.load(f'{ori_data_embedding_dir}/RBD_sequences_all_data.npy').astype(np.float16)
RBD_emb_2=np.load(f'{test_data_embedding_dir}/RBD_sequences_test.npy').astype(np.float16)
RBD_emb = np.concatenate([RBD_emb_1,RBD_emb_2],axis=0)
print(RBD_emb_1.shape)
print(RBD_emb_2.shape)
print(RBD_emb.shape)

shape_=RBD_emb.shape
RBD_emb=RBD_emb.reshape(shape_[0],-1)
rbd_mean=np.mean(RBD_emb,axis=0)
rbd_std=np.std(RBD_emb,axis=0)


print(rbd_mean.shape)
print(rbd_std.shape)

np.save(f'{test_data_embedding_dir}/RBD_embedding_flatten_mean_test_JN1era_add.npy',rbd_mean)
np.save(f'{test_data_embedding_dir}/RBD_embedding_flatten_std_test_JN1era_add.npy',rbd_std)


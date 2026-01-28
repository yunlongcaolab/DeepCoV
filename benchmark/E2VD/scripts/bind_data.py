import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

raw_data_path = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/test_data/RBD_seq_escape_test_JN1era.csv'
data_embedding_dir = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/test_data_embedding'


# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
import esm
model, alphabet = esm.pretrained.load_model_and_alphabet_local('/lustre/grp/cyllab/yangsj/ML/esm_checkpoints/esm2_650M/esm2_t33_650M_UR50D.pt')

batch_converter = alphabet.get_batch_converter()
model.eval().cuda()  # disables dropout for deterministic results

# test data

df=pd.read_csv(raw_data_path)
seqs=df['sequence'].values
labels = df['label'].values


data=[]
for i in range(len(seqs)):
    data.append(('seq{}'.format(i),seqs[i]))

res=[]

batch_size=8

for idx in tqdm(range(0,len(seqs),batch_size)):
    data_=data[idx:idx+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(data_)
    seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]
    #print(seq_lens)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33][:,1:seq_lens-1,:].cpu().data.numpy()

    res.append(token_representations)
res=np.concatenate(res,axis=0)

print(res.shape)
np.save(f'{data_embedding_dir}/single_bind_esm2_test_data.npy',res)
np.save(f'{data_embedding_dir}/single_bind_esm2_test_label.npy',labels)

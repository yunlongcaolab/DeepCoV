import pandas as pd
import numpy as np
from Bio import SeqIO
import gzip

def read_alignment_to_dict(file_path):
    seq_dict = {}
    with gzip.open(file_path, 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):  
            seq_dict[record.id] = str(record.seq)
    return seq_dict

JN1test = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd/2023-10-01/TestFull.csv')
alignment_dict = read_alignment_to_dict('/lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/uniq_rbd241030.aln.gz')
test_dir='/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/test_data'
data_embedding_dir = f'{test_dir}/test_data_embedding'


dat = pd.DataFrame(set(JN1test.rbd_name),columns=['rbd_name'])
dat['sequence'] = [alignment_dict[i] for i in dat['rbd_name']]
dat['label'] = [int(i[1:]) for i in dat['rbd_name']]

# dat.to_csv(f'{test_dir}/RBD_seq_test_JN1era.csv',index=False)
# labels=dat['label'].values
# np.save(f'{data_embedding_dir}/single_expr_esm2_label_test.npy',labels)
# np.save(f'{data_embedding_dir}/single_bind_esm2_label_test.npy',labels)

dat['antibody'] = 'BD57-0129'
dat.to_csv(f'{test_dir}/RBD_seq_escape_test_JN1era.csv',index=False)
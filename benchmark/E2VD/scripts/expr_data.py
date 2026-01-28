import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

import esm

test_data_dir='/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/test_data'
test_data_embedding_dir = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/test_data_embedding'
# ori_data_dir='/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/ori_data'
# ori_data_embedding_dir = '/lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/E2VD/data/ori_data_embedding'

###########
os.environ['CUDA_LAUNCH_BLOCKING']='1'

torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

# initialize the model with FSDP wrapper
fsdp_params = dict(
    mixed_precision=True,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
    cpu_offload=True,  # enable cpu offloading
)
with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
    # model, vocab = torch.hub.load("facebookresearch/esm:main", "esm2_t48_15B_UR50D")
    model, vocab = esm.pretrained.load_model_and_alphabet_local('/lustre/grp/cyllab/yangsj/ML/esm_checkpoints/esm2_650M/esm2_t33_650M_UR50D.pt') # esm.pretrained.load_model_and_alphabet_local(model_path)
    batch_converter = vocab.get_batch_converter()
    model.eval()

    # Wrap each layer in FSDP separately
    for name, child in model.named_children():
        if name == "layers":
            for layer_name, layer in child.named_children():
                wrapped_layer = wrap(layer)
                setattr(child, layer_name, wrapped_layer)
    model = wrap(model)

    model.eval()

    # Wrap each layer in FSDP separately
    for name, child in model.named_children():
        if name == "layers":
            for layer_name, layer in child.named_children():
                wrapped_layer = wrap(layer)
                setattr(child, layer_name, wrapped_layer)
    model = wrap(model)


model.eval()




# get data list
df=pd.read_csv(f'{test_data_dir}/RBD_seq_test_JN1era.csv')
seqs=df['sequence'].values
# labels=None
labels=df['label'].values

data=[]
for i in range(len(seqs)):
    data.append(('seq{}'.format(i),seqs[i]))

res=[]

batch_size=2 # ！！！！！

for idx in tqdm(range(0,len(seqs),batch_size)):
    #print('idx',idx)
    data_=data[idx:idx+batch_size]
    #print('len',len(data_))
    batch_labels, batch_strs, batch_tokens = batch_converter(data_)
    seq_lens = (batch_tokens != vocab.padding_idx).sum(1)[0]
    #print(seq_lens)

    # Extract per-residue representations (on CPU)
    # with torch.no_grad():
    #     results = model(batch_tokens.cuda(), repr_layers=[48], return_contacts=True)
    # token_representations = results["representations"][48][:,1:seq_lens-1,:].cpu().data.numpy()
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33][:,1:seq_lens-1,:].cpu().data.numpy() # last layer

    res.append(token_representations)

res=np.concatenate(res,axis=0)

print(res.shape)
np.save(f'{test_data_embedding_dir}/single_expr_esm2_test_data.npy',res)
np.save(f'{test_data_embedding_dir}/single_expr_esm2_test_label.npy',labels)


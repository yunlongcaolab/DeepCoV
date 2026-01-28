import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import json

# from s3_all_uniq_seq import read_ref_aln

# date_tag = '241030'
date_tag = '250516'
# os.chdir('/lustre/grp/cyllab/yangsj/evo_pred/0article')
# aln_path = f'data/reference/references_{date_tag}.aln'
# if date_tag == '241030':
#     base_strains="WT,Alpha,Beta,Delta,Gamma,Eta,BA.1,BA.2,BA.5,BF.7,BQ.1.1,XBB,XBB.1.5,EG.5,HK.3,Flip,BA.2.86,JN.1,KP.2,KP.3"
# if date_tag == '250516':
#     base_strains="WT,Alpha,Beta,Delta,Gamma,Eta,BA.1,BA.2,BA.5,BF.7,BQ.1.1,XBB,XBB.1.5,EG.5,HK.3,Flip,BA.2.86,JN.1,KP.2,KP.3,XEC,LF.7,LP.8,NB.1,NB.1.8.1,XFG,XFH"
# base_strains = base_strains.split(',')
# ref_aln = read_ref_aln(aln_path, base_strains, is_spike=False)

# refmut_msa = ref_aln['msa']
# refmut_name = ref_aln['base_strains']
# rbd_site_mapper = ref_aln['mapper']

meta = pd.read_csv(f'data/processed/{date_tag}/meta{date_tag}.csv.gz')

with open(f"data/processed/{date_tag}/rbd_name_mapper.json", "w") as json_file:
    json.dump(dict(zip(meta.rbd_name,meta.rbd_name_mut)), json_file, indent=4)

with open(f"data/processed/{date_tag}/spike_name_mapper.json", "w") as json_file:
    json.dump(dict(zip(meta.spike_name,meta.spike_name_mut)), json_file, indent=4)
import argparse
import gzip
import os
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json

def read_ref_aln(in_file, base_strains, is_spike=False):
    aln = list(AlignIO.parse(in_file, "fasta"))[0]
    ref_msa = np.array(aln)
    ref_names = [s.id.split("|")[1] for s in aln]
    print(f'Sequence name based on: {base_strains}')

    indexes = [ref_names.index(n) for n in base_strains]
    wt = ref_msa[ref_names.index("WT")]

    if is_spike:
        _mapper = dict()
        n = 0
        ins_i = None
        cur_ins_pos = None

        for i, aa in enumerate(wt):
            if aa != "-":
                n += 1
                _mapper[i] = str(n)
            else:
                if cur_ins_pos != n:
                    cur_ins_pos = n
                    ins_i = 1
                else:
                    ins_i += 1

                _mapper[i] = "%dins%d" % (cur_ins_pos, ins_i)

        return dict(msa=ref_msa[indexes], base_strains=base_strains, mapper=_mapper)
    else:
        msa_start = None
        msa_end = None

        rbd_start = 331
        rbd_end = 531

        _mapper = dict()
        n = 0
        ni = 0

        ins_i = None
        cur_ins_pos = None
        for i, aa in enumerate(wt):
            if aa != "-":
                n += 1
                if n == rbd_start:
                    msa_start = i
                elif n == rbd_end:
                    msa_end = i

                if rbd_start <= n <= rbd_end:
                    _mapper[ni] = str(n)
                    ni += 1

            else:
                if rbd_start <= n <= rbd_end:
                    if cur_ins_pos != n:
                        cur_ins_pos = n
                        ins_i = 1
                    else:
                        ins_i += 1

                    _mapper[ni] = "%dins%d" % (cur_ins_pos, ins_i)

        return dict(msa=ref_msa[indexes][:, msa_start:msa_end + 1], base_strains=base_strains, mapper=_mapper)


###############################################
### generate RBD single mut for all reference 
###############################################
directory='/lustre/grp/cyllab/yangsj/evo_pred/0article'
base_strains=['XBB','XBB.1.5','EG.5','HK.3','JN.1','KP.2','KP.3','LP.8','LF.7','NB.1','NB.1.8.1']

ref_aln = read_ref_aln(f'{directory}/data/reference/references_250516.aln',base_strains=base_strains, is_spike=False)
rbd_site_mapper = ref_aln['mapper']

refmut_msa = ref_aln['msa']
refmut_name = ref_aln['base_strains']
for j in tqdm(range(len(ref_aln['base_strains']))): # j :strain ; i : site
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    mutants = {}
    Lref_seq = ref_aln['msa'][j]
    for i in range(ref_aln['msa'].shape[1]): 
        if Lref_seq[i] != '-':
            for aa in amino_acids:
                if aa != Lref_seq[i]:
                    new_sequence = Lref_seq.copy()
                    new_sequence[i] = aa
                    new_sequence_name = ref_aln['base_strains'][j] + '+' + Lref_seq[i] + rbd_site_mapper[i] + aa # i 并不直接等价于编号位点 
                    refmut_msa = np.vstack((refmut_msa, new_sequence))
                    refmut_name = np.append(refmut_name, new_sequence_name)

np.savez(f'{directory}/insilico_mutational_hotspot_scanning/data/synthetic_RBD_singlemut.npz',
    refmut_msa = refmut_msa,
    refmut_name = refmut_name,
    mapper = ref_aln['mapper']
)

###############################################
### generate NTD single mut for all reference 
###############################################
# base_strains=['XBB','XBB.1.5','JN.1','KP.2','KP.3']
# ref_aln = read_ref_aln(f'{directory}/data/reference/references_250516.aln', base_strains=base_strains, is_spike=True)
# site_mapper = ref_aln['mapper']

# refmut_msa = ref_aln['msa']
# refmut_name = ref_aln['base_strains']
# for j in tqdm(range(len(ref_aln['base_strains']))): # j :strain ; i : site
#     amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
#     mutants = {}
#     Lref_seq = ref_aln['msa'][j]
#     # for i in range(ref_aln['msa'].shape[1]): # TODO：是否要处理 ‘-’ ？
#     for i in range(311): 
#         if Lref_seq[i] != '-':
#             for aa in amino_acids:
#                 if aa != Lref_seq[i]:
#                     new_sequence = Lref_seq.copy()
#                     new_sequence[i] = aa
#                     new_sequence_name = ref_aln['base_strains'][j] + '+' + Lref_seq[i] + site_mapper[i] + aa # i 并不直接等价于编号位点 
#                     refmut_msa = np.vstack((refmut_msa, new_sequence))
#                     refmut_name = np.append(refmut_name, new_sequence_name)

# np.savez(f'{directory}/insilico_mutational_hotspot_scanning/data/synthetic_NTD_singlemut_include_del.npz',
#     refmut_msa = refmut_msa,
#     refmut_name = refmut_name,
#     mapper = ref_aln['mapper']
# )
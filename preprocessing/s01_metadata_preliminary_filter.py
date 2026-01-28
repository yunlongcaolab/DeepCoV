import argparse
import os
import numpy as np 
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from Bio import SeqIO

parser = argparse.ArgumentParser()
parser.add_argument('--raw_metadata', type=str)
parser.add_argument('--tmp_dir', type=str)
parser.add_argument('--processed_metadata', type=str,help='spike_metaYYMMDD.csv.gz')
parser.add_argument('--out_seqfile', type=str,help='spike_metaYYMMDD.csv.gz')

opts = parser.parse_args()
raw_metadata = opts.raw_metadata
input_dir = opts.tmp_dir
processed_metadata = opts.processed_metadata
out_seqfile = opts.out_seqfile

def _fmt_date(in_str):
    try:
        out_str = datetime.strptime(in_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except:
        out_str = "*"
    return out_str

metadata = pd.read_csv(raw_metadata)

# keep data with correct data format
metadata["collection_date"] = metadata["collection_date"].apply(lambda d: _fmt_date(d))
metadata["submit_date"] = metadata["submit_date"].apply(lambda d: _fmt_date(d))

keep_isolates = metadata[(metadata["collection_date"] != "*") & (metadata["submit_date"] != "*")].reset_index(drop=True)

# # keep data with accesssible gisaid id
# DFid_combine = pd.read_csv('/lustre/grp/cyllab/yangsj/evo_pred/data/20250508_seqdata_refresh/results/all_epi_ids_250517.csv')

# keep_isolates = metadata[metadata.epi_id.isin(DFid_combine[0])]
# miss_isolates = metadata[~metadata.epi_id.isin(DFid_combine[0])]

keep_isolates.to_csv(processed_metadata, index=False)

id_dict = {epi_id: 1 for epi_id in keep_isolates.epi_id}
# filter sequence (generated from step 0) with checked metadata
miss_seq_file = os.path.join(input_dir, "missing_spike.fasta")

fp1 = open(out_seqfile, "a")
fp2 = open(miss_seq_file, "a")

recorders = SeqIO.parse(os.path.join(input_dir, "spikeprot_filter_s1.fasta"), "fasta")

total = 0
n_out = 0
for seq in tqdm(recorders):
    if seq.id in id_dict:
        SeqIO.write(seq, fp1, "fasta")
        n_out += 1
    else:
        SeqIO.write(seq, fp2, "fasta")
    total += 1

fp1.close()
fp2.close()
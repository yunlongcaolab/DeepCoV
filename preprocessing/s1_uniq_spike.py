import argparse
import gzip
import json
import os
import re
from collections import defaultdict

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


def filter_uniq_seq(fasta_file):
    # sequences are accepted with the following conditions matched
    # at spike level, X and non-standard AA < 10 and standard AA > 1230
    # # 20 standard aa
    standard_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    pattern = "|".join(standard_aa)
    threshold_1 = 1230 # min length
    threshold_2 = 10 # max non-standard

    uniq_spikes = defaultdict(list) 

    with gzip.open(fasta_file, "rt") as handle:
        recorders = SeqIO.parse(handle, "fasta")
        for seq in tqdm(recorders):
            s = str(seq.seq)
            n_standard_aa = re.sub(pattern, "@", s).count("@") 
            gap = s.count("-")
            non_n_standard_aa = len(s) - n_standard_aa - gap # TODO

            if n_standard_aa > threshold_1 and non_n_standard_aa < threshold_2:
                uniq_spikes[s].append(seq.id)
    return uniq_spikes


def save(inputs, out_dir):
    seq_file = os.path.join(out_dir, "uniq_spike_filter_1.fasta")
    if os.path.isfile(seq_file):
        os.remove(seq_file)

    fp1 = open(seq_file, "a") 
    id_mapper = dict()

    idx = 0
    total_isolates = 0
    for s, epi_ids in tqdm(inputs.items()):
        seq_name = "u%d" % idx
        cur_seq = SeqRecord(Seq(s), id=seq_name, name="", description="")
        SeqIO.write(cur_seq, fp1, "fasta")
        id_mapper[seq_name] = epi_ids
        total_isolates += len(epi_ids)
        idx += 1
    fp1.close()

    os.system("gzip %s" % seq_file)

    # mapper file
    map_file = os.path.join(out_dir, "uniq_spike_id_map_1.json")

    with open(map_file, "w") as fo:
        json.dump(id_mapper, fp=fo, indent=1)

    os.system("gzip %s" % map_file)

    print("%d Isolates, %d Uniq Spikes" % (total_isolates, len(inputs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_fasta', type=str)
    parser.add_argument('--out_dir', type=str, help="cleaning dir")
    opts = parser.parse_args()

    if not os.path.isdir(opts.out_dir):
        os.makedirs(opts.out_dir)

    print("Filter Uniq spike")
    uniq_recorder = filter_uniq_seq(opts.in_fasta)
    print("Save")

    save(uniq_recorder, opts.out_dir)

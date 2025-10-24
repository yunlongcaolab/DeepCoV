"""
Filter and rename all the unique sequences
"""
import argparse
import gzip
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

def filter_uniq_seq2(fasta_file):
    # sequences are accepted with the following conditions matched
    # (1) at spike level, X and non-standard AA < 10 and standard AA > 1230
    # (2) at RBD level (331-531), X and non-standard AA  < 1

    # under MSA with insertions, MSA length=1280, RBD region is 337:538
    # # 20 standard aa
    standard_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    pattern = "|".join(standard_aa)
    threshold_1 = 1230
    threshold_2 = 10
    # threshold_3 = 4
    threshold_3 = 1

    uniq_spikes = defaultdict(list)
    uniq_rbds = defaultdict(list)

    with gzip.open(fasta_file, "rt") as handle:
        recorders = SeqIO.parse(handle, "fasta")
        for seq in tqdm(recorders, desc="[Seq]"):
            s = str(seq.seq)
            rbd_s = s[337:538]

            n_standard_aa = re.sub(pattern, "@", s).count("@")
            gap = s.count("-")
            non_n_standard_aa = len(s) - n_standard_aa - gap

            n_standard_aa_rbd = re.sub(pattern, "@", rbd_s).count("@")
            gap_rbd = rbd_s.count("-")
            non_n_standard_aa_rbd = len(rbd_s) - n_standard_aa_rbd - gap_rbd

            if n_standard_aa > threshold_1 and non_n_standard_aa < threshold_2 and non_n_standard_aa_rbd < threshold_3:
                uniq_spikes[s].append(seq.id)
                uniq_rbds[rbd_s].append(seq.id)
    return uniq_spikes, uniq_rbds


def read_ref_aln(in_file, base_strains, is_spike=False):
    aln = list(AlignIO.parse(in_file, "fasta"))[0]
    ref_msa = np.array(aln)
    ref_names = [s.id.split("|")[1] for s in aln]
    # base_strains = np.array(['WT', 'Alpha', 'Beta', 'Delta', 'Gamma', 'Eta', 'BA.1',
    #                          'BA.2', 'BA.5', 'BF.7', 'BQ.1.1', 'XBB', 'XBB.1.5', 'EG.5', 'HK.3',
    #                          'BA.2.86', 'JN.1', 'KP.2', 'KP.3',
    #                          'XEC', 'LF.7', 'LP.8', 'NB.1', 'NB.1.8.1', 'XFG', 'XFH'
    #                          ]) # 并不是所有的reference都会作为命名时的index
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


def assign_name(query_seq, ref_info):
    assert len(query_seq) == ref_info["msa"].shape[1]

    cmp_res = query_seq[None] != ref_info["msa"]
    dist = cmp_res.sum(axis=-1)
    base_index = dist.argmin(axis=-1) 

    bs_name = ref_info["base_strains"][base_index]
    bs_seq = ref_info["msa"][base_index]

    mut_flag = query_seq != bs_seq
    from_aa = bs_seq[mut_flag]
    to_aa = query_seq[mut_flag]

    mut_idx = [ref_info["mapper"][i] for i in np.where(mut_flag)[0]]
    mutants = ["%s%s%s" % (a, b, c) for a, b, c in zip(from_aa, mut_idx, to_aa)]
    new_name = "+".join([bs_name] + mutants)
    return new_name


def fmt(in_mapper, inputs, tag, ref_aln_file, out_file,base_strains):
    """

    :param in_mapper:
    :param inputs:
    :param tag: spike or rbd
    :param ref_aln_file:
    :param out_file:
    :return:
    """
    assert tag in ["spike", "rbd"]

    outputs = []
    for key, values in tqdm(inputs.items(), desc="[%s unique seqs]" % tag):
        tmp = []
        for v in values:
            tmp.extend(in_mapper[v])
        outputs.append([key, tmp, len(tmp)])

    dk = pd.DataFrame(outputs, columns=["%s_seq" % tag, "epi_id", "n_isolate"])
    dk.insert(0, "%s_name" % tag, pd.Series(dk.index).apply(lambda i: "%s%d" % (tag[0], i)))

    if tag == "spike":
        info = read_ref_aln(ref_aln_file,base_strains, is_spike=True)
    else:
        info = read_ref_aln(ref_aln_file,base_strains, is_spike=False)

    mut_name = []
    for seq in tqdm(dk["%s_seq" % tag].to_numpy(), desc="[Assign mut name]"):
        query_seq = np.array(list(seq))
        mut_name.append(assign_name(query_seq, info))

    dk.insert(0, "%s_name_mut" % tag, pd.Series(mut_name))

    print("Save %s sequence" % tag)
    with open(out_file, "a") as fp:
        for name, seq in tqdm(zip(dk["%s_name" % tag], dk["%s_seq" % tag])):
            SeqIO.write(SeqRecord(Seq(seq), id=name, name="", description=""), fp, "fasta")
    os.system("gzip %s" % out_file)

    return dk


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--aln_file', type=str)
    parser.add_argument('--map_file', type=str)
    parser.add_argument('--meta_file', type=str)
    parser.add_argument('--base_strains', type=str)
    parser.add_argument('--ref_aln_file', type=str)
    parser.add_argument('--date_tag', type=str)
    parser.add_argument('--out_dir', type=str)

    opts = parser.parse_args()
    base_strains = opts.base_strains.split(',')

    if not os.path.isdir(opts.out_dir):
        os.makedirs(opts.out_dir)

    print("Load unique spike and RBD")
    uspike, urbd = filter_uniq_seq2(opts.aln_file)

    with gzip.open(opts.map_file, "rt") as fi:
        id_mapper = json.load(fi)

    # spike
    spike_file = os.path.join(opts.out_dir, f"uniq_spike{opts.date_tag}.aln")
    ds = fmt(id_mapper, uspike, "spike", opts.ref_aln_file, spike_file,base_strains)

    # rbd
    rbd_file = os.path.join(opts.out_dir, f"uniq_rbd{opts.date_tag}.aln")
    dr = fmt(id_mapper, urbd, "rbd", opts.ref_aln_file, rbd_file,base_strains)

    # merge meta
    print("Merge meta")
    spike = ds[['spike_name', 'spike_name_mut', 'epi_id']].explode(column="epi_id")
    rbd = dr[['rbd_name', 'rbd_name_mut', 'epi_id']].explode(column="epi_id")
    result = pd.merge(spike, rbd, on="epi_id")

    meta = pd.read_csv(opts.meta_file, keep_default_na=False)
    dm = pd.merge(meta, result, on="epi_id")

    print("Save meta")
    meta_file = os.path.join(opts.out_dir, f"meta{opts.date_tag}.csv.gz")
    dm.to_csv(meta_file, index=False)

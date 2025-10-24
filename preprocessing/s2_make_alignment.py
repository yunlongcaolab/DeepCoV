import argparse
import gzip
import os
import random

import numpy as np
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from joblib import Parallel, delayed
from tqdm import tqdm


def save_alignment(in_array, names, out_file):
    """

    Args:
        in_array: a list of AA sequence or a 2d numpy array of AA
        names: list, a list of sequences' names
        out_file: str, path of output file in fasta format

    Returns:
        None

    """
    outputs = []
    for name, seq in zip(names, in_array):
        outputs.append(SeqRecord(Seq("".join(seq)), id=name, name="", description=""))
    SeqIO.write(outputs, out_file, "fasta")


def chunk_fasta(fasta_file, n_chunks, out_dir, prefix="chunk"):
    """

    Args:
        fasta_file: str, path of a fasta file
        n_chunks: int, the number of chunks to split the input fasta file
        out_dir: str, path to save the chunk file in fasta format
        prefix: str, the name of chunk file

    Returns:
        (chunk sizes, chunk files)
        chunk sizes: a list of int
        chunk files: a list of path

    """
    if os.path.splitext(fasta_file)[-1] == ".fasta":
        recs = list(SeqIO.parse(fasta_file, "fasta"))
    elif os.path.splitext(fasta_file)[-1] == ".gz":
        with gzip.open(fasta_file, "rt") as fi:
            recs = list(SeqIO.parse(fi, "fasta"))
    else:
        raise RuntimeError("Not supported input")

    random.shuffle(recs)

    n_seq = len(recs)
    bounds = np.linspace(start=0, stop=n_seq, num=n_chunks + 1, dtype=np.int32)

    if not os.path.isdir(out_dir) and isinstance(out_dir, str):
        os.makedirs(out_dir)
    else:
        raise RuntimeError("Not a directory path for out_dir: %s" % out_dir)

    num = []
    files = []
    for i in range(n_chunks):
        out_file = os.path.join(out_dir, "%s_%d.fasta" % (prefix, i))
        SeqIO.write(recs[bounds[i]:bounds[i + 1]], out_file, "fasta")
        num.append(bounds[i + 1] - bounds[i])
        files.append(out_file)
    return num, files


def del_aln_gap_region(aln_file, ref_name, out_file):
    """

    Args:
        aln_file: an alignment file in fasta format.
        ref_name: str, the name of the reference sequence used to guide the deletion option.
        out_file: path of the alignment file in fasta format after deletion.

    Returns:
        None

    """
    aln = list(AlignIO.parse(aln_file, "fasta"))[0]
    msa = np.array(aln)

    seq_names = [s.id for s in aln]
    ref_idx = seq_names.index(ref_name)
    seq_names = np.array(seq_names)

    sel_cols = msa[ref_idx] != "-"
    s_msa = msa[:, sel_cols]

    save_alignment(s_msa, seq_names, out_file)


def align_chunk(infile, outfile, reffile, mafft_path):
    command = "%s --quiet --thread % s --add % s --reorder % s > %s" % (mafft_path,
                                                                        1,
                                                                        infile,
                                                                        reffile,
                                                                        outfile)
    os.system(command)
    return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='align')
    parser.add_argument('--in_fasta', type=str, required=True,
                        help="index for start chunk, include")
    parser.add_argument('--ref_aln', type=str, required=True,
                        help="path of ref_aln")
    parser.add_argument('--out_aln', type=str, required=True,
                        help="path of output alignment")
    parser.add_argument('--chunk_dir', type=str, required=True,
                        help="directory for chunks")
    parser.add_argument('--n_chunks', type=int, default=1000,
                        help="n_chunks, default 1000")
    parser.add_argument('--thread', type=int, default=32,
                        help="thread, default 32")
    parser.add_argument('--mafft_bin', type=str, required=True,
                        help="path of mafft_bin")
    parser.add_argument('--ref_seq_name', type=str, default="Spike|WT|00",
                        help="reference sequence name to locate the gap region, default Spike|WT|00")

    opts = parser.parse_args()

    chunk_sizes, chunk_files = chunk_fasta(fasta_file=opts.in_fasta,
                                           n_chunks=opts.n_chunks,
                                           out_dir=opts.chunk_dir,
                                           prefix="chunk")

    print("Chunks: %d, %d Sequences." % (len(chunk_sizes), sum(chunk_sizes)))
    print(chunk_sizes)

    # align by chunk
    aln_files = Parallel(n_jobs=opts.thread)(delayed(align_chunk)(path,
                                                                  os.path.splitext(path)[0] + ".aln",
                                                                  opts.ref_aln,
                                                                  opts.mafft_bin)
                                             for path in tqdm(chunk_files, desc="[Align Chunks]"))

    all_names_ref = [seq.id for seq in SeqIO.parse(opts.ref_aln, "fasta")]
    print("All reference sequences' name: ", all_names_ref)

    # delete gap region
    results = []
    print("Delete gap region, remove reference sequences in alignments, and merge chunks")
    for path in tqdm(aln_files, desc="[Delete Gap Region]"):
        out_file = os.path.splitext(path)[0] + "_del.aln"
        del_aln_gap_region(path, opts.ref_seq_name, out_file)

        results.extend([seq for seq in SeqIO.parse(out_file, "fasta") if seq.id not in all_names_ref]) 

    print("Save final alignment, %d sequences" % len(results))
    SeqIO.write(results, opts.out_aln, "fasta")
    os.system("gzip %s" % opts.out_aln)

import argparse
import gzip
import os
import re
from datetime import datetime

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

def _fmt_date(in_str):
    try:
        out_str = datetime.strptime(in_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except:
        out_str = "*"
    return out_str


ORIGINALS = ['oiginal', 'oiriginal', 'orginal', 'orifinal', 'orig9inal', 'origginal',
             'origianal', 'origianl', 'origin', 'origina', 'origina;',
             'origina\\u00f1', 'origina\\u010d', 'originaal',
             'originak', 'original', 'original isolate', 'original isolate isolate',
             'original sample', 'original swab', 'original,', 'original.',
             'originale', 'originall', 'originalo', 'originial', 'originjal',
             'originla', 'originzl', 'origional', 'orignal', 'orignial', 'origninal',
             'oriiginal', 'oriignal', 'oringinal', 'oroginal', 'orriginal']

PATTERN = r"Spike\|(.+)\|(\d{4}-\d{1,2}-\d{1,2})\|(EPI_ISL_\d+)\|(.+?)\|.+?\|(Human|Hunan)"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_spike_fasta', type=str, help='spikeprotYYDD.fasta.gz')
    parser.add_argument('--output_dir', type=str)
    opts = parser.parse_args()

    # set path
    if not os.path.isdir(opts.output_dir):
        os.makedirs(opts.output_dir)

    out_file = os.path.join(opts.output_dir, "spikeprot_filter_s1.fasta")
    outlier_file = os.path.join(opts.output_dir, "spike_outlier_s1.fasta")

    # convert raw fasta, remove star
    tmp_fasta = os.path.join(opts.output_dir, "tmp_s1.fasta")

    if not os.path.isfile(tmp_fasta):
        with gzip.open(opts.raw_spike_fasta, "r") as fi:
            with open(tmp_fasta, "w") as fo:
                for line in tqdm(fi):
                    b_line = line.decode("latin")
                    s_line = b_line.strip().strip("*")
                    print(s_line, file=fo)

    # filter
    if os.path.isfile(out_file):
        os.remove(out_file)

    if os.path.isfile(outlier_file):
        os.remove(outlier_file)

    fp1 = open(out_file, "a")
    fp2 = open(outlier_file, "a")

    n1 = 0
    n2 = 0

    meta_data = []

    recs = SeqIO.parse(tmp_fasta, "fasta")
    for seq in tqdm(recs):
        name = seq.description
        tmp = re.search(PATTERN, name)
        if tmp:
            # 1 for date
            # 3 for passage
            values = list(tmp.groups())

            cur_date = _fmt_date(values[1])
            if values[3].lower() in ORIGINALS and cur_date != "*":
                meta_data.append([values[0], cur_date, values[2]])
                seq.id = values[2]
                seq.description = ""
                seq.name = ""
                SeqIO.write(seq, fp1, "fasta")
                n1 += 1
            else:
                n2 += 1
                SeqIO.write(seq, fp2, "fasta")

        else:
            n2 += 1
            SeqIO.write(seq, fp2, "fasta")

    fp1.close()
    fp2.close()

    print("Step 1: %d isolates in total, %d select, %d drop" % (n1 + n2, n1, n2))
    # save meta file
    meta_file = os.path.join(opts.output_dir, "meta_filter_s1.csv.gz")
    dt = pd.DataFrame(meta_data, columns=["strain", "collection_date", "epi_id"])
    dt.to_csv(meta_file, index=False)

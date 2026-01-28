import argparse
import os

import numpy as np

from dms.dms_reader import DMSReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dms_npz', type=str,
                        help='_dms.npz')
    parser.add_argument('--count_npz', type=str,
                        help='get msa sequences')
    parser.add_argument('--tag', type=str,
                        help='spike or rbd')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='directory to save process results')

    opts = parser.parse_args()

    if not os.path.isdir(opts.out_dir):
        os.makedirs(opts.out_dir)

    msa = np.load(opts.count_npz)["msa"]
    for cluster in ['cluster13', 'cluster21', 'cluster18', 'cluster29', 'cluster37', 'cluster56']:
        print(cluster)
        func = DMSReader(dms_npz=opts.dms_npz, tag=opts.tag, ab_escape_cluster=cluster)
        out_file = os.path.join(opts.out_dir, "dms_%s.npz" % cluster)
        results = func.query_dms_features(query_sequences=msa)
        np.savez(out_file, ** results)

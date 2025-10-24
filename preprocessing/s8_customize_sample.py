"""
@Author: Luo Jiejian
@Date: 2024/11/13
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


class CaseMaker(object):
    def __init__(self, isolate_time_table, meta_csv,
                 candidate_file, count_npz_sm,
                 background_stat_npz, major_strain,tag="rbd",
                 n_bg_clusters_threshold=16,
                 t1_isolates_total=100,
                 locations=['Global']
                 ):
        assert tag in ["rbd", "spike"]
        self.col_name = "%s_name" % tag
        self.tag = tag
        self.major_strain = major_strain

        self.isolate_time_table = isolate_time_table
        self.meta_csv = meta_csv

        self.candidates = pd.read_csv(candidate_file)
        self.inputs = np.load(count_npz_sm)
        self.backgrounds = np.load(background_stat_npz)
        self.thres_n_bg_clusters = n_bg_clusters_threshold
        self.thres_t1_isolates_total = t1_isolates_total
        self.target_locations = locations

        self._date_fmt = "%Y-%m-%d"
        self._sequence_names = self.inputs["sequence_names"]
        self._location = self.inputs["location"]
        self._date = self.inputs["date"]
        self._loc_mapper, self._date_mapper, self._sequence_names_mapper = self._all_mapper()

    def _read_case_info(self):
        # load mappers
        tb = pd.read_csv(self.isolate_time_table, keep_default_na=False)
        start_date_mapper = dict(zip(tb.antigen, tb.date))

        dt = pd.read_csv(self.meta_csv, keep_default_na=False)
        name_mapper = dict(dt[["%s_name_mut" % self.tag, "%s_name" % self.tag]].drop_duplicates().to_numpy())

        values = []
        for location in self.target_locations:
            if location not in self._loc_mapper:
                continue  
            location_index = self._loc_mapper[location]

            for c in self.major_strain:
                cn = re.split(r"\+", c)[0]
                if c in name_mapper:
                    tmp = [location, location_index,
                       name_mapper[c], c, self._sequence_names_mapper[name_mapper[c]],
                       start_date_mapper[cn], self._date_mapper[start_date_mapper[cn]],
                       ]
                    values.append(tmp)

        return pd.DataFrame(values, columns=["location", "location_index",
                                            self.col_name, "%s_mut" % self.col_name, "%s_index" % self.tag,
                                            "start_date_t0", "start_date_t0_index"])

    def _all_mapper(self):
        loc_mapper = {l: i for i, l in enumerate(self._location)}
        date_mapper = {d: i for i, d in enumerate(self._date)}
        seq_name_mapper = {n: i for i, n in enumerate(self._sequence_names)}
        return loc_mapper, date_mapper, seq_name_mapper

    def run(self):
        dt = self._read_case_info()

        outputs = []
        for idx in range(len(dt)):
            item = dict(dt.loc[idx])
            for t0_idx in tqdm(range(item["start_date_t0_index"], len(self._date)), "case"):
                n_bg_isolates_target = self.backgrounds["n_isolates_win"][item["location_index"],
                item["%s_index" % self.tag], t0_idx]

                n_isolates_target_cusum = self.backgrounds["n_isolates_all"][item["location_index"],
                item["%s_index" % self.tag], t0_idx]

                n_bg_isolates = self.backgrounds["n_bg_isolates_win"][item["location_index"], t0_idx]
                n_bg_clusters = self.backgrounds["n_bg_clusters_win"][item["location_index"], t0_idx]

                target_isolates_t0 = self.inputs["count"][item["location_index"], item["%s_index" % self.tag], t0_idx]
                total_isolates_t0 = self.inputs["total"][item["location_index"], t0_idx]
                target_ratio_t0 = target_isolates_t0 / total_isolates_t0 if total_isolates_t0 != 0.0 else 0.0

                outputs.append(["major_test",
                                item["location_index"],
                                item["%s_index" % self.tag],
                                t0_idx,
                                n_bg_isolates_target,
                                n_isolates_target_cusum,
                                item["location"],
                                item[self.col_name],
                                self._date[t0_idx],
                                n_bg_isolates,
                                n_bg_clusters,
                                target_isolates_t0,
                                total_isolates_t0,
                                target_ratio_t0,
                                item["%s_mut" % self.col_name]
                                ])
        return pd.DataFrame(outputs, columns=['ds', 'location_index', '%s_index' % self.tag, 't0_index',
                                              'n_bg_isolates_target',
                                              'n_isolates_target_cusum', 'location', '%s_name' % self.tag, 't0',
                                              'n_bg_isolates', 'n_bg_clusters', 'target_isolates_t0',
                                              'total_isolates_t0', 'target_ratio_t0', "%s_mut" % self.col_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count_npz_sm', type=str, help="*_count_smooth.npz")
    parser.add_argument('--candidate_file', type=str, help="*_candidates.csv")
    parser.add_argument('--background_stat_npz', type=str, help="*_background_stat_npz")
    parser.add_argument('--tag', type=str, help="spike or rbd")
    parser.add_argument('--major_strain', type=str, help="JN.1,KP.2,KP.3")
    parser.add_argument('--n_bg_clusters_threshold', type=int, default=16,
                        help="the min number of clusters in background (180d), default 16")
    parser.add_argument('--t1_isolates_total', type=float, default=100.0,
                        help="the min number of isolates at t1, default 100.0")
    parser.add_argument('--max_date', type=str,
                        help="max date for train and validate samples, 2023-10-01 or 2023-08-01")
    parser.add_argument('--isolate_time_table', type=str,
                        help="isolate_time_table.csv")
    parser.add_argument('--meta_csv', type=str,
                        help="meta1030.csv.gz")
    parser.add_argument('--locations', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--out_file', type=str)

    opts = parser.parse_args()
    major_strain = opts.major_strain.split(',')
    locations = opts.locations.split(',')

    sub_dir = os.path.join(opts.out_dir, opts.max_date)

    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    # for case dataset
    func_case = CaseMaker(opts.isolate_time_table,
                          opts.meta_csv,
                          opts.candidate_file, opts.count_npz_sm,
                          opts.background_stat_npz, major_strain, tag=opts.tag,
                          n_bg_clusters_threshold=opts.n_bg_clusters_threshold,
                          t1_isolates_total=opts.t1_isolates_total,
                          locations=locations
                          )
    case = func_case.run()
    case.to_csv(os.path.join(sub_dir, opts.out_file), index=False)




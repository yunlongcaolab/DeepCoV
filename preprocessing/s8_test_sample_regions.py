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


class FullTestMaker(object):
    def __init__(self, candidate_file, count_npz_sm,
                 background_stat_npz, tag="rbd",
                 n_bg_clusters_threshold=16,
                 t1_isolates_total=100,
                 min_date="2023-10-01",
                 ):
        assert tag in ["rbd", "spike"]
        self.col_name = "%s_name" % tag
        self.tag = tag

        self.candidates = pd.read_csv(candidate_file)
        self.inputs = np.load(count_npz_sm)
        self.backgrounds = np.load(background_stat_npz)
        self.thres_n_bg_clusters = n_bg_clusters_threshold
        self.thres_t1_isolates_total = t1_isolates_total
        self.D5 = min_date

        self._date_fmt = "%Y-%m-%d"
        self._sequence_names = self.inputs["sequence_names"]
        self._location = self.inputs["location"]
        self._date = self.inputs["date"]
        self._loc_mapper, self._date_mapper, self._sequence_names_mapper = self._all_mapper()

        self.flags = self._conditions()

    def _all_mapper(self):
        loc_mapper = {l: i for i, l in enumerate(self._location)}
        date_mapper = {d: i for i, d in enumerate(self._date)}
        seq_name_mapper = {n: i for i, n in enumerate(self._sequence_names)}
        return loc_mapper, date_mapper, seq_name_mapper

    def _read_candidate_flag(self):
        out = np.full((len(self._location), len(self._sequence_names)), fill_value=False, dtype=np.bool_)
        for loc, name in zip(self.candidates["location"], self.candidates[self.col_name]):
            i = self._loc_mapper[loc]
            j = self._sequence_names_mapper[name]
            out[i, j] = True
        return out

    def _read_date_flag(self):
        out = np.full((len(self._location), len(self._date)), fill_value=False, dtype=np.bool_)
        # set True for [D5, ) with Global location
        for loc in ["Asia", "Europe", "North America"]:
            out[self._loc_mapper[loc], self._date_mapper['2022-08-01']:self._date_mapper['2024-05-01']] = True
        return out

    def _conditions(self):
        # candidate condition (n_location, n_names)
        candidate_flag = self._read_candidate_flag()
        # date condition (n_location, n_dates)
        date_flag = self._read_date_flag()
        # background condition (n_location, n_dates)
        background_flag = self.backgrounds["n_bg_clusters_win"] >= self.thres_n_bg_clusters

        flag_base = date_flag[:, None, :] * background_flag[:, None, :] * candidate_flag[:, :, None]

        # t1 total condition
        t1_total_flag = self.inputs["total"] > self.thres_t1_isolates_total
        # t1 target ratio condition
        _total = np.where(self.inputs["total"] == 0, 1, self.inputs["total"])
        _ratio = self.inputs["count"] / _total[:, None, :]

        return dict(base_flag=flag_base,
                    t1_total_flag=t1_total_flag,
                    ratio=_ratio,
                    )

    def _gen_dataset(self, mask):
        loc_indexes, name_indexes, t0_indexes = np.where(mask)
        location = self._location[loc_indexes]
        names = self._sequence_names[name_indexes]
        t0 = self._date[t0_indexes]

        n_bg_isolates_target = self.backgrounds["n_isolates_win"][mask]
        n_isolates_target_cusum = self.backgrounds["n_isolates_all"][mask]
        n_bg_isolates = np.broadcast_to(self.backgrounds["n_bg_isolates_win"][:, None, :], mask.shape)[mask]
        n_bg_clusters = np.broadcast_to(self.backgrounds["n_bg_clusters_win"][:, None, :], mask.shape)[mask]

        target_isolates_t0 = self.inputs["count"][mask]
        total_isolates_t0 = np.broadcast_to(self.inputs["total"][:, None, :], mask.shape)[mask]
        target_ratio_t0 = self.flags["ratio"][mask]

        outputs = pd.DataFrame({"location_index": loc_indexes,
                                "%s_index" % self.tag: name_indexes,
                                "t0_index": t0_indexes,
                                "n_bg_isolates_target": n_bg_isolates_target.round(2),
                                "n_isolates_target_cusum": n_isolates_target_cusum.round(2),
                                "location": location,
                                "%s_name" % self.tag: names,
                                "t0": t0,
                                "n_bg_isolates": n_bg_isolates.round(2),
                                "n_bg_clusters": n_bg_clusters,
                                "target_isolates_t0": target_isolates_t0.round(2),
                                "total_isolates_t0": total_isolates_t0.round(2),
                                "target_ratio_t0": target_ratio_t0.round(5),
                                })

        outputs.insert(0, "ds", "full_test")
        return outputs

    def run(self):
        mask = self.flags["base_flag"]
        return self._gen_dataset(mask)


# class MajorTestMaker(object):
#     # occurrence date from isolate_time_table.csv
#     # keep Global location
#     # CASE_NAMES = [
#     #     # 'WT', 'Alpha', 'Beta', 'Delta', 'Gamma', 'BA.1',
#     #     # 'BA.2', 'BA.5', 'BF.7', 'BQ.1.1', 'XBB', 'XBB.1.5', 'EG.5', 'HK.3',
#     #     'BA.2.86', 'JN.1', 'KP.2', 'KP.3', 
#     #     'XEC', 'LF.7', 'LP.8', 'NB.1', 'NB.1.8.1', 'XFG', 'XFH']
    

#     def __init__(self, isolate_time_table, meta_csv,
#                  candidate_file, count_npz_sm,
#                  background_stat_npz, major_strain,tag="rbd",
#                  n_bg_clusters_threshold=16,
#                  t1_isolates_total=100):
#         assert tag in ["rbd", "spike"]
#         self.col_name = "%s_name" % tag
#         self.tag = tag
#         self.major_strain = major_strain

#         self.isolate_time_table = isolate_time_table
#         self.meta_csv = meta_csv

#         self.candidates = pd.read_csv(candidate_file)
#         self.inputs = np.load(count_npz_sm)
#         self.backgrounds = np.load(background_stat_npz)
#         self.thres_n_bg_clusters = n_bg_clusters_threshold
#         self.thres_t1_isolates_total = t1_isolates_total

#         self._date_fmt = "%Y-%m-%d"
#         self._sequence_names = self.inputs["sequence_names"]
#         self._location = self.inputs["location"]
#         self._date = self.inputs["date"]
#         self._loc_mapper, self._date_mapper, self._sequence_names_mapper = self._all_mapper()

#     def _read_case_info(self):
#         # load mappers
#         tb = pd.read_csv(self.isolate_time_table, keep_default_na=False)
#         start_date_mapper = dict(zip(tb.antigen, tb.date))
#         # start_date_mapper["KP.2"] = start_date_mapper["JN.1"] # 为什么把KP2和KP3的出现时间定的和JN1一样了？
#         # start_date_mapper["KP.3"] = start_date_mapper["JN.1"]

#         dt = pd.read_csv(self.meta_csv, keep_default_na=False)
#         name_mapper = dict(dt[["%s_name_mut" % self.tag, "%s_name" % self.tag]].drop_duplicates().to_numpy())

#         location = "Global"
#         location_index = self._loc_mapper[location]
#         values = []

#         for c in self.major_strain:
#             cn = re.split(r"\+", c)[0]
#             if c in name_mapper:
#                 tmp = [location, location_index,
#                        name_mapper[c], c, self._sequence_names_mapper[name_mapper[c]],
#                        start_date_mapper[cn], self._date_mapper[start_date_mapper[cn]],
#                        ]
#                 values.append(tmp)

#         return pd.DataFrame(values, columns=["location", "location_index",
#                                              self.col_name, "%s_mut" % self.col_name, "%s_index" % self.tag,
#                                              "start_date_t0", "start_date_t0_index"])

#     def _all_mapper(self):
#         loc_mapper = {l: i for i, l in enumerate(self._location)}
#         date_mapper = {d: i for i, d in enumerate(self._date)}
#         seq_name_mapper = {n: i for i, n in enumerate(self._sequence_names)}
#         return loc_mapper, date_mapper, seq_name_mapper

#     def run(self):
#         dt = self._read_case_info()

#         outputs = []
#         for idx in range(len(dt)):
#             item = dict(dt.loc[idx])
#             for t0_idx in tqdm(range(item["start_date_t0_index"], len(self._date)), "case"):
#                 n_bg_isolates_target = self.backgrounds["n_isolates_win"][item["location_index"],
#                 item["%s_index" % self.tag], t0_idx]

#                 n_isolates_target_cusum = self.backgrounds["n_isolates_all"][item["location_index"],
#                 item["%s_index" % self.tag], t0_idx]

#                 n_bg_isolates = self.backgrounds["n_bg_isolates_win"][item["location_index"], t0_idx]
#                 n_bg_clusters = self.backgrounds["n_bg_clusters_win"][item["location_index"], t0_idx]

#                 target_isolates_t0 = self.inputs["count"][item["location_index"], item["%s_index" % self.tag], t0_idx]
#                 total_isolates_t0 = self.inputs["total"][item["location_index"], t0_idx]
#                 target_ratio_t0 = target_isolates_t0 / total_isolates_t0 if total_isolates_t0 != 0.0 else 0.0

#                 outputs.append(["major_test",
#                                 item["location_index"],
#                                 item["%s_index" % self.tag],
#                                 t0_idx,
#                                 n_bg_isolates_target,
#                                 n_isolates_target_cusum,
#                                 item["location"],
#                                 item[self.col_name],
#                                 self._date[t0_idx],
#                                 n_bg_isolates,
#                                 n_bg_clusters,
#                                 target_isolates_t0,
#                                 total_isolates_t0,
#                                 target_ratio_t0,
#                                 item["%s_mut" % self.col_name]
#                                 ])
#         return pd.DataFrame(outputs, columns=['ds', 'location_index', '%s_index' % self.tag, 't0_index',
#                                               'n_bg_isolates_target',
#                                               'n_isolates_target_cusum', 'location', '%s_name' % self.tag, 't0',
#                                               'n_bg_isolates', 'n_bg_clusters', 'target_isolates_t0',
#                                               'total_isolates_t0', 'target_ratio_t0', "%s_mut" % self.col_name])


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
    parser.add_argument('--out_dir', type=str)

    opts = parser.parse_args()
    major_strain = opts.major_strain.split(',')

    sub_dir = os.path.join(opts.out_dir, opts.max_date)

    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    # for case dataset
    # func_case = MajorTestMaker(opts.isolate_time_table,
    #                       opts.meta_csv,
    #                       opts.candidate_file, opts.count_npz_sm,
    #                       opts.background_stat_npz, major_strain, tag=opts.tag,
    #                       n_bg_clusters_threshold=opts.n_bg_clusters_threshold,
    #                       t1_isolates_total=opts.t1_isolates_total)
    # case = func_case.run()
    # case.to_csv(os.path.join(sub_dir, "TestMajor.csv"), index=False)

    # for test dataset
    func_test = FullTestMaker(opts.candidate_file, opts.count_npz_sm,
                                opts.background_stat_npz, tag=opts.tag,
                                n_bg_clusters_threshold=opts.n_bg_clusters_threshold,
                                t1_isolates_total=opts.t1_isolates_total,
                                min_date=opts.max_date)
    test = func_test.run()
    test.to_csv(os.path.join(sub_dir, "TestFullRegions2022-08-01to2024-05-01.csv"), index=False)



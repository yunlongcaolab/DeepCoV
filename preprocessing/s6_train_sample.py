"""
@Author: Luo Jiejian
@Date: 2024/11/13
"""

import argparse
import os

import numpy as np
import pandas as pd


class TrainSampleMaker(object):

    # start date, the min t0
    D0 = "2020-02-01"

    # five bin dates for balancing sampling
    # [D0, D1), [D1, D2), [D2, D3), [D3, D4), [D4, D5)
    # D5 set by test_date_min in initialization, use as the max date for train-val samples
    D1 = "2021-07-01"
    D2 = "2022-04-01"
    D3 = "2022-12-01"
    D4 = "2023-05-01"

    # the min t0 to use `Global` location only for samples
    SD0 = "2023-01-01"

    # Given loc, seq_name and t0,
    # the [number] of t1 in range 1-60 with t1_target_ratio > max_target_ratio_threshold_for_random_sample
    # For (loc, seq_name, t0) samples with [number] >= N_T1_HIGH_PROPORTION are picked out all
    # The other (loc, seq_name, t0) groups will go random sampling
    N_T1_HIGH_PROPORTION = 1

    def __init__(self, candidate_file, count_npz_sm,
                 background_stat_npz, tag="rbd",
                 n_bg_clusters_threshold=16,
                 t1_isolates_total=100,
                 max_target_ratio_threshold_for_random_sample=0.005,
                 max_date="2023-10-01",
                 val_ratio=0.1,
                 ):
        assert tag in ["rbd", "spike"]
        self.col_name = "%s_name" % tag
        self.tag = tag

        self.candidates = pd.read_csv(candidate_file)
        self.inputs = np.load(count_npz_sm)
        self.backgrounds = np.load(background_stat_npz)
        self.thres_n_bg_clusters = n_bg_clusters_threshold
        self.thres_t1_isolates_total = t1_isolates_total
        self.thres_for_random_sample = max_target_ratio_threshold_for_random_sample
        self.D5 = max_date
        self.val_ratio = val_ratio

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
        # set True for [D0, SD0) for all locations
        out[:, self._date_mapper[self.D0]: self._date_mapper[self.SD0]] = True

        # set True for [SD0, D5) only Global location
        out[self._loc_mapper["Global"], self._date_mapper[self.SD0]: self._date_mapper[self.D5]] = True

        # consider occurrence date
        out_b = np.broadcast_to(out[:, None, :],
                                shape=(len(self._location), len(self._sequence_names), len(self._date))).copy()

        for loc, name, occ_date in zip(self.candidates["location"],
                                       self.candidates[self.col_name],
                                       self.candidates["occurrence_date"]):
            # drop samples with t0 < occurrence date
            out_b[self._loc_mapper[loc], self._sequence_names_mapper[name], 0: self._date_mapper[occ_date]] = False

        return out_b

    def _conditions(self):
        # candidate condition (n_location, n_names)
        candidate_flag = self._read_candidate_flag()
        # date condition (n_location, n_names, n_dates)
        date_flag = self._read_date_flag()
        # background condition (n_location, n_dates)
        background_flag = self.backgrounds["n_bg_clusters_win"] >= self.thres_n_bg_clusters

        flag_base = date_flag * background_flag[:, None, :] * candidate_flag[:, :, None]

        # t1 total condition
        t1_total_flag = self.inputs["total"] > self.thres_t1_isolates_total
        # t1 target ratio condition
        _total = np.where(self.inputs["total"] == 0, 1, self.inputs["total"])
        _ratio = self.inputs["count"] / _total[:, None, :]

        t1_target_ratio_flag = _ratio > self.thres_for_random_sample

        _flag_b = t1_total_flag[:, None, :] * t1_target_ratio_flag
        _flag_bp = np.pad(_flag_b, ((0, 0), (0, 0), (0, 60)), mode="constant", constant_values=False)[:, :, 1:]
        n_high_proportion = np.apply_along_axis(self._t1_win_sum, 2, _flag_bp)

        return dict(base_flag=flag_base,
                    t1_total_flag=t1_total_flag,
                    t1_target_ratio_flag=t1_target_ratio_flag,  # t0_target_ratio_flag = t1_target_ratio_flag
                    high_proportion_flag=n_high_proportion >= self.N_T1_HIGH_PROPORTION,
                    n_high_proportion=n_high_proportion,
                    ratio=_ratio,
                    )

    def dataset_flag_with_nfold_s1(self, n_fold):
        # dataset_s1 (N1) + n_fold * N1 (loc, seq_name, t0) without enough number of high proportion t1
        ds1 = self.flags["base_flag"] * self.flags["high_proportion_flag"]

        if n_fold > 0.0:
            n_samples = int(n_fold * ds1.sum())
            other_mask = self.flags["base_flag"] * np.logical_not(self.flags["high_proportion_flag"])
            other_mask_sample = self._date_stratification_sampling(other_mask, n_samples)

            return np.logical_or(ds1, other_mask_sample)
        else:
            return ds1

    def dataset_s1(self):
        # only keep (loc, seq_name, t0) samples with enough number of high proportion t1: N1 (loc, seq_name, t0)
        mask = self.dataset_flag_with_nfold_s1(0.0)
        return self._gen_dataset(mask)

    def dataset_s2(self):
        mask = self.dataset_flag_with_nfold_s1(0.5)
        return self._gen_dataset(mask)

    def dataset_s3(self):
        mask = self.dataset_flag_with_nfold_s1(1.0)
        return self._gen_dataset(mask)

    def dataset_s4(self):
        mask = self.dataset_flag_with_nfold_s1(2.0)
        return self._gen_dataset(mask)

    def _date_stratification_sampling(self, mask, n_samples):
        assert n_samples > 5
        # sample from [D0, D1)
        mask_1 = self._sample_some_true(self._sub_mask(mask, self.D0, self.D1), n_samples // 5)

        # sample from [D1, D2)
        mask_2 = self._sample_some_true(self._sub_mask(mask, self.D1, self.D2), n_samples // 5)

        # sample from [D2, D3)
        mask_3 = self._sample_some_true(self._sub_mask(mask, self.D2, self.D3), n_samples // 5)

        # sample from [D3, D4)
        mask_4 = self._sample_some_true(self._sub_mask(mask, self.D3, self.D4), n_samples // 5)

        # sample from [D4, D5)
        mask_5 = self._sample_some_true(self._sub_mask(mask, self.D4, self.D5), n_samples // 5)

        out = np.logical_or.reduce([mask_1, mask_2, mask_3, mask_4, mask_5])
        return out

    def _sub_mask(self, mask, ta, tb):
        mask_ = np.full_like(mask, fill_value=False, dtype=np.bool_)
        mask_[:, :, self._date_mapper[ta]: self._date_mapper[tb]] = True
        return mask_ * mask

    @staticmethod
    def _sample_some_true(mask, n_samples):
        """
        True values are used to sample
        :param mask: np.ndarray, np.bool_
        :return:
        """
        mask_f = mask.flatten()
        true_indices = np.where(mask_f)[0]

        n_trues = len(true_indices)
        _n_samples = min(n_trues, n_samples)

        choice_indices = np.random.choice(true_indices, _n_samples, replace=False)

        new_mask = np.full_like(mask_f, fill_value=False, dtype=np.bool_)
        new_mask[choice_indices] = True
        return new_mask.reshape(mask.shape)

    @staticmethod
    def _t1_win_sum(x):
        return np.convolve(x, np.ones(60, dtype=np.int32), "valid")

    def _gen_dataset(self, mask):
        loc_indexes, name_indexes, t0_indexes = np.where(mask)
        location = self._location[loc_indexes]
        names = self._sequence_names[name_indexes]
        t0 = self._date[t0_indexes]

        n_bg_isolates_target = self.backgrounds["n_isolates_win"][mask]
        n_isolates_target_cusum = self.backgrounds["n_isolates_all"][mask]
        n_bg_isolates = np.broadcast_to(self.backgrounds["n_bg_isolates_win"][:, None, :], mask.shape)[mask]
        n_bg_clusters = np.broadcast_to(self.backgrounds["n_bg_clusters_win"][:, None, :], mask.shape)[mask]
        n_high_proportion = self.flags["n_high_proportion"][mask]

        target_isolates_t0 = self.inputs["count"][mask]
        total_isolates_t0 = np.broadcast_to(self.inputs["total"][:, None, :], mask.shape)[mask]
        target_ratio_t0 = self.flags["ratio"][mask]

        # t1: 1-60, too big due to lots of duplicates, not save in .csv
        # get value with (loc_index, name_index, t0_index) in *_count.npz
        # t1_index
        # target_isolates_t1 = []
        # total_isolates_t1 = []
        # target_ratio_t1 = []
        # target_ratio_t1_mask = []  # according to the total_isolates_t1 values, 1 for confident target_ratio_t1
        #
        # for l_i, n_j, t_k in tqdm(zip(loc_indexes, name_indexes, t0_indexes), desc="[Search t1]"):
        #     # trimmed train samples does not exist, no need to check
        #     target_isolates_t1.append(json.dumps(self.inputs["count"][l_i, n_j, t_k + 1: t_k + 61].round(2).tolist()))
        #     _t1_total = self.inputs["total"][l_i, t_k + 1: t_k + 61]
        #     total_isolates_t1.append(json.dumps(_t1_total.round(2).tolist()))
        #     target_ratio_t1.append(json.dumps(self.flags["ratio"][l_i, n_j, t_k + 1: t_k + 61].round(5).tolist()))
        #
        #     # just used to check, all same
        #     f1 = self.flags["t1_total_flag"][l_i, t_k + 1: t_k + 61]
        #     # f2 = _t1_total > self.thres_t1_isolates_total
        #     # assert np.all(f1 == f2)
        #     target_ratio_t1_mask.append(json.dumps(f1.astype(np.int32).tolist()))

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
                                "n_high_proportion_t1": n_high_proportion,
                                "target_isolates_t0": target_isolates_t0.round(2),
                                "total_isolates_t0": total_isolates_t0.round(2),
                                "target_ratio_t0": target_ratio_t0.round(5),
                                # "target_isolates_t1": target_isolates_t1,
                                # "total_isolates_t1": total_isolates_t1,
                                # "target_ratio_t1": target_ratio_t1,
                                # "target_ratio_t1_mask": target_ratio_t1_mask,
                                })

        # split train, validate
        # remove r160 for BA.5 + Global
        flag_ = (outputs[self.col_name] == "r160") & (outputs["location"] == "Global")

        # remove r160 for BA.5 for all locations (suggest this personally)
        # flag_ = outputs[self.col_name] == "r160"

        dk = outputs[~flag_].reset_index(drop=True)
        groups = dk[["location", self.col_name]].drop_duplicates().reset_index(drop=True)
        val_gs = groups.sample(frac=self.val_ratio) # 目前是按照9:1去分
        val_gs.insert(0, "ds", "validate")

        train_gs = groups[~groups.index.isin(val_gs.index)]
        train_gs.insert(0, "ds", "train")

        groups_assigned = pd.concat([train_gs, val_gs], axis=0).reset_index(drop=True)
        dm = pd.merge(groups_assigned, dk, on=["location", self.col_name])
        return dm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count_npz_sm', type=str, help="*_count_smooth.npz")
    parser.add_argument('--candidate_file', type=str, help="*_candidates.csv")
    parser.add_argument('--background_stat_npz', type=str, help="*_background_stat_npz")
    parser.add_argument('--tag', type=str, help="spike or rbd")
    parser.add_argument('--n_bg_clusters_threshold', type=int, default=16,
                        help="the min number of clusters in background (180d), default 16")
    parser.add_argument('--t1_isolates_total', type=float, default=100.0,
                        help="the min number of isolates at t1, default 100.0")
    parser.add_argument('--t1_ratio_high_threshold', type=float, default=0.005,
                        help="")
    parser.add_argument('--max_date', type=str,
                        help="max date for train and validate samples, 2023-10-01 or 2023-08-01")
    parser.add_argument('--out_dir', type=str)

    opts = parser.parse_args()

    sub_dir = os.path.join(opts.out_dir, opts.max_date)

    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)

    func = TrainSampleMaker(opts.candidate_file, opts.count_npz_sm,
                            opts.background_stat_npz, tag=opts.tag,
                            n_bg_clusters_threshold=16,
                            t1_isolates_total=100,
                            max_target_ratio_threshold_for_random_sample=opts.t1_ratio_high_threshold,
                            max_date=opts.max_date)

    ds1 = func.dataset_s1()
    # ds1.to_csv(os.path.join(sub_dir, "dataset_fold_0.csv"), index=False)
    ds1.to_csv(os.path.join(sub_dir, "TrainVal.csv"), index=False)

    # ds2 = func.dataset_s2()
    # ds2.to_csv(os.path.join(sub_dir, "dataset_fold_0.5.csv"), index=False)

    # ds3 = func.dataset_s3()
    # ds3.to_csv(os.path.join(sub_dir, "dataset_fold_1.0.csv"), index=False)

    # ds4 = func.dataset_s4()
    # ds4.to_csv(os.path.join(sub_dir, "dataset_fold_2.0.csv"), index=False)

"""
@Author: Luo Jiejian
@Date: 2024/11/6

count by day and pick the candidate unique sequences
"""
import argparse
import gzip
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# NOTES：
# 用enumerator生成date/country/continent的index
# 对cluster做seq > thershold 的过滤
# 分天分地区 统计counts并做smooth

class DayStat(object):
    def __init__(self, meta_file, uniq_seq_aln, seq_type, threshold=30):
        assert seq_type in ["spike", "rbd"]
        self.target_name = "%s_name" % seq_type

        self.meta_file = meta_file
        self.uniq_seq_aln = uniq_seq_aln
        self.threshold = threshold
        self._date_format = "%Y-%m-%d"

        self.info = self._load_meta()

    def _load_meta(self): # # 用enumerator生成date/country/continent的index
        print("> Load meta")
        dk = pd.read_csv(self.meta_file, keep_default_na=False)

        all_dates = [datetime.strptime(d, self._date_format) for d in set(dk["collection_date"])]
        date_list = np.array([datetime.strftime(d, self._date_format)
                              for d in pd.date_range(min(all_dates), max(all_dates))])
        date_mapper = {d: i for i, d in enumerate(date_list)}

        country_list = np.array(list(set(dk["country"])))
        country_list.sort()
        country_mapper = {c: i for i, c in enumerate(country_list)}

        continent_list = np.array(list(set(dk["continent"])))
        continent_list.sort()
        continent_mapper = {c: i for i, c in enumerate(continent_list)}

        return dict(meta=dk,
                    date=date_list,
                    date_mapper=date_mapper,
                    country=country_list,
                    country_mapper=country_mapper,
                    continent=continent_list,
                    continent_mapper=continent_mapper)

    def _global_total(self):
        # global
        dc = (self.info["meta"].groupby("collection_date")
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        values = np.zeros(shape=len(self.info["date"]), dtype=np.int32)

        for t, v in zip(dc.collection_date, dc.n_isolates):
            values[self.info["date_mapper"][t]] = v
        return values

    def _continent_total(self):
        # continent
        dc = (self.info["meta"].groupby(["continent", "collection_date"])
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        values = np.zeros(shape=(len(self.info["continent"]), len(self.info["date"])),
                          dtype=np.int32)

        for continent, t, v in zip(dc["continent"], dc["collection_date"], dc["n_isolates"]):
            ri = self.info["continent_mapper"][continent]
            rj = self.info["date_mapper"][t]
            values[ri, rj] = v
        return values

    def _country_total(self):
        # country
        dc = (self.info["meta"].groupby(["country", "collection_date"])
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        values = np.zeros(shape=(len(self.info["country"]), len(self.info["date"])),
                          dtype=np.int32)

        for country, t, v in zip(dc["country"], dc["collection_date"], dc["n_isolates"]):
            ri = self.info["country_mapper"][country]
            rj = self.info["date_mapper"][t]
            values[ri, rj] = v
        return values

    def _global_filter(self):
        dc = (self.info["meta"].groupby(self.target_name)
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )
        dc.insert(0, "location", "Global")
        out = dc[dc["n_isolates"] >= self.threshold].reset_index(drop=True)
        return out

    def _continent_filter(self, continents):
        dc = (self.info["meta"].groupby(["continent", self.target_name])
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        out = dc[(dc["n_isolates"] >= self.threshold) & (dc["continent"].isin(continents))].reset_index(drop=True)
        out = out.rename(columns={"continent": "location"})
        return out

    def _country_filter(self, countries):
        dc = (self.info["meta"].groupby(["country", self.target_name])
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        out = dc[(dc["n_isolates"] >= self.threshold) & (dc["country"].isin(countries))].reset_index(drop=True)
        out = out.rename(columns={"country": "location"})
        return out

    def count_by_global(self):
        total = self._global_total()
        selection = self._global_filter()

        seq_names = selection[self.target_name].to_numpy().tolist()
        seq_names.sort(key=lambda k: int(k[1:]))
        seq_names_mapper = {n: i for i, n in enumerate(seq_names)}

        dt = self.info["meta"][self.info["meta"][self.target_name].isin(seq_names)]

        dc = (dt.groupby([self.target_name, "collection_date"])
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        values = np.zeros(shape=(len(seq_names),
                                 len(self.info["date"])),
                          dtype=np.int32)

        for name, t, v in tqdm(zip(dc[self.target_name], dc["collection_date"], dc["n_isolates"]),
                               desc="[global]"):
            ri = seq_names_mapper[name]
            rj = self.info["date_mapper"][t]
            values[ri, rj] = v

        return dict(total=total, count=values,
                    sequence_names=np.array(seq_names),
                    )

    def count_by_continent(self, a_continent, seq_names):
        c_index = self.info["continent_mapper"][a_continent]

        total = self._continent_total()[c_index]
        # selection = self._continent_filter([a_continent])
        #
        # seq_names = selection[self.target_name].drop_duplicates().to_numpy().tolist()
        # seq_names.sort(key=lambda k: int(k[1:]))
        seq_names_mapper = {n: i for i, n in enumerate(seq_names)}

        flag_1 = self.info["meta"][self.target_name].isin(seq_names)
        flag_2 = self.info["meta"]["continent"] == a_continent
        flag = np.logical_and(flag_1, flag_2)
        dt = self.info["meta"][flag]

        dc = (dt.groupby([self.target_name, "collection_date"])
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        values = np.zeros(shape=(len(seq_names),
                                 len(self.info["date"])),
                          dtype=np.int32)

        for name, t, v in tqdm(zip(dc[self.target_name], dc["collection_date"], dc["n_isolates"]),
                               desc="[continent: %s]" % a_continent):
            ri = seq_names_mapper[name]
            rj = self.info["date_mapper"][t]
            values[ri, rj] = v

        return dict(total=total, count=values,
                    sequence_names=np.array(seq_names),
                    )

    def count_by_country(self, a_country, seq_names):
        c_index = self.info["country_mapper"][a_country]

        total = self._country_total()[c_index]
        # selection = self._country_filter([a_country])
        #
        # seq_names = selection[self.target_name].drop_duplicates().to_numpy().tolist()
        # seq_names.sort(key=lambda k: int(k[1:]))
        seq_names_mapper = {n: i for i, n in enumerate(seq_names)}

        flag_1 = self.info["meta"][self.target_name].isin(seq_names)
        flag_2 = self.info["meta"]["country"] == a_country
        flag = np.logical_and(flag_1, flag_2)
        dt = self.info["meta"][flag]

        dc = (dt.groupby([self.target_name, "country", "collection_date"])
              .aggregate(n_isolates=pd.NamedAgg("epi_id", "count"))
              .reset_index()
              )

        values = np.zeros(shape=(len(seq_names),
                                 len(self.info["date"])),
                          dtype=np.int32)

        for name, t, v in tqdm(zip(dc[self.target_name], dc["collection_date"], dc["n_isolates"]),
                               desc="[country: %s]" % a_country):
            ri = seq_names_mapper[name]
            rj = self.info["date_mapper"][t]
            values[ri, rj] = v

        return dict(total=total, count=values,
                    sequence_names=np.array(seq_names),
                    )

    def candidate_targets(self, _countries, _continents):
        dc1 = self._global_filter()
        dc2 = self._continent_filter(_continents)
        dc3 = self._country_filter(_countries)

        dc = pd.concat([dc1, dc2, dc3], axis=0).reset_index(drop=True)

        return dc

    def run(self, _countries, _continents): # counts by country/continent/global; smooth
        print("Generate Counts")
        results = dict()

        results["Global"] = self.count_by_global()
        seq_names = results["Global"]["sequence_names"]

        for continent in _continents:
            results[continent] = self.count_by_continent(continent, seq_names)
            assert np.all(results[continent]["sequence_names"] == seq_names)

        for country in _countries:
            results[country] = self.count_by_country(country, seq_names)
            assert np.all(results[country]["sequence_names"] == seq_names)

        print("Perform 7-window smooth")
        results_smooth = dict()
        for loc, values in results.items():
            tmp = dict()
            for key, value in values.items():
                if key == "total":
                    tmp[key] = self.week_smooth(value)
                elif key == "count":
                    tmp[key] = np.apply_along_axis(self.week_smooth, 1, value)
                else:
                    tmp[key] = value
            results_smooth[loc] = tmp

        locations = ["Global"] + _continents + _countries

        _total = []
        _count = []
        _total_s = []
        _count_s = []
        for loc in locations:
            _total.append(results[loc]["total"])
            _total_s.append(results_smooth[loc]["total"])
            _count.append(results[loc]["count"])
            _count_s.append(results_smooth[loc]["count"])

        _total = np.stack(_total, axis=0)
        _count = np.stack(_count, axis=0)
        _total_s = np.stack(_total_s, axis=0)
        _count_s = np.stack(_count_s, axis=0)

        msa = self.get_msa(seq_names)
        return dict(day_count=dict(total=_total, count=_count),
                    smooth_day_count=dict(total=_total_s, count=_count_s),
                    msa=msa,
                    location=np.array(locations),
                    sequence_names=seq_names,
                    date=self.info["date"])

    def get_msa(self, seq_names):
        _mapper = {n: 1 for n in seq_names}

        select_seqs = dict()
        with gzip.open(self.uniq_seq_aln, "rt") as handle:
            recorders = SeqIO.parse(handle, "fasta")
            for cur_seq in tqdm(recorders, desc="[search sequences msa]"):
                if cur_seq.id in _mapper:
                    select_seqs[cur_seq.id] = list(str(cur_seq.seq))
        out = []
        for n in seq_names:
            out.append(select_seqs[n])

        return np.array(out)

    @staticmethod
    def week_smooth(x):
        return np.convolve(x, np.ones(7, dtype=np.int32) / 7, "same").astype(np.float32)

    @staticmethod
    def show_total(outputs):
        fig = plt.figure(figsize=(22, 22))
        gs = GridSpec(4, 2)

        i = 0
        for loc, v1, v2 in zip(outputs["location"],
                               outputs["day_count"]["total"],
                               outputs["smooth_day_count"]["total"]
                               ):
            ax = fig.add_subplot(gs[i])
            ax.plot(pd.to_datetime(outputs["date"], format="%Y-%m-%d"), v1, label="raw")
            ax.plot(pd.to_datetime(outputs["date"], format="%Y-%m-%d"), v2, label="smooth")
            ax.set_title(loc)
            ax.xaxis.set_minor_locator(mdates.MonthLocator(np.arange(1, 13)))
            ax.xaxis.set_major_locator(mdates.MonthLocator(np.arange(1, 13, 4)))  # 设置月份刻度
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
            ax.xaxis.set_tick_params(rotation=45)
            i += 1
        return fig

    def show_seq_count_max(self, outputs):
        fig = plt.figure(figsize=(22, 22))
        gs = GridSpec(4, 2)

        i = 0
        for loc, v1, v2 in zip(outputs["location"],
                               outputs["day_count"]["count"],
                               outputs["smooth_day_count"]["count"]
                               ):
            max_i = v1.sum(axis=1).argmax()

            ax = fig.add_subplot(gs[i])
            ax.plot(pd.to_datetime(outputs["date"], format="%Y-%m-%d"), v1[max_i], label="raw")
            ax.plot(pd.to_datetime(outputs["date"], format="%Y-%m-%d"), v2[max_i], label="smooth")

            s_name = outputs["sequence_names"][max_i]
            s_name_m = (self.info["meta"].loc[self.info["meta"][self.target_name] == s_name,
                                              self.target_name + "_mut"].to_numpy()[0])
            ax.set_title("%s %s [%s]" % (loc, s_name_m, s_name))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(np.arange(1, 13)))
            ax.xaxis.set_major_locator(mdates.MonthLocator(np.arange(1, 13, 4)))  # 设置月份刻度
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
            ax.xaxis.set_tick_params(rotation=45)
            i += 1
        return fig

    @staticmethod
    def show_size(outputs, file=None):
        for k, v in outputs.items():
            print(">>> ", k, file=file)
            if isinstance(v, dict):
                for sk, sv in v.items():
                    print("\t size of %s: %s" % (sk, str(sv.shape)), file=file)
                    if sk == "total":
                        print("\t total isolate counts for locations: ", sv.sum(-1).tolist(), file=file)
            else:
                print("\t size of %s: %s" % (k, str(v.shape)), file=file)
                if k == "location":
                    print("\t %d locations: %s" % (len(v), str(v)), file=file)
                if k == "sequence_names":
                    print("\t the first 10 names: ", v[0:10].tolist(), file=file)
                if k == "date":
                    print("\t the first and last date: %s and %s" % (v[0], v[-1]), file=file)


def process_counts(_countries, _continents, meta_file, aln_seq_file, tag, out_dir, threshold=30):
    _worker = DayStat(meta_file, aln_seq_file, tag, threshold)
    outputs = _worker.run(_countries, _continents)

    np.savez(os.path.join(out_dir, "%s_count.npz" % tag),
             total=outputs["day_count"]["total"],
             count=outputs["day_count"]["count"],
             msa=outputs["msa"],
             sequence_names=outputs["sequence_names"],
             location=outputs["location"],
             date=outputs["date"],
             )

    np.savez(os.path.join(out_dir, "%s_count_smooth.npz" % tag),
             total=outputs["smooth_day_count"]["total"],
             count=outputs["smooth_day_count"]["count"],
             msa=outputs["msa"],
             sequence_names=outputs["sequence_names"],
             location=outputs["location"],
             date=outputs["date"],
             )

    fig1 = _worker.show_total(outputs)
    fig1.savefig(os.path.join(out_dir, "%s_total_count.png" % tag), dpi=300,
                 bbox_inches="tight",
                 facecolor="white")
    fig1.clf()

    fig2 = _worker.show_seq_count_max(outputs)
    fig2.savefig(os.path.join(out_dir, "%s_seq_count_sample.png" % tag), dpi=300,
                 bbox_inches="tight",
                 facecolor="white")
    fig2.clf()

    with open(os.path.join(out_dir, "%s_count.log" % tag), "w") as fo:
        _worker.show_size(outputs, file=fo)

    candidates = _worker.candidate_targets(_countries, _continents)

    flag = outputs["day_count"]["count"][0] > 0
    first_date = []
    for f in flag:
        idx = int(np.where(f)[0][0])
        first_date.append(outputs["date"][idx])

    flag1 = outputs["smooth_day_count"]["count"][0] > 0
    first_date1 = []
    for f in flag1:
        idx = int(np.where(f)[0][0])
        first_date1.append(outputs["date"][idx])

    df = pd.DataFrame({_worker.target_name: outputs["sequence_names"],
                       "occurrence_date": first_date,
                       "occurrence_date_sm": first_date1,
                       })
    dm = pd.merge(candidates, df, on=_worker.target_name)
    dm.to_csv(os.path.join(out_dir, "%s_candidates.csv" % tag), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aln_file', type=str, help="uniq_spike1030.aln.gz or uniq_rbd1030.aln.gz")
    parser.add_argument('--meta_file', type=str, help="meta1030.csv.gz")
    parser.add_argument('--tag', type=str, help="spike or rbd")
    parser.add_argument('--threshold', type=int, default=30,
                        help="the min isolates for unique seq, default 30")
    parser.add_argument('--out_dir', type=str)

    opts = parser.parse_args()
    if not os.path.isdir(opts.out_dir):
        os.makedirs(opts.out_dir)

    use_countries = ["USA", "United Kingdom", "Japan"]
    use_continents = ["North America", "Europe", "Asia"]

    process_counts(use_countries,
                   use_continents,
                   opts.meta_file,
                   opts.aln_file,
                   opts.tag,
                   opts.out_dir,
                   opts.threshold)

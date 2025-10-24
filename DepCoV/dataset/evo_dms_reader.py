import gc
from datetime import datetime, timedelta

import numpy as np


class _DMS(object):
    MutantList = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
                           'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

    def __init__(self, dms_npz, tag="spike", ab_escape_cluster='cluster13'):
        self.inputs = np.load(dms_npz, allow_pickle=True)["dms"].item()
        self._check_mutant()

        self._date_fmt = "%Y-%m-%d"
        self._aligned_seqs = self._get_all_sequences()

        assert tag in ["spike", "rbd"]
        self.tag = tag

        # ['cluster13', 'cluster21', 'cluster18', 'cluster29', 'cluster37', 'cluster56']
        self._ab_escape_acceptable_clusters = self.inputs["rbd_antibody_escape"]["cluster_names"].tolist()
        assert ab_escape_cluster in self._ab_escape_acceptable_clusters
        self.ab_escape_cluster = ab_escape_cluster
        print("choose cluster type: %s" % self.ab_escape_cluster)

    def _check_mutant(self):
        for key, values in self.inputs.items():
            if not np.all(values["mutants"] == self.MutantList):
                raise RuntimeError("mutant order not match")

    def _fmt_dms_v1(self, x):
        antigen, antigen_time, feature_name = np.broadcast_arrays(x['antigens'][:, None],
                                                                  x['antigen_dates'][:, None],
                                                                  x['features'][None])

        n_antigen, n_feature, n_antigen, n_site = x['data'].shape
        data_r = x['data'].reshape(-1, n_antigen, n_site)

        return dict(value=data_r,
                    antigen=antigen.reshape(-1),
                    feature_name=np.array(["%s@%s" % (a, f) for a, f in zip(antigen.reshape(-1),
                                                                            feature_name.reshape(-1))]),
                    antigen_time=np.array([datetime.strptime(d, self._date_fmt)
                                           for d in antigen_time.reshape(-1)]
                                          )
                    )

    def fmt_spike_sera_escape(self):
        # drop Delta
        x = self.inputs["spike_sera_escape"]
        idx = np.where(x["antigens"] != "Delta")[0]

        y = dict(features=x["features"],
                 mutants=x["mutants"],
                 antigens=x["antigens"][idx],
                 antigen_dates=x["antigen_dates"][idx],
                 sequences=x["sequences"][idx],
                 data=x["data"][idx])
        return self._fmt_dms_v1(y)

    def fmt_rbd_ace2binding_expression(self):
        x = self.inputs["rbd_ace2binding_expression"]
        return self._fmt_dms_v1(x)

    def fmt_spike_ace2neutralizing_entry(self):
        x = self.inputs["spike_ace2neutralizing_entry"]
        return self._fmt_dms_v1(x)

    def fmt_rbd_antibody_escape(self):
        x = self.inputs["rbd_antibody_escape"]
        # using antibody_selection to filter antibodies, True to kept, False to set as np.nan
        ab_select = np.broadcast_to(x["antibody_selection"][:, None, :, None, None], x["data"].shape)
        data = np.where(ab_select, x["data"], np.nan)

        n_antigen, n_feature, n_antibody, n_site, n_mutant = data.shape

        # average antibody in cluster level
        cluster_idx = self._ab_escape_acceptable_clusters.index(self.ab_escape_cluster)
        ab_cluster_value = x["cluster_values"][cluster_idx]

        uniq_clusters = np.sort(np.unique(ab_cluster_value))

        mean_values = []
        for uc in uniq_clusters:
            cur_flag = ab_cluster_value == uc
            cur_data = data[:, :, cur_flag, :, :]
            _value = np.nanmean(cur_data, axis=2)
            mean_values.append(_value)

        # (n_antigen, n_feature, n_cluster, n_site, n_mutant)
        mean_values = np.stack(mean_values, axis=2)
        mean_values_f = mean_values.reshape((n_antigen, -1, n_site, n_mutant))
        feature_name_b, uniq_clusters_b = np.broadcast_arrays(x['features'][None], uniq_clusters[:, None])

        new_feature_names = []
        for f_name, u_cluster in zip(feature_name_b.reshape(-1), uniq_clusters_b.reshape(-1)):
            new_feature_names.append("%s@%s_%s" % (f_name, self.ab_escape_cluster, u_cluster))

        new_feature_names = np.array(new_feature_names)

        nx = dict(features=new_feature_names,
                  mutants=x["mutants"],
                  antigens=x["antigens"],
                  antigen_dates=x["antigen_dates"],
                  sequences=x["sequences"],
                  data=mean_values_f)
        return self._fmt_dms_v1(nx)

    def _get_all_sequences(self):
        out = dict()
        for key, values in self.inputs.items():
            out.update(dict(zip(values["antigens"], values["sequences"])))
        return out

    def run(self):
        y1 = self.fmt_rbd_ace2binding_expression()
        y2 = self.fmt_spike_ace2neutralizing_entry()

        bind_feature = dict()
        for key in y1.keys():
            bind_feature[key] = np.concatenate([y1[key], y2[key]], axis=0)

        # add ref_seq msa
        bind_msa = np.stack([self._aligned_seqs[ag] for ag in bind_feature["antigen"]], axis=0)
        bind_feature["msa"] = bind_msa

        y3 = self.fmt_spike_sera_escape()
        y4 = self.fmt_rbd_antibody_escape()

        escape_feature = dict()
        for key in y3.keys():
            escape_feature[key] = np.concatenate([y3[key], y4[key]], axis=0)

        escape_msa = np.stack([self._aligned_seqs[ag] for ag in escape_feature["antigen"]], axis=0)
        escape_feature["msa"] = escape_msa

        if self.tag == "rbd":
            # rbd region of alignment region: 337:538
            bind_feature["msa"] = bind_feature["msa"][:, 337:538]
            bind_feature["value"] = bind_feature["value"][:, 337:538, :]

            escape_feature["msa"] = escape_feature["msa"][:, 337:538]
            escape_feature["value"] = escape_feature["value"][:, 337:538, :]

        # pad -, ignore deletion mutation
        bind_feature["value"] = np.pad(bind_feature["value"], ((0, 0), (0, 0), (1, 0)),
                                       mode="constant", constant_values=np.nan)
        escape_feature["value"] = np.pad(escape_feature["value"], ((0, 0), (0, 0), (1, 0)),
                                         mode="constant", constant_values=np.nan)
        mutant = ["-"] + self.MutantList.tolist()
        return dict(bind_feature=bind_feature, escape_feature=escape_feature, mutant=np.array(mutant))


def read_dms(dms_npz, tag="rbd", ab_escape_cluster='cluster13'):
    """

    :param dms_npz:str, path of _dms.npz
    :param tag: str, rbd or spike
    :param ab_escape_cluster: str, one of ['cluster13', 'cluster21', 'cluster18', 'cluster29', 'cluster37', 'cluster56']
    :return:
    """
    _func = _DMS(dms_npz, tag=tag, ab_escape_cluster=ab_escape_cluster)
    out = _func.run()
    del _func
    gc.collect()
    return out


def onehot_array(in_array, alphabet_order=None):
    """

    Args:
        in_array: numpy.ndarray
        alphabet_order:

    Returns:

    """
    alphabet_own = np.unique(in_array).tolist()

    if alphabet_order is None:
        use_order = alphabet_own
    elif isinstance(alphabet_order, list):
        if set(alphabet_own).issubset(set(alphabet_order)):
            use_order = alphabet_order
        else:
            raise RuntimeError("Include Unknown alphabets in input array:",
                               ";".join(set(alphabet_own) - set(alphabet_order)))
    else:
        raise RuntimeError("alphabet_order accept None or a list of letters in in_array")

    mapper = {c: i for i, c in enumerate(use_order)}

    in_flat = np.array(list(map(lambda i: mapper[i], in_array.flatten())))

    n_items = len(use_order)

    _encoder = np.eye(n_items, dtype=np.int32)
    outputs = _encoder[in_flat]
    outputs = outputs.reshape(in_array.shape + (n_items,))
    return outputs, np.array(use_order)


class DMSReader(object):
    def __init__(self, dms_npz, tag="rbd", ab_escape_cluster='cluster13'):
        self.dms = read_dms(dms_npz, tag=tag, ab_escape_cluster=ab_escape_cluster)
        self._seq_len = self.dms["bind_feature"]["msa"].shape[1]

    def _query_features(self, sequences, base_features):
        """

        :param sequences: 1-dim or 2-dim np.ndarray
        :return:
        """
        if isinstance(sequences, np.ndarray):
            if sequences.ndim == 1:
                query_seqs = sequences[None]
            elif sequences.ndim == 2:
                query_seqs = sequences
            else:
                raise RuntimeWarning("sequences must be 1-dim or 2-dim np.ndarray")

            if query_seqs.shape[1] != self._seq_len:
                raise RuntimeWarning("The len of input sequences not match the settings. Please Check")

            # X sites to np.nan
            x_sites = query_seqs == "X"
            # replace X to -
            query_seqs = np.where(x_sites, "-", query_seqs)

            msa = base_features["msa"]
            data = base_features["value"]

            # (n_query, n_ref, n_sites)
            query_seqs_b, _ = np.broadcast_arrays(query_seqs[:, None], msa[None])
            query_oh, _ = onehot_array(query_seqs_b, self.dms["mutant"].tolist())
            query_oh = query_oh.astype(np.float32)
            query_ohm = np.where(query_oh == 0.0, np.nan, 1.0).astype(np.float32)

            v = query_ohm * data[None]
            na_flag = np.all(np.isnan(v), axis=-1)
            sv = np.nansum(v, axis=-1)
            out = np.where(na_flag, np.nan, sv)

            # transpose feature to the last axis
            query_features = np.transpose(out, axes=(0, 2, 1))
            return query_features
        else:
            raise RuntimeWarning("sequences must be 1-dim or 2-dim np.ndarray")

    def query_dms_features(self, query_sequences):
        # for bind features
        bind_ = self._query_features(query_sequences, self.dms["bind_feature"])
        escape_ = self._query_features(query_sequences, self.dms["escape_feature"])

        return dict(dms_features_bind=bind_,
                    dms_features_bind_names=self.dms["bind_feature"]["feature_name"],
                    dms_features_bind_times=self.dms["bind_feature"]["antigen_time"],
                    dms_features_escape=escape_,
                    dms_features_escape_names=self.dms["escape_feature"]["feature_name"],
                    dms_features_escape_times=self.dms["escape_feature"]["antigen_time"],
                    )


class DMSAccessor(object):
    def __init__(self, dms_cluster_npz, base_dms_npz, tag, cluster, dms_delay=0):
        """

        :param dms_cluster_npz: str
            /lustre/grp/cyllab/share/ljj/public/dms1030/dms_results/rbd/dms_cluster18.npz,
        :param base_dms_npz: str,
            /lustre/grp/cyllab/share/ljj/public/dms1030/merge/_dms.npz
        :param tag: str,
            spike or rbd
        :param cluster: str
            one of ['cluster13', 'cluster21', 'cluster18', 'cluster29', 'cluster37', 'cluster56']
        :param dms_delay: int, default 0
            delay days to consider DMS features
        """
        inputs = np.load(dms_cluster_npz, allow_pickle=True)
        self.bind = inputs["dms_features_bind"]
        self.bind_names = inputs["dms_features_bind_names"]
        self.bind_times = inputs["dms_features_bind_times"]
        self.escape = inputs["dms_features_escape"]
        self.escape_names = inputs["dms_features_escape_names"]
        self.escape_times = inputs["dms_features_escape_times"]
        self.delay_delta_days = timedelta(days=dms_delay)

        self._date_fmt = "%Y-%m-%d"

        # only use when sequences_indexes = -1 for target sequence
        self.dms_reader = DMSReader(base_dms_npz, tag=tag,
                                    ab_escape_cluster=cluster)

        self.n_bind_features = len(self.bind_names)
        self.n_escape_features = len(self.escape_names)

    def _feature_by_indexes(self, indexes):
        bind_value = self.bind[indexes]
        escape_value = self.escape[indexes]
        return bind_value, escape_value

    def _feature_by_sequence(self, sequence):
        res = self.dms_reader.query_dms_features(sequence)
        return res["dms_features_bind"], res["dms_features_escape"]

    def query_dms_features(self, bg_indexes, target_seq, t0):
        """
        训练的时候，可以用target_sequence_index获取序列
        对于生成的序列，直接传序列
        :param bg_indexes: list or 1-d np.ndarray, index for background sequences
        :param target_seq: 1-d np.ndarray, aligned target sequence
        :param t0: str, "YYYY-MM-DD" format
        :return:
        dms_features_bind：contain nan value, (n_seq, n_site, n_feature)
        dms_features_bind_mask：(n_feature,), true for features that can be used, after time filtering
        dms_features_escape：contain nan value, (n_seq, n_site, n_feature)
        dms_features_escape_mask (n_feature,), true for features that can be used after time filtering

        """
        t0_ = datetime.strptime(t0, self._date_fmt)

        target_bind, target_escape = self._feature_by_sequence(target_seq)
        bg_bind, bg_escape = self._feature_by_indexes(bg_indexes)

        out_bind = np.concatenate([target_bind, bg_bind], axis=0)
        out_escape = np.concatenate([target_escape, bg_escape], axis=0)

        bind_mask = self.bind_times < t0_ - self.delay_delta_days
        escape_mask = self.escape_times < t0_ - self.delay_delta_days

        bind_mask_out = np.broadcast_to(bind_mask[None, None], out_bind.shape).astype(np.int32)
        escape_mask_out = np.broadcast_to(escape_mask[None, None], out_escape.shape).astype(np.int32)

        return dict(dms_features_bind=out_bind,
                    dms_features_bind_mask=bind_mask_out,
                    dms_features_escape=out_escape,
                    dms_features_escape_mask=escape_mask_out)

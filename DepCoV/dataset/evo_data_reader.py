import numpy as np


class CountReader(object):
    def __init__(self, count_npz):
        inputs = np.load(count_npz)
        self._total = inputs["total"]
        self._count = inputs["count"]
        self._msa = inputs["msa"]
        self._sequence_names = inputs["sequence_names"]
        self._location = inputs["location"]
        self._date = inputs["date"]

        self._loc_mapper, self._date_mapper, self._sequence_names_mapper = self._all_mapper()
        self._msa_length = self._msa.shape[-1]

    def _all_mapper(self):
        loc_mapper = {l: i for i, l in enumerate(self._location)}
        date_mapper = {d: i for i, d in enumerate(self._date)}
        seq_name_mapper = {n: i for i, n in enumerate(self._sequence_names)}
        return loc_mapper, date_mapper, seq_name_mapper

    def index2name(self, tag, index):
        """
        convert indexes of location/date/sequence names to actual names
        :param tag: str, one of ["location", "date", "sequence_name"]
        :param index: int, or a list of int
        :return:
        """
        assert isinstance(index, (int, list))

        if tag == "location":
            return self._location[index]
        elif tag == "date":
            return self._date[index]
        elif tag == "sequence_names":
            return self._sequence_names[index]
        else:
            raise RuntimeError('tag must be one of  ["location", "date", "sequence_name"], but got %s' % tag)

    def name2index(self, tag, name):
        """
        convert name of location/date/sequence_names to index
        :param tag: str, one of ["location", "date", "sequence_name"]
        :param name: str
        :return:
        """
        assert isinstance(name, str)
        if tag == "location":
            return self._loc_mapper[name]
        elif tag == "date":
            return self._date_mapper[name]
        elif tag == "sequence_name":
            return self._sequence_names_mapper[name]
        else:
            raise RuntimeError('tag must be one of  ["location", "date", "sequence_name"], but got %s' % tag)

    def seq_count_by_date(self, loc, seq_name, ta, tb=None):
        """
        get count of sequence by location, sequence_names at ta or between [ta, tb]
        :param loc: str,
        :param seq_name: str
        :param ta: str
        :param tb: str, or None, default None to return count at ta
        :return:
        """
        loc_i = self.name2index("location", loc)
        name_i = self.name2index("sequence_name", seq_name)
        ta_i = self.name2index("date", ta)

        if tb is None:
            return self._count[loc_i, name_i, ta_i]
        else:
            tb_i = self.name2index("date", tb)
            if tb_i > ta_i:
                return self._count[loc_i, name_i, ta_i: tb_i + 1]
            elif tb_i == ta_i:
                return self._count[loc_i, name_i, ta_i]
            else:
                raise RuntimeError("date of tb must be after ta")

    def total_count_by_date(self, loc, ta, tb=None):
        """
        get total count by location at ta or between [ta, tb]
        :param loc: str,
        :param ta: str, date
        :param tb: str, or None, date, default None to return count at ta
        :return:
        """
        loc_i = self.name2index("location", loc)
        ta_i = self.name2index("date", ta)

        if tb is None:
            return self._total[loc_i, ta_i]
        else:
            tb_i = self.name2index("date", tb)
            if tb_i > ta_i:
                return self._total[loc_i, ta_i: tb_i + 1]
            elif tb_i == ta_i:
                return self._total[loc_i, ta_i]
            else:
                raise RuntimeError("date of tb must be after ta")

    def query_background(self, loc, t0, top_k, n_bg_days=180, stride=3, shuffle=True):
        """
        query background count between [t0-n_bg_days, t0), if background is not enough, pad 0
        :param loc: str,
        :param t0:
        :param top_k: int, return top_k count of sequences
        :param n_bg_days: int, default 180 days
        :param stride: int, default 3
        :param shuffle: bool, default True to shuffle the order of top_k sequences.
        :return:
        """
        t0_i = self.name2index("date", t0)
        loc_i = self.name2index("location", loc)
        ta_i = t0_i - n_bg_days

        if ta_i >= 0:
            cur_total = self._total[loc_i, ta_i:t0_i]
            seq_count = self._count[loc_i, :, ta_i:t0_i]
        else:
            # pad 0.0
            n_pad = abs(ta_i)
            part_total = self._total[loc_i, 0:t0_i]
            part_count = self._count[loc_i, :, 0:t0_i]
            cur_total = np.pad(part_total, (n_pad, 0), mode="constant", constant_values=0)
            seq_count = np.pad(part_count, ((0, 0), (n_pad, 0)), mode="constant", constant_values=0)

        _total = np.where(cur_total == 0, 1, cur_total)
        seq_ratio = seq_count / _total[None]

        # sum by sequences
        seq_count_s = seq_count.sum(axis=1)

        # reversed sort, and top_k
        sort_indexes = np.argsort(seq_count_s * -1)
        top_k_indexes = sort_indexes[0: top_k].copy()
        if shuffle:
            np.random.shuffle(top_k_indexes)

        out_total = cur_total[::-stride][::-1]
        select_seq_count = seq_count[top_k_indexes]
        select_seq_ratio = seq_ratio[top_k_indexes]

        out_seq_count = select_seq_count[:, ::-stride][:, ::-1]
        out_seq_ratio = select_seq_ratio[:, ::-stride][:, ::-1]
        out_msa = self._msa[top_k_indexes]

        return dict(total=out_total,
                    count=out_seq_count,
                    ratio=out_seq_ratio,
                    indexes=top_k_indexes,
                    msa=out_msa,
                    t0_index=t0_i,
                    location_index=loc_i,
                    )

    def query_target(self, loc, t0, query_seq, n_bg_days=180, stride=3):
        """

        :param loc:
        :param t0:
        :param query_seq: str or 1-dim numpy.ndarray, aligned seq, length need to be equal to the msa
        :param n_bg_days:
        :param stride:
        :return:

        if query_seq not in the MSA, returned index will be -1
        """
        if isinstance(query_seq, str):
            if len(query_seq) == self._msa_length:
                q_seq = np.array(list(query_seq))
            else:
                raise RuntimeError("Length of query_seq is not match with the MSA")
        elif isinstance(query_seq, np.ndarray):
            if query_seq.ndim == 1 and len(query_seq) == self._msa_length:
                q_seq = query_seq
            else:
                raise RuntimeError("Length of query_seq is not match with the MSA")
        else:
            raise RuntimeError("Not acceptable query_seq")

        flag = np.all(self._msa == q_seq[None], axis=1)
        _indexes = np.where(flag)[0]

        if len(_indexes) == 1:
            q_idx = _indexes[0]
            t0_i = self.name2index("date", t0)
            loc_i = self.name2index("location", loc)
            ta_i = t0_i - n_bg_days

            if ta_i >= 0:
                cur_total = self._total[loc_i, ta_i:t0_i]
                seq_count = self._count[loc_i, q_idx, ta_i:t0_i]
            else:
                # pad 0.0
                n_pad = abs(ta_i)
                part_total = self._total[loc_i, 0:t0_i]
                part_count = self._count[loc_i, q_idx, 0:t0_i]
                cur_total = np.pad(part_total, (n_pad, 0), mode="constant", constant_values=0)
                seq_count = np.pad(part_count, (n_pad, 0), mode="constant", constant_values=0)

            _total = np.where(cur_total == 0, 1, cur_total)
            seq_ratio = seq_count / _total
        elif len(_indexes) > 1:
            raise RuntimeError("Check the input MSA please, duplicates exist.")
        else:
            # fill 0.0
            q_idx = -1
            seq_count = np.zeros(n_bg_days, dtype=self._count.dtype)
            seq_ratio = np.zeros(n_bg_days, dtype=np.float32)

        out_seq_count = seq_count[::-stride][::-1]
        out_seq_ratio = seq_ratio[::-stride][::-1]
        return dict(count=out_seq_count,
                    ratio=out_seq_ratio,
                    index=q_idx,
                    sequence=q_seq)

    def query_sample_by_name(self, loc, t0, target_name, top_k, n_bg_days=180, stride=3, shuffle=True):
        """
        For Training samples
        :param loc:
        :param t0:
        :param target_name:
        :param top_k:
        :param n_bg_days:
        :param stride:
        :param shuffle:
        :return:
        """
        name_i = self.name2index("sequence_name", target_name)
        target_seq = self._msa[name_i]
        return self.query_sample_by_seq(loc, t0, target_seq, top_k, n_bg_days, stride, shuffle)

    def query_sample_by_seq(self, loc, t0, target_seq, top_k, n_bg_days=180, stride=3, shuffle=True):
        """
        For generated samples, also can be used to train of course.
        :param loc: str, location
        :param t0: str, t0 date
        :param target_seq: str or 1-dim numpy.ndarray, aligned seq, length need to be equal to the msa
        :param top_k: int,
        :param n_bg_days:int, background days
        :param stride: int, background stride
        :param shuffle:bool,
        :return:
        """

        target_values = self.query_target(loc, t0, target_seq, n_bg_days, stride)
        background_values = self.query_background(loc, t0, top_k, n_bg_days, stride, shuffle)

        out = dict(count=np.concatenate([target_values["count"][None],
                                         background_values["count"]], axis=0),
                   total=background_values["total"],
                   ratio=np.concatenate([target_values["ratio"][None],
                                         background_values["ratio"]], axis=0),
                   msa=np.concatenate([target_values["sequence"][None],
                                       background_values["msa"]], axis=0),
                   target_sequence_index=target_values["index"],
                   t0_index=background_values["t0_index"],
                   location_index=background_values["location_index"],
                   background_sequence_indexes=background_values["indexes"],
                   )
        return out

    def get_t1_values(self, loc_index, seq_index, t0_index, min_total_isolates_t1=100.0):
        """
        only for sequence in *_count_sm.npz
        :param loc_index:
        :param seq_index:
        :param t0_index:
        :param min_total_isolates_t1:
        :return:

        1-60 days
        target_isolates_t1
        total_isolates_t1
        target_ratio_t1
        target_ratio_t1_mask: int32, 1 for confident target_ratio_t1
        """
        total_isolates_t1 = self._total[loc_index, t0_index+1: t0_index+61]
        target_ratio_t1_mask = (total_isolates_t1 > min_total_isolates_t1).astype(np.int32)
        target_isolates_t1 = self._count[loc_index, seq_index, t0_index+1: t0_index+61]

        _total = np.where(total_isolates_t1 == 0.0, 1.0, total_isolates_t1)
        target_ratio_t1 = target_isolates_t1 / _total

        return dict(target_isolates_t1=target_isolates_t1,
                    total_isolates_t1=total_isolates_t1,
                    target_ratio_t1=target_ratio_t1,
                    target_ratio_t1_mask=target_ratio_t1_mask)


if __name__ == "__main__":
    # methods
    # index2name
    # name2index
    # seq_count_by_date
    # total_count_by_date
    # query_background
    # query_target
    # query_sample_by_seq
    # query_sample_by_name
    in_file = "/Users/daolu/Desktop/spike1030/seq_results/rbd/rbd_count_smooth.npz"
    func = CountReader(in_file)

    # use your sequence
    s = func._msa[0]
    a = func.query_sample_by_seq("Asia", "2023-01-01", s, 16)
    b = func.query_sample_by_name("Asia", "2023-01-01", "r0", 16)

    # t1 value
    out = func.get_t1_values(0, 0, 54)

"""
@Author: Luo Jiejian
@Date: 2024/11/7

# 180 days background stat for all t0
"""
import argparse

import numpy as np

# NOTES：
# 在不同的location 层面，计算180天内每天的序列数，cluster数（序列数非0）；180天内的序列总数；各cluster每天的序列数，序列累积数

class BackgroundStat(object):
    def __init__(self, count_npz_sm, bg_n_days):
        """
        :param count_npz_sm:
        :param bg_n_days:
        """

        self.bg_n_days = bg_n_days
        self.inputs = np.load(count_npz_sm)
        self.count = self.inputs["count"]
 
    def _win_sum(self, x):# 滑动窗口求和 window sums
        return np.convolve(x, np.ones(self.bg_n_days, dtype=np.int32), "valid") 

    def stat_background(self):
        count_p = np.pad(self.count, ((0, 0), (0, 0), (self.bg_n_days, 0)), # (before location,after location),(before cluster,after cluster),(before date, after pad)
                         mode="constant", constant_values=0)[:, :, 0:-1] # 对于起始的天数，会凑不够180天(只pad before date)

        # (n_locations, n_uniq_seq, n_dates): isolate count of uniq sequences in background windows. [t0-180, t0)
        bg_win_count = np.apply_along_axis(self._win_sum, 2, count_p) # 在不同location层面，每180天算和

        # The number of unique sequences (clusters) and total isolates in backgroun windows.[t0-180, t0)
        # (n_locations, n_dates)
        n_bg_isolates_win = bg_win_count.sum(axis=1)
        n_bg_clusters_win = (bg_win_count > 0).sum(axis=1) # 序列数 = 非0元素的个数

        # isolate count of uniq sequences for all the time in the past.[2019-12-30, t0)
        count_p1 = np.pad(self.count, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)[:, :, 0:-1]

        # (n_locations, n_uniq_seq, n_dates)
        n_isolates_all = np.cumsum(count_p1, axis=-1)

        return dict(n_isolates_win=bg_win_count.astype(np.float32),
                    n_bg_isolates_win=n_bg_isolates_win.astype(np.float32),
                    n_bg_clusters_win=n_bg_clusters_win.astype(np.int32),
                    n_isolates_all=n_isolates_all.astype(np.float32),
                    sequence_names=self.inputs["sequence_names"],
                    location=self.inputs["location"],
                    date=self.inputs["date"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count_npz_sm', type=str, help="*_count_smooth.npz")
    parser.add_argument('--bg_days', type=int, default=180, help="background days, default 180")
    parser.add_argument('--outfile', type=str)

    opts = parser.parse_args()

    func = BackgroundStat(count_npz_sm=opts.count_npz_sm, bg_n_days=opts.bg_days)
    outputs = func.stat_background()

    np.savez(opts.outfile, **outputs)

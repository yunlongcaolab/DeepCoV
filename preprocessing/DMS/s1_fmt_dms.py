"""
@Author: Luo Jiejian
@Date: 2024/11/9
"""
import argparse
import os

import numpy as np

from dms.dms_func import FormatReader, ArrayDMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--public_dms_dir', type=str,
                        default="/lustre/grp/cyllab/share/ljj/public/dms_data/origin_public_DMS",
                        help='directory for origin public DMS')
    parser.add_argument('--lab_dms_dir', type=str,
                        default="/lustre/grp/cyllab/share/ljj/public/dms_data/DMS_20240717",
                        help="path of Cao lab DMS directory")
    parser.add_argument('--ref_aln', type=str,
                        help='isolates_manual_1030.aln')
    parser.add_argument('--isolate_time_csv', type=str,
                        help="isolate_time_table.csv")
    parser.add_argument('--out_dir', type=str, required=True,
                        help='directory to save process results')

    opts = parser.parse_args()

    if not os.path.isdir(opts.out_dir):
        os.makedirs(opts.out_dir)

    print("Format DMS")
    worker = FormatReader(public_dms_dir=opts.public_dms_dir,
                          lab_dms_dir=opts.lab_dms_dir)

    out_1 = worker.read_sera_escape() # concat delta, xbb15 sera escape
    out_1_file = os.path.join(opts.out_dir, "spike_sera_escape.csv")
    out_1.to_csv(out_1_file, index=False)

    out_2 = worker.read_ace2_binding_rbd_expression() # 做一下rename，词替换和格式转换；检查一下氨基酸，数据格式等等（mutant多一个‘-’）
    out_2_file = os.path.join(opts.out_dir, "rbd_ace2binding_expression.csv")
    out_2.to_csv(out_2_file, index=False)

    out_3 = worker.read_ace2_neutralizing_entry() # 合并BA2和XBB15数据；做一下词替换和格式转换，增加antigen列；检查一下氨基酸，数据格式等等（mutant多一个‘-’）
    out_3_file = os.path.join(opts.out_dir, "spike_ace2neutralizing_entry.csv")
    out_3.to_csv(out_3_file, index=False)

    out_4 = worker.read_antibody_escape()
    out_4_file = os.path.join(opts.out_dir, "rbd_antibody_escape.csv.gz") # antigen source 更换名称;latest_virus_of_source 对于抗体source的简写；cluster包括cluster13	cluster21	cluster18	cluster29	cluster37	cluster56
    out_4.to_csv(out_4_file, index=False)

    ###
    print("Array DMS")
    worker2 = ArrayDMS(opts.ref_aln, opts.isolate_time_csv)

    results = dict()
    results["spike_sera_escape"] = worker2.array_spike_sera_escape(out_1_file) # 序列编号做align处理，选出关注的列和aa，降成2维array，与其他meta信息一起输出（1280）
    results["rbd_ace2binding_expression"] = worker2.array_rbd_ace2binding_expression(out_2_file)
    results["spike_ace2neutralizing_entry"] = worker2.array_spike_ace2neutralizing_entry(out_3_file)
    results["rbd_antibody_escape"] = worker2.array_rbd_antibody_escape(out_4_file) # 挺慢的

    npz_out = os.path.join(opts.out_dir, "_dms.npz")
    np.savez(npz_out, dms=results) # 存储了csv格式，也存储了npz格式

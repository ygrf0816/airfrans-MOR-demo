# -*- coding: utf-8 -*-
"""把 11 个反演目标汇总成一张小表（含"训练集内 / 官方留出集"分组），供报告与复现引用。

产物：04_results/inversion/results/s6_targets_all.csv
列：name,u_inf,aoa,l,p,q,t,cl_official,group,cp_relL2,cp_true_relL2,rec_l,rec_p,rec_q,rec_t
"""
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "04_results", "inversion", "results")
man = json.load(open(os.path.join(ROOT, "03_code", "probe", "manifest.json"), encoding="utf-8"))
tr, te = set(man["full_train"]), set(man["full_test"])
oz = np.load(os.path.join(ROOT, "04_results", "prediction", "official_coefs_1000.npz"),
             allow_pickle=True)
cl = {str(n): float(c) for n, c in zip(oz["names"], oz["CL"])}

rows = []
for p in sorted(glob.glob(os.path.join(RES, "invert_*.npz"))):
    z = np.load(p, allow_pickle=True)
    tp = np.asarray(z["true_params"], float)
    rec = np.asarray(z["best_params"], float)
    # 真名要从 JSON 拿（npz 文件名被截断到 32 字符）
    j = p[:-4] + ".json"
    name = (json.load(open(j, encoding="utf-8"))["target"] if os.path.isfile(j)
            else os.path.basename(p)[:-4].replace("invert_", ""))
    f = name.replace("airFoil2D_SST_", "").split("_")
    rows.append({
        "name": name, "u_inf": f[0], "aoa": f[1],
        "l": f[2], "p": f[3], "q": f[4], "t": f[5],
        "cl_official": "%.4f" % cl.get(name, float("nan")),
        "group": "heldout" if name in te else ("train" if name in tr else "?"),
        "cp_relL2": "%.4f" % float(z["best_cp_relL2"]),
        "cp_true_relL2": "%.4f" % float(z["cp_true_relL2"]),
        "rec_l": "%.3f" % rec[0], "rec_p": "%.3f" % rec[1],
        "rec_q": "%d" % round(rec[2]), "rec_t": "%.3f" % rec[3],
    })
rows.sort(key=lambda r: (r["group"], r["name"]))
out = os.path.join(RES, "s6_targets_all.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("目标 %d 个：留出集 %d，训练集内 %d"
      % (len(rows), sum(r["group"] == "heldout" for r in rows),
         sum(r["group"] == "train" for r in rows)))
for r in rows:
    print("  %-6s %-34s cp=%s 地板=%s 反演(L,P,Q,T)=%s,%s,%s,%s"
          % (r["group"], r["name"][:34], r["cp_relL2"], r["cp_true_relL2"],
             r["rec_l"], r["rec_p"], r["rec_q"], r["rec_t"]))
print("->", out)

# -*- coding: utf-8 -*-
"""导出八个案例的全精度压力曲线与期望指标，给前端口径自检用。

为什么需要这一份：results.json 里的压力曲线保留到小数点后四位，
而分离代理的判据是梯度小于 0.05，一个格宽约 0.007874，
四位小数带来的梯度量化步长约 0.0127，与阈值同量级，
所以用取整后的曲线算分离代理会在临界点上翻一个格。
要判定前端移植对不对，必须拿全精度曲线比。

产物写到 app/scripts/fixtures/cases_cp_precise.json，只是开发用的核对基准，
不放进 public/data，不会进交付站点。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "04_results", "inversion", "results")
OUT = os.path.join(ROOT, "04_results", "demo", "app", "scripts", "fixtures")

# 期望指标直接取自已落盘的 CSV，是权威值
import csv  # noqa: E402


def read_csv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


cases = []
for fname, tag in (("cases_eval_s8.csv", "ga"), ("cases_eval_db.csv", "db")):
    for r in read_csv(os.path.join(RES, fname)):
        cases.append({"tag": tag, "case": int(r["case"]),
                      "L": float(r["L"]), "P": float(r["P"]),
                      "Q": int(float(r["Q"])), "T": float(r["T"]),
                      "cl4": float(r["CL4"]), "cl8": float(r["CL8"]),
                      "cl10": float(r["CL10"]),
                      "machMax": float(r["m_max"]), "sepMax": float(r["sep"])})

curves = {}
for fname, tag in (("cases_eval_s8.npz", "ga"), ("cases_eval_db.npz", "db")):
    z = np.load(os.path.join(RES, fname), allow_pickle=True)
    for i in range(len(z["params"])):
        curves["%s-%d" % (tag, i + 1)] = {
            "cp4": [float(v) for v in np.asarray(z["cp4"][i], float)],
            "cp8": [float(v) for v in np.asarray(z["cp8"][i], float)],
            "cp10": [float(v) for v in np.asarray(z["cp10"][i], float)],
        }

os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "cases_cp_precise.json")
json.dump({"cases": cases, "curves": curves,
           "note": ("压力曲线是全精度，期望指标取自 cases_eval_*.csv。"
                    "仅供 app/scripts/check-proxies.mjs 核对前端移植，不随站点发布。")},
          open(dst, "w", encoding="utf-8"), ensure_ascii=False)
print("-> %s  %d 个案例  %.1f KB" % (dst, len(cases), os.path.getsize(dst) / 1024))
for c in cases:
    print("  %s-%d  machMax %.4f  sepMax %.6f  CL8 %.4f"
          % (c["tag"], c["case"], c["machMax"], c["sepMax"], c["cl8"]))

# -*- coding: utf-8 -*-
"""把三个种子的落盘产物汇总成两张小表（本地跑，不需要 torch / GPU）。

产物（都在 04_results/inversion/results/）：
  seed_summary.csv    每个种子一行：最优可行 CL8/参数/Δ/m/分离代理、前沿规模与去重后规模、
                      第几代稳定、修正口径超体积首末值
  hv_corrected.csv    每个种子逐快照的**修正口径**超体积（history 里的旧列方向反了，见
                      04_results/inversion/03_weight_verify.md §3）

用法：python 03_code/inversion/summarize_seeds.py
"""
import csv
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "04_results", "inversion", "results")
REF = [0.5, 1.5]
TAGS = ["_s42", "_s7", "_s2024"]


def hv2d(F, ref=REF):
    """正确的二维超体积（最小化）：剔被支配点，按 f1 升序累加条带面积。"""
    F = np.asarray(F, float)
    F = F[np.all(F <= ref, axis=1)]
    if len(F) == 0:
        return 0.0
    nd = [p for p in F if not np.any(np.all(F <= p, axis=1) & np.any(F < p, axis=1))]
    pts = np.array(nd)[np.argsort(np.array(nd)[:, 0])]
    hv = 0.0
    for i, (f1, f2) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
        hv += (nxt - f1) * (ref[1] - f2)
    return float(hv)


def main():
    rows, hvrows = [], []
    for tag in TAGS:
        fcsv = os.path.join(RES, "ga_front%s.csv" % tag)
        if not os.path.isfile(fcsv):
            print("缺 %s，跳过" % fcsv)
            continue
        fr = [r for r in csv.DictReader(open(fcsv, encoding="utf-8"))
              if r["feasible"] == "True"]
        best = max(fr, key=lambda r: float(r["CL8"]))
        uniq = {tuple(round(float(r[k]), 9) for k in ("L", "P", "Q", "T")) for r in fr}
        hist = list(csv.DictReader(open(os.path.join(RES, "ga_history%s.csv" % tag),
                                       encoding="utf-8")))
        bf = np.array([float(r["best_feasible_CL8"]) for r in hist])
        final = bf[-1]
        conv = int(hist[int(np.argmax(bf >= 0.999 * final))]["gen"])
        snap = os.path.join(RES, "ga_snaps%s.npz" % tag)
        gs, hvs = [], []
        if os.path.isfile(snap):
            z = np.load(snap, allow_pickle=True)
            for k in range(len(z["gens"])):
                F, G = z["F%d" % k], z["G%d" % k]
                ok = np.all(G <= 0, axis=1)
                hv = hv2d(F[ok] if ok.any() else F)
                gs.append(int(z["gens"][k]))
                hvs.append(hv)
                hvrows.append({"tag": tag.strip("_"), "gen": int(z["gens"][k]),
                               "hv_corrected": round(hv, 6)})
        rows.append({
            "tag": tag.strip("_"), "front_rows": len(fr), "unique_params": len(uniq),
            "best_CL8": "%.10f" % float(best["CL8"]),
            "L": "%.4f" % float(best["L"]), "P": "%.4f" % float(best["P"]),
            "Q": int(float(best["Q"])), "T": "%.4f" % float(best["T"]),
            "delta": "%.4f" % float(best["delta"]), "m_max": "%.4f" % float(best["m_max"]),
            "sep": "%.6f" % float(best["sep"]),
            "cl4": "%.4f" % float(best["CL4"]), "cl10": "%.4f" % float(best["CL10"]),
            "n_feasible_gen0": int(hist[0]["n_feasible"]),
            "gen_stable_999": conv,
            "hv_first": round(hvs[0], 6) if hvs else "",
            "hv_last": round(hvs[-1], 6) if hvs else "",
            "P_at_bound": sum(1 for r in fr if abs(float(r["P"]) - 7.969) < 1e-9),
            "L_gt_3.9": sum(1 for r in fr if float(r["L"]) > 3.9),
            "beat_db_best": sum(1 for r in fr if float(r["CL8"]) > 1.1464),
        })
    with open(os.path.join(RES, "seed_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "hv_corrected.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "gen", "hv_corrected"])
        w.writeheader()
        w.writerows(hvrows)
    print("=== seed_summary.csv ===")
    for r in rows:
        print("  %-8s CL8=%s Δ=%s sep=%s 前沿 %d 行/%d 唯一  第 %s 代稳定  超体积 %s→%s"
              % (r["tag"], r["best_CL8"], r["delta"], r["sep"], r["front_rows"],
                 r["unique_params"], r["gen_stable_999"], r["hv_first"], r["hv_last"]))
    print("-> results/seed_summary.csv, results/hv_corrected.csv")


if __name__ == "__main__":
    main()

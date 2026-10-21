# -*- coding: utf-8 -*-
"""本地独立核对（不需要 torch）：确认几何口径与"Cp 积分得 CL"的权重口径。

为什么要在本地再核一遍：整条第二阶段链路把 CL 全部建立在
`gen_geometry`（官方生成器 -> 128 维几何）和 `build_weight_256`（Cp -> CL 线性权重）上。
这两件事只用 numpy + 官方 `naca_generator.py` 就能算，所以可以在本机独立复核，
不用相信远端脚本的自述。

做法：把 torch / train_B_transformer / input_side 替换成空壳后**直接 import 远端用的同一个
`ga_design.py`**，调用它的真函数（而不是重写一份，避免口径漂移）。

输出：打印 + `04_results/inversion/03_weight_verify.csv`
数据：`09_tmp/data/{geomvec_1000.npz,wallcp_1000.npz}`（从工作站拉回）+ `04_results/prediction/official_coefs_1000.npz`
用法：python 03_code/inversion/verify_weight_local.py
"""
import csv
import os
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "09_tmp", "data")
OUT = os.path.join(ROOT, "04_results", "inversion")


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# 让 ga_design 能在没有 torch / 预测模块的机器上被 import
_stub("torch")
_stub("train_B_transformer", BBackbone=object)
_stub("input_side", input_vector=None)
sys.path.insert(0, os.path.join(ROOT, "01_initial", "AirfRANS-main"))
sys.path.insert(0, os.path.join(ROOT, "03_code", "inversion"))
import ga_design as G  # noqa: E402


def load():
    zv = np.load(os.path.join(DATA, "geomvec_1000.npz"), allow_pickle=True)
    wz = np.load(os.path.join(DATA, "wallcp_1000.npz"), allow_pickle=True)
    oz = np.load(os.path.join(ROOT, "04_results", "prediction", "official_coefs_1000.npz"),
                 allow_pickle=True)
    return ({str(n): i for i, n in enumerate(zv["names"])}, zv,
            {str(n): i for i, n in enumerate(wz["names"])}, wz,
            {str(n): i for i, n in enumerate(oz["names"])}, oz)


def main():
    gi, zv, wi, wz, oi, oz = load()
    names5, names4 = [], []
    for n in zv["names"]:
        f = str(n).replace("airFoil2D_SST_", "").split("_")
        (names5 if len(f) == 6 else names4).append(str(n))
    print("样本数：5 位族 %d，4 位族 %d" % (len(names5), len(names4)))

    # ---------- 1) 几何口径：官方生成器 + 余弦站位 vs 数据集 128 维几何 ----------
    rows, dmax_all = [], []
    for n in (names5[:12] + names4[:8]):
        f = str(n).replace("airFoil2D_SST_", "").split("_")
        pr = [float(x) for x in f[2:]]
        if len(pr) == 4:
            g = G.gen_geometry(pr)
        else:                     # 4 位族 (M,P,T) -> 生成器 3 参数调用
            g = G.gen_geometry_4(pr) if hasattr(G, "gen_geometry_4") else None
        gd = np.asarray(zv["G"][gi[n]], float).ravel()
        if g is None:
            continue
        d = np.abs(g - gd)
        dmax_all.append(d.max())
        rows.append({"kind": "geometry", "name": n, "n_param": len(pr),
                     "max_abs_diff": round(float(d.max()), 6),
                     "mean_abs_diff": round(float(d.mean()), 6)})
    print("几何核对（%d 个样本）：max|diff| 最大 %.2e，中位 %.2e"
          % (len(dmax_all), max(dmax_all), float(np.median(dmax_all))))

    # ---------- 2) 升力口径：w(params,AoA)·Cp_true vs 官方 CL ----------
    rel = []
    for n in names5:
        f = str(n).replace("airFoil2D_SST_", "").split("_")
        L, P, Q, T = [float(x) for x in f[2:]]
        aoa = float(f[1])
        cp = np.asarray(wz["CP"][wi[n]], float).ravel()
        cl_w = float(np.dot(G.build_weight_256((L, P, Q, T), aoa), cp))
        cl_o = float(oz["CL"][oi[n]])
        rel.append((cl_w - cl_o) / cl_o)
        if len(rel) <= 5:
            rows.append({"kind": "lift", "name": n, "n_param": 4,
                         "cl_weight": round(cl_w, 4), "cl_official": round(cl_o, 4),
                         "rel_pct": round(100 * (cl_w - cl_o) / cl_o, 3)})
    rel = np.array(rel)
    print("升力口径（%d 个 5 位族样本）：平均偏差 %+.2f%%  中位 %+.2f%%  |偏差| 中位 %.2f%%  "
          "p95 |偏差| %.2f%%  max %.2f%%"
          % (len(rel), 100 * rel.mean(), 100 * np.median(rel),
             100 * np.median(np.abs(rel)), 100 * np.quantile(np.abs(rel), 0.95),
             100 * np.abs(rel).max()))

    # ---------- 3) 闭环/前缘锚定的结构核对（F1/F2/F6 的口径） ----------
    print("wallcp['S'] 前 3 个 =", np.round(np.asarray(wz["S"]).ravel()[:3], 4),
          "（若与 CP 的 POD 奇异值一致，说明 S 不是坐标）")
    cp0 = np.asarray(wz["CP"][0], float).ravel()
    c = cp0 - cp0.mean()
    sv = np.linalg.svd((np.asarray(wz["CP"][:800], float) - cp0.mean()), compute_uv=False)
    print("  CP[:800] 的 POD 奇异值前 3 个 =", np.round(sv[:3], 4))
    print("  anchor/te_idx/split 示例 =", np.asarray(wz["anchor"]).ravel()[:3],
          np.asarray(wz["te_idx"]).ravel()[:3], np.asarray(wz["split"]).ravel()[:3])
    print("  Cp 首尾是否都在前缘：CP[0] 前 3 =", np.round(cp0[:3], 3),
          " 末 3 =", np.round(cp0[-3:], 3))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "03_weight_verify.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "name", "n_param", "max_abs_diff",
                                          "mean_abs_diff", "cl_weight", "cl_official",
                                          "rel_pct"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        for k, r_ in enumerate(rel):
            w.writerow({"kind": "lift_all", "name": names5[k], "n_param": 4,
                        "rel_pct": round(100 * float(r_), 3)})
    print("->", p)


if __name__ == "__main__":
    main()

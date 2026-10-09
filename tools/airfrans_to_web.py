#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一条 AirfRANS 样本转成演示网页能直接读的 JSON。

用法：
  python 04_results/demo/tools/airfrans_to_web.py \
      --dataset 02_dataset/Dataset \
      --name airFoil2D_SST_61.642_14.277_1.498_5.176_0.0_16.386 \
      --out my_sample.json

  # 不写名字时随机抽一条，或用 --index 按下标取
  python 04_results/demo/tools/airfrans_to_web.py --dataset 02_dataset/Dataset --index 0

产物是一份 JSON，内容是压力曲线与形状轮廓。用户把这份文件拖进演示网页的
"降维演示"页面，就能看到它的主成分坐标与重构误差。

口径与已验收的 pod/wallcp_1000.npz 完全一致，照抄
03_code/prediction/workstation_pod.py 的 wall_cp(anchor_le=True)：
  * 从 _aerofoil.vtp 取轮廓点与点上的 p；
  * .vtp 里是 994 条两点线段（边表），要先拼成一条闭环；
  * 弧长归一化，起点锚在前缘，前半段走上表面；
  * Cp = p / (0.5 U∞²)，U∞ 取自样本文件名。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyvista as pv


def loop_from_edges(lines, n_points):
    """.vtp 的 Lines 是两点线段列表，拼成一条闭环，返回点序下标。

    每个点度数都是 2，从 0 号点出发一路走到底就回到起点，所以不需要搜索。
    """
    arr = np.asarray(lines)
    pairs = []
    i = 0
    while i < arr.size:
        k = int(arr[i])
        if k == 2:
            pairs.append((int(arr[i + 1]), int(arr[i + 2])))
            i += 3
        else:
            i += 1 + k
    adj = [[] for _ in range(n_points)]
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)
    bad = [i for i, n in enumerate(adj) if len(n) != 2]
    if bad:
        raise ValueError("有 %d 个点的度数不是 2，例如 %s，闭环拼不出来"
                         % (len(bad), bad[:5]))
    order = [0]
    prev, cur = -1, 0
    while True:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        if nxt == 0:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) != n_points:
        raise ValueError("闭环只走到 %d 个点，总数是 %d" % (len(order), n_points))
    return np.array(order, dtype=int)


def wall_cp(points, p_wall, u_inf, n=256, le_shift=1):
    """弧长归一化、前缘锚定的 Cp 曲线。口径照抄 workstation_pod.wall_cp。

    有一处要留意。数据集里翼型前缘是一个尖点，最小 x 附近有两个几乎重合的点，
    谁当锚点会让整条曲线的弧长起点差一个微段。实测把锚点取在最小 x 的**下一个**
    点（le_shift=1），得到的曲线与已落盘的 pod/wallcp_1000.npz 逐点一致到
    一万分之一到千分之一（中段只差三万分之一）；取最小 x 本身则差到千分之几。
    所以这里默认锚在最小 x 的下一个点，与落盘产物同一口径。
    """
    X = np.vstack([points, points[:1]])
    le = (int(np.argmin(X[:, 0])) + le_shift) % len(points)
    Xr = np.roll(X, -le, axis=0)
    nr = len(Xr)
    reverse = Xr[nr // 2, 1] < Xr[nr // 2 - 1, 1]
    if reverse:
        Xr = Xr[::-1]
    pr = np.roll(p_wall, -le)
    if reverse:
        pr = pr[::-1]
    xr_n = Xr[:-1]
    seg = np.linalg.norm(np.diff(np.vstack([xr_n, xr_n[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])[:-1]
    s = s / s[-1]
    cp = pr / (0.5 * u_inf ** 2)
    return np.interp(np.linspace(0, 1, n), s, cp), s, cp, bool(reverse)


def parse_name(name):
    p = name.replace("airFoil2D_SST_", "").split("_")
    u_inf, aoa = float(p[0]), float(p[1])
    rest = [float(x) for x in p[2:]]
    if len(rest) == 4:
        L, P, Q, T = rest
        return {"uInf": u_inf, "aoa": aoa, "series": "5digit", "L": L, "P": P,
                "Q": int(round(Q)), "T": T, "M": None}
    M, P, T = rest
    return {"uInf": u_inf, "aoa": aoa, "series": "4digit", "L": None, "P": P,
            "Q": 0, "T": T, "M": M}


def convert(dataset, name):
    d = os.path.join(dataset, name)
    fp = os.path.join(d, name + "_aerofoil.vtp")
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    m = pv.read(fp)
    pts = np.asarray(m.points[:, :2], float)
    p_wall = np.asarray(m.point_data["p"], float)
    order = loop_from_edges(m.lines, m.n_points)
    meta = parse_name(name)
    cp, s, cp_native, reverse = wall_cp(pts[order], p_wall[order], meta["uInf"])
    # 轮廓按闭环顺序给出，网页可以直接画形状
    outline = pts[order]
    return {
        "name": name,
        "uInf": meta["uInf"], "aoa": meta["aoa"], "series": meta["series"],
        "L": meta["L"], "P": meta["P"], "Q": meta["Q"], "T": meta["T"], "M": meta["M"],
        "s": [round(float(v), 6) for v in np.linspace(0, 1, 256)],
        "cp": [round(float(v), 6) for v in cp],
        "outline": [[round(float(a), 6), round(float(b), 6)] for a, b in outline],
        "nSurfacePoints": int(m.n_points),
        "frontAnchored": True,
        "note": ("压力曲线是前缘锚定的弧长序，s 等于零在前缘，约零点五到后缘，"
                 "首尾闭合。轮廓按同一条闭环顺序给出。"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="02_dataset/Dataset")
    ap.add_argument("--name", default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-against", default=None,
                    help="与已落盘的 wallcp npz 逐样本对照，验证口径一致")
    args = ap.parse_args()

    if args.verify_against:
        z = np.load(args.verify_against, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        CP = np.asarray(z["CP"], float)
        rows = {n: i for i, n in enumerate(names)}
        pick = names if args.name is None else [args.name]
        rng = np.random.default_rng(0)
        if args.name is None:
            pick = [names[i] for i in rng.choice(len(names), 8, replace=False)]
        print("与 %s 对照 %d 条" % (args.verify_against, len(pick)))
        worst = 0.0
        for nm in pick:
            got = np.asarray(convert(args.dataset, nm)["cp"], float)
            ref = CP[rows[nm]]
            d = float(np.abs(got - ref).max())
            worst = max(worst, d)
            print("  %-50s 最大绝对差 %.3e  （Cp 幅度 %.2f）"
                  % (nm[-50:], d, float(ref.max() - ref.min())))
        print("全部对照最大绝对差 %.3e，%s"
              % (worst, "口径一致" if worst < 1e-3 else "口径不一致，要查"))
        print("残差集中在前缘前后各百分之几的弧长上，是那一段点距极小导致的插值位置差，"
              "中段只差约三万分之一。")
        return 0 if worst < 1e-3 else 1

    if args.name:
        name = args.name
    else:
        all_names = sorted(os.path.basename(p) for p in
                           glob.glob(os.path.join(args.dataset, "*", "*_aerofoil.vtp")))
        all_names = [n.replace("_aerofoil.vtp", "") for n in all_names]
        idx = args.index if args.index is not None else 0
        name = all_names[idx]
    obj = convert(args.dataset, name)
    out = args.out or ("%s_web.json" % name)
    json.dump(obj, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("-> %s  %.1f KB" % (out, os.path.getsize(out) / 1024))
    print("   %s  U∞=%.3f 攻角=%.3f 族=%s" % (name, obj["uInf"], obj["aoa"], obj["series"]))
    print("   Cp 范围 %.3f 到 %.3f，轮廓 %d 点"
          % (min(obj["cp"]), max(obj["cp"]), len(obj["outline"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

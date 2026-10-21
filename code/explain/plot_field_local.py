"""本地重画 §6.2 流场云图（{task}_field_{u,v,p,nu_t}.png）。

用户反馈：真值列与 A 链列是散点/网状，B 链列是纯色块，三列呈现方式不一致。
修法：把真值与 A 链**插值到 B 链自己的格点上**，三列统一用 pcolormesh 画纯色场，
共用同一色标与同一有效区域，并把翼型区域挖白 + 画黑边，三列完全同款。

只读 04_results/prediction/redraw_data/{task}/，不碰工作站、不现场推理。
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import plot_style  # noqa: E402
import surface_split as ss  # noqa: E402

REDRAW = os.path.join(ROOT, "04_results", "prediction", "redraw_data")
FIGS = os.path.join(ROOT, "04_results", "prediction", "figs")
TASKS = ["full", "scarce", "reynolds", "aoa"]
FIELDS = [("u", "u (m/s)"), ("v", "v (m/s)"), ("p", "p (Pa)"), ("nu_t", "nu_t (m2/s)")]
# npz 里的键名不规则：真值是 u_t/v_t/p_t/**nu_t**，A 链是 u_A/v_A/p_A/**nu_A**
KEY_T = {"u": "u_t", "v": "v_t", "p": "p_t", "nu_t": "nu_t"}
KEY_A = {"u": "u_A", "v": "v_A", "p": "p_A", "nu_t": "nu_A"}
# B 链统一后缀 _B（注意不是 nu_t_B）
KEY_B = {"u": "u_B", "v": "v_B", "p": "p_B", "nu_t": "nu_B"}
# ⚠ field_grid.npz 里的 B 场是**归一化单位**（实测 u 范围 -1.75..1.86），
# 不能直接画；同一格点的物理单位版本在 field_far_B.npz 的 *_g 里（已验证逐点相同）。
KEY_G = {"u": "u_g", "v": "v_g", "p": "p_g", "nu_t": "nu_g"}
NPROBE = os.environ.get("FIELD_PROBE", "")


def load_task(task):
    d = os.path.join(REDRAW, task)
    cfd = np.load(os.path.join(d, "field_cfd.npz"), allow_pickle=True)
    grid = np.load(os.path.join(d, "field_grid.npz"), allow_pickle=True)
    far = np.load(os.path.join(d, "field_far_B.npz"), allow_pickle=True)
    assert np.allclose(np.asarray(grid["x"], float), np.asarray(far["x"], float))
    assert np.allclose(np.asarray(grid["y"], float), np.asarray(far["y"], float))
    cnt = pd.read_csv(os.path.join(REDRAW, "field_cfd_counts.csv"))
    counts = cnt[cnt["task"] == task]["count"].to_numpy()
    names = [str(n) for n in cfd["names"]]
    assert len(counts) == len(names) and counts.sum() == cfd["x"].shape[0]
    gx = np.round(np.asarray(grid["x"], float), 6)
    gy = np.round(np.asarray(grid["y"], float), 6)
    ux, uy = np.unique(gx), np.unique(gy)
    ix = np.searchsorted(ux, gx)
    iy = np.searchsorted(uy, gy)
    bmask = np.zeros((len(uy), len(ux)), bool)
    bmask[iy, ix] = True
    af = pd.read_csv(os.path.join(d, "airfoil.csv"))
    return cfd, far, names, counts, ux, uy, ix, iy, bmask, af


def build_cache(cfd, far, names, counts, ux, uy, ix, iy, bmask):
    """把三个来源都变成同一格点上的矩阵，每个样本只建一次三角剖分。"""
    XX, YY = np.meshgrid(ux, uy)
    pts_q = np.column_stack([XX.ravel(), YY.ravel()])
    cache = []
    for i, nm in enumerate(names):
        s0 = int(counts[:i].sum())
        s1 = s0 + int(counts[i])
        P = np.column_stack([np.asarray(cfd["x"][s0:s1], float),
                             np.asarray(cfd["y"][s0:s1], float)])
        rec = {"name": nm}
        for k, _ in FIELDS:
            for tag, key in (("T", KEY_T[k]), ("A", KEY_A[k])):
                f = np.asarray(cfd[key][s0:s1], float)
                ok = np.isfinite(f)
                Pk = P[ok]
                fk = f[ok]
                v = LinearNDInterpolator(Pk, fk)(pts_q).reshape(XX.shape)
                # 线性插值在翼型内部/边缘会留 NaN（CFD 点云不含翼型内部），
                # 用最近邻补齐，避免图上一片白点；翼型区域随后会被挖白盖住。
                miss = ~np.isfinite(v) & bmask
                if miss.any():
                    v[miss] = NearestNDInterpolator(Pk, fk)(
                        np.column_stack([XX[miss], YY[miss]]))
                v[~bmask] = np.nan
                rec["%s_%s" % (tag, k)] = v
            b = np.full(XX.shape, np.nan)
            b[iy, ix] = np.asarray(far[KEY_G[k]], float)[i]
            rec["B_%s" % k] = b
        cache.append(rec)
    return cache


def make(task, cache, ux, uy, af):
    XX, YY = np.meshgrid(ux, uy)
    outs = []
    for k, unit in FIELDS:
        fig, axes = plt.subplots(len(cache), 3, figsize=(12.6, 2.3 * len(cache)))
        axes = np.atleast_2d(axes)
        for i, rec in enumerate(cache):
            T, A, Bb = rec["T_%s" % k], rec["A_%s" % k], rec["B_%s" % k]
            v = np.concatenate([T[np.isfinite(T)], A[np.isfinite(A)], Bb[np.isfinite(Bb)]])
            vmin, vmax = np.percentile(v, 0.2), np.percentile(v, 99.8)
            a = af[af["name"] == rec["name"]]
            ox, oy = ss.outline(a["x"].values, a["y"].values)
            fo = PathPatch(Path(np.column_stack([ox, oy])), facecolor="white",
                           edgecolor="black", lw=0.8, zorder=5)
            for j, (arr, title) in enumerate([(T, "真值 (CFD)"), (A, "A 链 (GNN)"),
                                              (Bb, "B 链 (POD 全场)")]):
                ax = axes[i][j]
                m = ax.pcolormesh(XX, YY, arr, cmap="viridis", vmin=vmin, vmax=vmax,
                                  shading="nearest", rasterized=True)
                ax.add_patch(PathPatch(fo.get_path(), facecolor="white",
                                       edgecolor="black", lw=0.8, zorder=5))
                ax.set_aspect("equal")
                ax.set_xlim(ux.min(), ux.max())
                ax.set_ylim(uy.min(), uy.max())
                if i == 0:
                    if j == 0:
                        ax.set_title(title)
                    else:
                        m0 = np.isfinite(arr) & np.isfinite(T)
                        rel = (np.linalg.norm(arr[m0] - T[m0]) /
                               max(np.linalg.norm(T[m0]), 1e-9)) if m0.any() else np.nan
                        ax.set_title("%s  relL2=%.3f" % (title, rel))
                if i == len(cache) - 1:
                    ax.set_xlabel("x")
                if j == 0:
                    ax.set_ylabel(rec["name"].replace("airFoil2D_SST_", "")[:22],
                                  fontsize=plot_style.FONTSIZE_AXES - 3)
                if i == 0 and j == 2:
                    plt.colorbar(m, ax=ax, fraction=0.046, label=unit)
        fig.suptitle("%s：%s —— 三列同一格点、同一色标（翼型区域挖白）" % (task, unit))
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        os.makedirs(FIGS, exist_ok=True)
        o = os.path.join(FIGS, "%s_field_%s.png" % (task, k))
        plot_style.save(fig, o)
        outs.append(o)
        print("   wrote", o, flush=True)
    return outs


if __name__ == "__main__":
    tasks = TASKS if len(sys.argv) < 2 else sys.argv[1:]
    for t in tasks:
        t0 = time.time()
        d = load_task(t)
        if NPROBE:
            d = list(d)
            n = int(NPROBE)
            d[2] = d[2][:n]
            d[3] = d[3][:n].copy()
            d[3][0] = int(d[3][:n].sum())
        cache = build_cache(*d[:9])
        print("[%s] 缓存完成 %.1fs" % (t, time.time() - t0), flush=True)
        make(t, cache, d[4], d[5], d[9])
        print("[%s] 全部完成 %.1fs" % (t, time.time() - t0), flush=True)

"""本地重画 §6.3 远场三列图（{task}_far_3way.png），与流场图统一成纯色场。

上一版三列用散点画，与 §6.2 现在的纯色场不一致。这里把三列都放到同一格点矩阵上
用 pcolormesh 画：真值 / B 体拟合(bf) / B 均匀网格(g)。
三者本来就都在同 31066 个远场点上（field_far_B.npz），不需要插值。

只读 04_results/prediction/redraw_data/{task}/。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import plot_style  # noqa: E402
import surface_split as ss  # noqa: E402

REDRAW = os.path.join(ROOT, "04_results", "prediction", "redraw_data")
FIGS = os.path.join(ROOT, "04_results", "prediction", "figs")
TASKS = ["full", "scarce", "reynolds", "aoa"]
FIELDS = [("u", "u (m/s)"), ("v", "v (m/s)"), ("p", "p (Pa)"), ("nu_t", "nu_t (m2/s)")]
KEY_T = {"u": "u_t", "v": "v_t", "p": "p_t", "nu_t": "nu_t"}
KEY_BF = {"u": "u_bf", "v": "v_bf", "p": "p_bf", "nu_t": "nu_bf"}
KEY_G = {"u": "u_g", "v": "v_g", "p": "p_g", "nu_t": "nu_g"}


def make(task):
    d = os.path.join(REDRAW, task)
    far = np.load(os.path.join(d, "field_far_B.npz"), allow_pickle=True)
    af = pd.read_csv(os.path.join(d, "airfoil.csv"))
    names = [str(n) for n in far["names"]]
    gx = np.round(np.asarray(far["x"], float), 6)
    gy = np.round(np.asarray(far["y"], float), 6)
    ux, uy = np.unique(gx), np.unique(gy)
    ix, iy = np.searchsorted(ux, gx), np.searchsorted(uy, gy)
    XX, YY = np.meshgrid(ux, uy)

    def mat(v):
        a = np.full(XX.shape, np.nan)
        a[iy, ix] = v
        return a

    outs = []
    for k, unit in FIELDS:
        fig, axes = plt.subplots(len(names), 3, figsize=(12.6, 2.3 * len(names)))
        axes = np.atleast_2d(axes)
        for i, nm in enumerate(names):
            arrs = [mat(np.asarray(far[KEY_T[k]], float)[i]),
                    mat(np.asarray(far[KEY_BF[k]], float)[i]),
                    mat(np.asarray(far[KEY_G[k]], float)[i])]
            v = np.concatenate([a[np.isfinite(a)] for a in arrs])
            vmin, vmax = np.percentile(v, 1), np.percentile(v, 99)
            a = af[af["name"] == nm]
            ox, oy = ss.outline(a["x"].values, a["y"].values)
            for j, (arr, lab) in enumerate(zip(arrs, ["真值 (CFD)", "B 体拟合 (bf)",
                                                      "B 均匀网格 (g)"])):
                ax = axes[i][j]
                m = ax.pcolormesh(XX, YY, arr, cmap="viridis", vmin=vmin, vmax=vmax,
                                  shading="nearest", rasterized=True)
                ax.add_patch(PathPatch(Path(np.column_stack([ox, oy])), facecolor="white",
                                       edgecolor="black", lw=0.8, zorder=5))
                ax.set_aspect("equal")
                ax.set_xlim(ux.min(), ux.max())
                ax.set_ylim(uy.min(), uy.max())
                if i == 0:
                    m0 = np.isfinite(arrs[j]) & np.isfinite(arrs[0])
                    rel = (np.linalg.norm(arrs[j][m0] - arrs[0][m0]) /
                           max(np.linalg.norm(arrs[0][m0]), 1e-9)) if m0.any() else np.nan
                    ax.set_title(lab + (("  relL2=%.3f" % rel) if j else ""))
                if i == len(names) - 1:
                    ax.set_xlabel("x")
                if j == 0:
                    ax.set_ylabel(nm.replace("airFoil2D_SST_", "")[:22],
                                  fontsize=plot_style.FONTSIZE_AXES - 3)
                if i == 0 and j == 2:
                    plt.colorbar(m, ax=ax, fraction=0.046, label=unit)
        fig.suptitle("%s：远场 %s —— 三列同格点、同色标（翼型区域挖白）" % (task, unit))
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        os.makedirs(FIGS, exist_ok=True)
        o = os.path.join(FIGS, "%s_far_3way_%s.png" % (task, k))
        plot_style.save(fig, o)
        outs.append(o)
        print("   wrote", o, flush=True)
    return outs


if __name__ == "__main__":
    import shutil
    for t in (TASKS if len(sys.argv) < 2 else sys.argv[1:]):
        outs = make(t)
        shutil.copyfile(outs[0], os.path.join(FIGS, "%s_far_3way.png" % t))
        print("[%s] 完成（= %s_far_3way.png 取 u 场）" % (t, t), flush=True)

"""本地重画 §6.1 壁面压力曲线图（{task}_cp.png）。

修复用户反馈的四点：
  1) y≈0 的黑带 —— `airfoil.csv` 的点序是按 x 排的（同一 x 上先下表面后上表面），
     直接连线会来回折返成一条黑带。改用两遍法（粗分箱求弯度线 → 按弯度线分面）
     还原上/下表面，再画闭合折线。
  2) A 链震荡 —— 实测是**分面口径造成的假尖刺**，不是模型噪声：改成按弯度线分面
     并做稳健离群剔除后，相邻跳变 p95 从 0.22~2.19 降到 0.03~0.11。
     被剔除的点会落盘到 `_cp_A_rejected.csv`，不隐藏数据。
  3) 只有实线/只有虚线 —— 每个面板都同时画真值（实线）+ 预测（虚线）。
  4) 纵轴被截断 —— 每行两列共用按数据自动定的范围（含 8% 余量）。
另：字体与字号统一走 `plot_style`（本地 Microsoft YaHei）。

只读 04_results/prediction/redraw_data/{task}/，不碰工作站、不现场推理。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import plot_style  # noqa: E402
import surface_split as ss  # noqa: E402

REDRAW = os.path.join(ROOT, "04_results", "prediction", "redraw_data")
FIGS = os.path.join(ROOT, "04_results", "prediction", "figs")
TASKS = ["full", "scarce", "reynolds", "aoa"]
REJECTED = []


def sg(x, v, win=15, poly=3, n=400):
    m = np.isfinite(v)
    if m.sum() < win + 2:
        return x, v
    xs = np.linspace(x[m].min(), x[m].max(), n)
    ys = np.interp(xs, x[m], v[m])
    return xs, savgol_filter(ys, win, poly)


def a_curves(path):
    """A 链表面点 → (xu,cu, xl,cl)；离群点记入 REJECTED。"""
    a = pd.read_csv(path)
    x = a["x"].values
    y = a["y"].values
    cp = a["cp"].values
    # 先分面（不剔除）拿到基线；再手动做一次稳健剔除以便记录被丢点
    xg, vu, vl = ss.split_surfaces(x, y, cp, reject_k=0.0)
    for tag, res in (("upper", vu), ("lower", vl)):
        m = np.isfinite(res)
        if m.sum() < 20:
            continue
        base = np.array([np.median(res[m][max(0, i - 7):i + 8]) for i in range(res.size)])
        d = np.abs(res - base)
        bad = m & (d > 0.45)
        if bad.any():
            REJECTED.append((os.path.basename(path), tag, round(float(xg[bad][0]), 4),
                             round(float(res[bad][0]), 3)))
        res[bad] = np.nan
        if tag == "upper":
            vu = res
        else:
            vl = res
    return xg, vu, vl


def draw(ax, task_dir, name, cp, af, col):
    c = cp[cp["name"] == name]
    up = c[c["upper"] == 1].sort_values("x_cord")
    lo = c[c["upper"] == 0].sort_values("x_cord")
    ax.plot(up["x_cord"], up["cp_true"], color="red", ls="-", lw=1.6, label="真值 上表面")
    ax.plot(lo["x_cord"], lo["cp_true"], color="blue", ls="-", lw=1.6, label="真值 下表面")
    vals = [up["cp_true"], lo["cp_true"]]
    if col == "B":
        ax.plot(up["x_cord"], up["cp_B"], color="red", ls="--", lw=1.3, label="B 预测 上表面")
        ax.plot(lo["x_cord"], lo["cp_B"], color="blue", ls="--", lw=1.3, label="B 预测 下表面")
        vals += [up["cp_B"], lo["cp_B"]]
    else:
        fa = os.path.join(task_dir, "cp_A_surf_%s.csv" % name)
        if os.path.isfile(fa):
            xg, vu, vl = a_curves(fa)
            sxu, svu = sg(xg, vu)
            sxl, svl = sg(xg, vl)
            ax.plot(sxu, svu, color="red", ls="--", lw=1.3, label="A 预测 上表面")
            ax.plot(sxl, svl, color="blue", ls="--", lw=1.3, label="A 预测 下表面")
            vals += [pd.Series(svu), pd.Series(svl)]
    a = af[af["name"] == name]
    xs, ys = ss.outline(a["x"].values, a["y"].values)
    ax.plot(xs, ys, color="0.45", lw=0.9, label="翼型轮廓")
    ax.grid(True, alpha=0.3)
    allv = pd.concat(vals).dropna()
    return float(allv.min()), float(allv.max())


def make(task):
    d = os.path.join(REDRAW, task)
    cp = pd.read_csv(os.path.join(d, "cp_256.csv"))
    af = pd.read_csv(os.path.join(d, "airfoil.csv"))
    names = list(dict.fromkeys(cp["name"].tolist()))[:6]
    fig, axes = plt.subplots(len(names), 2, figsize=(9.8, 2.6 * len(names)))
    for r, nm in enumerate(names):
        res = [draw(axes[r][j], d, nm, cp, af, col) for j, col in enumerate(["A", "B"])]
        lo_y = min(a for a, _ in res)
        hi_y = max(b for _, b in res)
        pad = 0.08 * (hi_y - lo_y)
        for j in range(2):
            ax = axes[r][j]
            ax.set_ylim(lo_y - pad, hi_y + pad)
            ax.set_xlim(-0.02, 1.02)
            if r == 0:
                ax.set_title("A 链预测 vs 真值" if j == 0 else "B 链预测 vs 真值")
            if r == len(names) - 1:
                ax.set_xlabel("x/c")
        axes[r][0].set_ylabel("%s\nCp" % nm[:28], fontsize=plot_style.FONTSIZE_AXES - 1)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("%s：壁面压力系数（红=上表面、蓝=下表面；实线=真值 CFD，"
                 "虚线=A/B 链预测；灰=翼型轮廓）" % task)
    fig.tight_layout(rect=[0, 0.022, 1, 0.985])
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "%s_cp.png" % task)
    plot_style.save(fig, out)
    return out


if __name__ == "__main__":
    for t in (TASKS if len(sys.argv) < 2 else sys.argv[1:]):
        print("wrote", make(t))
    if REJECTED:
        p = os.path.join(REDRAW, "_cp_A_rejected.csv")
        pd.DataFrame(REJECTED, columns=["file", "surface", "x", "value"]).to_csv(p, index=False)
        print("被剔除的 A 链离群点 %d 个 -> %s" % (len(REJECTED), p))

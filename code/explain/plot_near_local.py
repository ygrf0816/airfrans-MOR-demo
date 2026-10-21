"""本地重画 §6.3 近壁带图（{task}_near_2way_<field>.png，另存 nu_t 版为 {task}_near_2way.png）。

用户反馈与实测到的两个问题：
  1) 坐标轴看不懂：横轴 `s` **不是弦向坐标**，而是沿翼型**闭合回路**的归一化弧长
     —— s=0 与 s=1 都是**前缘**，s=0.5 是**后缘**。上一版标成"s (chord) 0→1"是错的。
  2) 上一版在一个 axes 上循环画了 4 个场量，**后画的把先画的覆盖掉**，
     实际只显示最后一个场量（nu_t），标题却按当前循环变量写 → 图文不符。
     现在改成**一个场量一张图**，标题写明场量与单位。
另：纵轴是壁法向距离的对数 log10(d)，d 从 1e-5 到 0.06（无量纲，按弦长归一）。

只读 04_results/prediction/redraw_data/{task}/field_near_{true,B}.npz。
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import plot_style  # noqa: E402

REDRAW = os.path.join(ROOT, "04_results", "prediction", "redraw_data")
FIGS = os.path.join(ROOT, "04_results", "prediction", "figs")
TASKS = ["full", "scarce", "reynolds", "aoa"]
FIELDS = [("u", "u (m/s)"), ("v", "v (m/s)"), ("p", "p (Pa)"), ("nu_t", "nu_t (m2/s)")]
KEYS = {"u": ("u", "u_B"), "v": ("v", "v_B"), "p": ("p", "p_B"), "nu_t": ("nu_t", "nu_B")}


def make(task, field):
    k, unit = field
    kt, kb = KEYS[k]
    d = os.path.join(REDRAW, task)
    T = np.load(os.path.join(d, "field_near_true.npz"), allow_pickle=True)
    B = np.load(os.path.join(d, "field_near_B.npz"), allow_pickle=True)
    names = [str(n) for n in T["names"]]
    ns, nd = int(T["ns"]), int(T["nd"])
    lo_d, hi_d = float(np.log10(np.min(10 ** np.asarray(T["log10d"])))), \
        float(np.log10(np.max(10 ** np.asarray(T["log10d"]))))
    fig, axes = plt.subplots(len(names), 2, figsize=(10.6, 2.5 * len(names)))
    for i, nm in enumerate(names):
        a_t = np.asarray(T[kt], float)[i].reshape(ns, nd).T
        a_b = np.asarray(B[kb], float)[i].reshape(ns, nd).T
        v = np.concatenate([a_t[np.isfinite(a_t)], a_b[np.isfinite(a_b)]])
        vmin, vmax = np.percentile(v, 1), np.percentile(v, 99)
        for j, (arr, lab) in enumerate([(a_t, "真值 (CFD)"), (a_b, "B 链 (贴体 POD)")]):
            ax = axes[i][j]
            m = ax.imshow(arr, origin="lower", aspect="auto", interpolation="nearest",
                          cmap="viridis", vmin=vmin, vmax=vmax,
                          extent=[0, 1, lo_d, hi_d])
            ax.axvline(0.5, color="white", ls="--", lw=0.8, alpha=0.8)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            if i == 0:
                ax.set_title("%s —— %s" % (lab, unit))
            if i == len(names) - 1:
                ax.set_xlabel("s：沿表面闭合回路弧长（0 与 1 = 前缘，0.5 = 后缘）")
            if j == 0:
                ax.set_ylabel("%s\nlog10(d)" % nm.replace("airFoil2D_SST_", "")[:22],
                              fontsize=plot_style.FONTSIZE_AXES - 3)
            if i == 0 and j == 0:
                ax.annotate("后缘 TE", xy=(0.5, hi_d), xytext=(0.56, hi_d - 0.15),
                            color="white", fontsize=plot_style.FONTSIZE_LEGEND)
            if i == 0 and j == 1:
                plt.colorbar(m, ax=ax, fraction=0.046, label=unit)
                ax.set_yticklabels([])
    fig.suptitle("%s：近壁贴体带 —— 横轴 s 是绕翼型一圈的弧长（左半 s∈[0,0.5]＝上表面、"
                 "右半 s∈[0.5,1]＝下表面；0/1 前缘、0.5 后缘），纵轴壁法向距离对数 "
                 "log10(d)，d∈[1e-5, 0.06]" % task)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "%s_near_2way_%s.png" % (task, k))
    plot_style.save(fig, out)
    return out


if __name__ == "__main__":
    tasks = TASKS if len(sys.argv) < 2 else sys.argv[1:]
    import shutil
    for t in tasks:
        for f in FIELDS:
            o = make(t, f)
            print("wrote", o, flush=True)
            if f[0] == "nu_t":
                shutil.copyfile(o, os.path.join(FIGS, "%s_near_2way.png" % t))
                print("   (= %s_near_2way.png)" % t)

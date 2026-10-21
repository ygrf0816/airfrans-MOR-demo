"""为可解释性分析实验报告补两张说明图（其余图已有）。

产物（写到 04_results/explain/）：
  1. report_station_mapping.png —— §2 口径：站点索引如何折成真实弦向位置、
     平均 Cp 曲线的闭合回路形状、以及 `S` 其实是奇异值谱（三条并排，一张图讲清那次纠错）。
  2. report_sensitivity_compare.png —— §3 全局依赖：数据侧标准化系数与模型侧 ±1σ 位移并列。
"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import plot_style  # noqa: E402

EXPLAIN = os.path.join(ROOT, "04_results", "explain")
MOR = os.path.join(ROOT, "04_results", "mor")
COMP = ["yc_mid", "th_peak_half", "th_max_xc", "re_radius_c"]
CN = {"yc_mid": "弯度中点", "th_peak_half": "最大半厚",
      "th_max_xc": "最厚位置", "re_radius_c": "前缘半径"}
FAMC = {"4digit": "#1f77b4", "5digit": "#d62728"}


def seg_of(idx):
    xh = idx / 127.0 if idx <= 127 else (255 - idx) / 127.0
    return "front" if xh <= 0.2 else ("rear" if xh >= 0.8 else "mid")


def fig_station():
    rows = list(csv.DictReader(open(os.path.join(EXPLAIN, "geom_station_xy.csv"),
                                   encoding="utf-8")))
    idx = np.array([int(r["station_idx"]) for r in rows])
    xc = np.array([float(r["x_chord"]) for r in rows])
    z = np.load(os.path.join(MOR, "wallcp_1000_LE_anchored.npz"), allow_pickle=True)
    cp_mean = np.asarray(z["CP"], float).mean(axis=0)
    S = np.asarray(z["S"], float)
    seg = np.array([seg_of(i) for i in idx])
    col = {"front": "#ffe0b2", "mid": "#f5f5f5", "rear": "#c8e6c9"}

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.3))
    for i in range(256):
        ax[0].axvspan(i - 0.5, i + 0.5, color=col[seg[i]], lw=0)
        ax[1].axvspan(i - 0.5, i + 0.5, color=col[seg[i]], lw=0)
    ax[0].plot(idx, xc, "-", color="k", lw=1.6)
    ax[0].plot([0, 127, 255], [xc[0], xc[127], xc[255]], "o", color="crimson",
               ms=5, zorder=5)
    for k, t, dy, ha in ((0, "索引 0\n前缘", 22, "left"),
                         (127, "索引 127\n后缘", 8, "center"),
                         (255, "索引 255\n前缘", 20, "right")):
        ax[0].annotate(t, (k, xc[k]), textcoords="offset points",
                       xytext=(0, dy) if k != 127 else (0, dy), ha=ha,
                       fontsize=plot_style.FONTSIZE_LEGEND)
    ax[0].set_xlabel("Cp 曲线上的站点索引")
    ax[0].set_ylabel("真实弦向位置 $x/c$")
    ax[0].set_title("(a) 索引 → 弦向位置（闭合回路）")
    ax[0].set_ylim(-0.16, 1.30)
    ax[0].set_xlim(-14, 269)

    ax[1].plot(idx, cp_mean, "-", color="navy", lw=1.5)
    ax[1].set_xlabel("站点索引")
    ax[1].set_ylabel("平均 $C_p$（1000 样本）")
    ax[1].set_title("(b) 平均 $C_p$：两端都是前缘吸力峰")
    ax[1].set_ylim(float(cp_mean.min()) - 0.45, float(cp_mean.max()) + 0.55)
    ax[1].set_xlim(-14, 269)
    ax[1].annotate("前缘吸力峰 %.2f" % cp_mean[0], (0, cp_mean[0]),
                   textcoords="offset points", xytext=(10, -18),
                   fontsize=plot_style.FONTSIZE_LEGEND, ha="left")
    ax[1].annotate("前缘吸力峰 %.2f" % cp_mean[255], (255, cp_mean[255]),
                   textcoords="offset points", xytext=(-18, -30),
                   fontsize=plot_style.FONTSIZE_LEGEND, ha="right")
    ax[1].annotate("全局最大 %.2f\n（后缘附近）" % cp_mean[127], (127, cp_mean[127]),
                   textcoords="offset points", xytext=(24, 16),
                   fontsize=plot_style.FONTSIZE_LEGEND, ha="left",
                   arrowprops=dict(arrowstyle="-", lw=0.8, color="0.3"))

    ax[2].semilogy(idx, S, "-", color="darkgreen", lw=1.5)
    ax[2].set_xlabel("索引（= 奇异值序号）")
    ax[2].set_ylabel("$S$（半对数）")
    ax[2].set_title("(c) $S$ 是 POD 奇异值谱，不是坐标")
    ax[2].annotate("单调不增，末尾 %.2e\n（没有任何一个是 0）" % S[-1], (150, S[150]),
                   textcoords="offset points", xytext=(-30, -46),
                   fontsize=plot_style.FONTSIZE_LEGEND, ha="left",
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    handles = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in ("front", "mid", "rear")]
    fig.legend(handles, ["前缘段（$x_{half}\\leq0.2$：索引 0..25 与 230..255）",
                         "中段（索引 26..101 与 154..229）",
                         "后缘段（$x_{half}\\geq0.8$：索引 102..153）"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("口径纠错的三条依据：闭合回路的两端都是前缘、后缘在中间、$S$ 不是坐标")
    fig.tight_layout(rect=[0, 0.06, 1, 0.955])
    out = os.path.join(EXPLAIN, "report_station_mapping.png")
    plot_style.save(fig, out)
    return out


def fig_sensitivity():
    d = list(csv.DictReader(open(os.path.join(EXPLAIN, "sensitivity_data_side.csv"),
                                 encoding="utf-8")))
    m = list(csv.DictReader(open(os.path.join(EXPLAIN, "sensitivity_model_side.csv"),
                                 encoding="utf-8")))
    dcoef, mdisp = {}, {}
    for r in d:
        if r["fam"] in ("4digit", "5digit"):
            dcoef.setdefault((r["fam"], r["comp"]), []).append(float(r["coef_std"]))
    for r in m:
        mdisp.setdefault((r["fam"], r["comp"]), []).append(abs(float(r["pod_disp"])))
    x = np.arange(len(COMP))
    w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.2))
    for k, fam in enumerate(("4digit", "5digit")):
        v = [np.mean(dcoef[(fam, c)]) for c in COMP]
        b = ax[0].bar(x + (k - 0.5) * w, v, w, color=FAMC[fam],
                      label="%s 族" % ("4 位" if fam == "4digit" else "5 位"))
        ax[0].bar_label(b, fmt="%.2f", fontsize=plot_style.FONTSIZE_LEGEND - 1, padding=1)
        v2 = [np.mean(mdisp[(fam, c)]) for c in COMP]
        b2 = ax[1].bar(x + (k - 0.5) * w, v2, w, color=FAMC[fam],
                       label="%s 族" % ("4 位" if fam == "4digit" else "5 位"))
        ax[1].bar_label(b2, fmt="%.2f", fontsize=plot_style.FONTSIZE_LEGEND - 1, padding=1)
    for a, t, yl in ((ax[0], "(a) 数据侧：标准化偏回归系数（8 个 POD 模取均值）", "标准化系数"),
                     (ax[1], "(b) 模型侧：±1σ 几何扰动引起的 POD 位移范数", "位移范数")):
        a.set_xticks(x)
        a.set_xticklabels([CN[c] for c in COMP])
        a.set_title(t)
        a.set_ylabel(yl)
        a.axhline(0, color="k", lw=0.8)
        a.legend()
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("两条腿都指向同一个结论：最大半厚（厚度）是最强的几何分量；"
                 "最厚位置与弯度最弱")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(EXPLAIN, "report_sensitivity_compare.png")
    plot_style.save(fig, out)
    return out


if __name__ == "__main__":
    print("wrote", fig_station())
    print("wrote", fig_sensitivity())

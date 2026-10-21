#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文式报告用的图（只读落盘数据，不现场推理）。

与 `ga_plots.py` 的分工：`ga_plots.py` 出的是**过程图/记录图**（每个种子一套、带 tag），
本脚本出的是**报告图**——按论文的叙述顺序组织：方法示意 → 设计空间 → 前沿与收敛 →
案例的形状/流场/压力曲线 → 反演与多解 → 真实样本锚定。

用法：
    python 03_code/inversion/plot_report_figs.py            # 出全部（缺数据的自动跳过）
输出：04_results/inversion/report_figs/*.png
依赖的落盘数据：results/ 下的 ga_*、grid_all_full.csv、cases_eval_s8.*、cases_field.npz、
invert_*.npz、anchor_real.npz、grad_check.npz；以及 03_code/probe/manifest.json。
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "04_results", "inversion", "results")
# 报告插图**直接放在报告同一目录**（与第一阶段可解释性报告的做法一致）：
# 这样 Markdown 里用裸文件名引用，任何查看器（Typora / VS Code / Web GUI）都能显示。
FIGS = os.path.join(ROOT, "04_results", "inversion")
FOLD = FIGS                     # 兼容旧变量名
sys.path.insert(0, os.path.join(ROOT, "03_code", "explain"))
sys.path.insert(0, os.path.join(ROOT, "01_initial", "AirfRANS-main"))
try:
    import plot_style
    SAVE = plot_style.save
except Exception:
    def SAVE(fig, path):
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)

DB_DESIGN_BEST = 1.1464      # 设计点（攻角 8°±1°、U∞=62±4 m/s）库内最优
REACH_HI = 1.4225            # 工作带（攻角 4°–10°）可达上界
SEP_MAX = 0.007874015748     # 分离代理阈值（=511 个真实样本 p90 = 1 格）
M_MAX = 0.6                  # 局部马赫数代理上限
MAKE = []                    # (文件名, 一句话图注)


def _rows(name):
    p = os.path.join(RES, name)
    if not os.path.isfile(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def _note(fn, cap):
    MAKE.append((fn, cap))
    print("  ->", fn, "|", cap)


def geom128(params, nb=400):
    """(L,P,Q,T) -> 128 维几何（与 ga_design.gen_geometry 同口径，本地不需要 torch）。"""
    import naca_generator as NG
    L, P, Q, T = params
    pos = NG.naca_generator((L, P, int(round(Q)), T), nb_samples=nb,
                            cosine_spacing=True, verbose=False)
    X, Y = pos[:, 0], pos[:, 1]
    x0, x1 = X.min(), X.max()
    c = x1 - x0
    n = 64
    beta = np.pi * np.linspace(1, 0, n, endpoint=False)
    xs = (1 - np.cos(beta)) / 2.0 * c + x0
    vec = np.zeros(2 * n)
    for i, xx in enumerate(xs):
        band = np.abs(X - xx) < (c / n) * 0.5
        ys = Y[band]
        if ys.size >= 2:
            vec[i], vec[n + i] = ys.max() / c, ys.min() / c
        elif ys.size == 1:
            vec[i] = vec[n + i] = ys[0] / c
    return vec


def surface(params, nb=600):
    """(L,P,Q,T) -> 闭合轮廓点（前缘锚定），用于在流场图上画翼型。"""
    import naca_generator as NG
    L, P, Q, T = params
    pos = NG.naca_generator((L, P, int(round(Q)), T), nb_samples=nb,
                            cosine_spacing=True, verbose=False)
    P2 = pos[:, :2]
    if np.allclose(P2[0], P2[-1]):
        P2 = P2[:-1]
    le = int(np.argmin(P2[:, 0]))
    return np.roll(P2, -le, axis=0)


def case_table():
    """案例点（来自 cases_eval_s8.csv），返回 [{name,params,CL8,delta,m,sep,feasible}]。"""
    rows = _rows("cases_eval_s8.csv")
    labels = {1: "案例1 成功（seed 7 最优）", 2: "案例2 边界（分离代理贴阈值）",
              3: "案例3 失败·局部马赫超限", 4: "案例4 失败·带内不稳健",
              5: "案例5 不贴边成功解"}
    out = []
    for r in rows:
        out.append(dict(idx=int(r["case"]), label=labels.get(int(r["case"]), r["case"]),
                        params=[float(r[k]) for k in ("L", "P", "Q", "T")],
                        CL4=float(r["CL4"]), CL8=float(r["CL8"]), CL10=float(r["CL10"]),
                        delta=float(r["delta"]), m=float(r["m_max"]),
                        sep=float(r["sep"]), feas=(r["feasible"] == "True"),
                        binding=r["binding"]))
    return out


# ------------------------------------------------------------------ 图 1 方法示意
def fig_workflow():
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.62, "AirfRANS 数据集\n1000 个二维翼型绕流样本\n（RANS，含压力/速度场）", "#e8f0fe"),
        (0.26, 0.62, "降维（POD）\n压力曲线 256 点 → 8 个坐标\n流场 → 32 个坐标/场", "#e8f0fe"),
        (0.50, 0.62, "前向模型（本文用「B 链」）\n输入：攻角、风速、几何等\n输出：压力曲线与流场", "#e8f0fe"),
        (0.74, 0.62, "本文要解决的两件事\n① 反着设计几何（搜索）\n② 反着猜几何（Cp 反演）", "#fde8e8"),
        (0.02, 0.10, "验证一：与数据库真实样本比\n511 个样本的升力排序/偏差", "#e8f5e9"),
        (0.26, 0.10, "验证二：与官方留出集比\n89 个模型没训过的样本", "#e8f5e9"),
        (0.50, 0.10, "验证三：把模型自己当尺子\n看反演有没有到达「模型地板」", "#e8f5e9"),
        (0.74, 0.10, "验证四：三类真实案例\n成功 / 边界 / 失败各一", "#e8f5e9"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.22, 0.24, boxstyle="round,pad=0.012",
                                    linewidth=1.1, edgecolor="0.35", facecolor=c))
        ax.text(x + 0.11, y + 0.12, t, ha="center", va="center", fontsize=9.5)
    for x in (0.24, 0.48, 0.72):
        ax.add_patch(FancyArrowPatch((x, 0.74), (x + 0.02, 0.74), arrowstyle="-|>",
                                     mutation_scale=14, color="0.35"))
    for x in (0.13, 0.37, 0.61, 0.85):
        ax.add_patch(FancyArrowPatch((x, 0.60), (x, 0.36), arrowstyle="-|>",
                                     mutation_scale=14, color="0.55", linestyle="--"))
    ax.text(0.5, 0.475, "同一套落盘数据既是输入，也是验收对象（图中虚线）",
            ha="center", fontsize=9, color="0.4")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 0.92)
    ax.set_title("图 1　本文的技术路线：数据 → 降维 → 前向模型 → 反向设计/反演 → 四类验证",
                 fontsize=12)
    os.makedirs(FIGS, exist_ok=True)
    fn = "fig1_workflow.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "技术路线示意图（无数据）")


# --------------------------------------------------- 图 2 设计空间与可行区（穷举）
def fig_design_space(cases):
    rows = _rows("grid_all_full.csv")
    if not rows:
        print("  (缺 grid_all_full.csv，跳过设计空间图)")
        return
    L = np.array([float(r["L"]) for r in rows])
    T = np.array([float(r["T"]) for r in rows])
    P = np.array([float(r["P"]) for r in rows])
    CL = np.array([float(r["CL8"]) for r in rows])
    D = np.array([float(r["delta"]) for r in rows])
    FE = np.array([r["feasible"] == "True" for r in rows])
    Ls, Ts = np.unique(np.round(L, 4)), np.unique(np.round(T, 4))
    M = np.full((len(Ts), len(Ls)), np.nan)
    Fr = np.full((len(Ts), len(Ls)), np.nan)
    for i, lv in enumerate(Ls):
        for j, tv in enumerate(Ts):
            k = (np.round(L, 4) == lv) & (np.round(T, 4) == tv) & (np.round(P, 3) > 7.0)
            if not k.any():
                continue
            ok = k & FE
            Fr[j, i] = ok.sum() / k.sum()
            if ok.any():
                M[j, i] = CL[ok].max()
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.3))
    for ax, A, ttl, cm in ((axes[0], M, "$C_L(8°)$ 最好值", "viridis"),
                           (axes[1], Fr, "该点可行的参数比例", "cividis")):
        im = ax.pcolormesh(Ls, Ts, A, cmap=cm, shading="nearest")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("L（第一位数字，决定弯度大小）")
        ax.set_ylabel("T（厚度百分比）")
        ax.set_title(ttl)
    axes[0].contour(Ls, Ts, np.nan_to_num(M, nan=-9), levels=[DB_DESIGN_BEST],
                    colors="w", linewidths=1.4)
    # 可行/不可行的直观分区
    ax = axes[2]
    for i, lv in enumerate(Ls):
        for j, tv in enumerate(Ts):
            k = (np.round(L, 4) == lv) & (np.round(T, 4) == tv)
            if not k.any():
                continue
            frac = FE[k].mean()
            ax.add_patch(plt.Rectangle((lv - 0.18, tv - 0.83), 0.36, 1.66,
                                       facecolor=plt.cm.RdYlGn(frac), edgecolor="none"))
    ax.set_xlim(Ls.min() - 0.4, Ls.max() + 0.4)
    ax.set_ylim(Ts.min() - 1.5, Ts.max() + 1.5)
    ax.set_xlabel("L（第一位数字，决定弯度大小）")
    ax.set_ylabel("T（厚度百分比）")
    ax.set_title("可行（绿）/ 不可行（红）：约束把左上角切掉了")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label="可行比例")
    for c in cases:
        ax.plot(c["params"][0], c["params"][3], "k*", ms=13)
        ax.annotate(str(c["idx"]), (c["params"][0], c["params"][3]),
                    textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.plot(2.248, 8.267, "k^", ms=9)
    ax.annotate("数据库设计点最优", (2.248, 8.267), textcoords="offset points",
                xytext=(-8, -16), fontsize=8.5)
    fig.suptitle("图 2　设计空间：弯度码 L × 厚度 T（每格内再取弯度位置 P 的最优值）；"
                 "星号=本文五个案例，三角=数据库对照", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fn = "fig2_design_space.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "grid_all_full.csv（1920 点穷举）")


# --------------------------------------------- 图 3 三种子前沿叠在一张图
def fig_pareto_all(cases):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    colors = {"_s42": "#1f77b4", "_s7": "#d62728", "_s2024": "#2ca02c"}
    for tag, col in colors.items():
        rows = [r for r in _rows("ga_front%s.csv" % tag) if r["feasible"] == "True"]
        if not rows:
            continue
        d = np.array([float(r["delta"]) for r in rows])
        c = np.array([float(r["CL8"]) for r in rows])
        o = np.argsort(d)
        ax.plot(d[o], c[o], "-o", ms=3.2, lw=1.1, color=col,
                label="随机种子 %s（%d 个解）" % (tag.strip("_"), len(rows)))
    ax.axhline(DB_DESIGN_BEST, ls="--", lw=1, color="0.35")
    ax.text(0.62, DB_DESIGN_BEST + 0.006, "数据库设计点最优 %.4f" % DB_DESIGN_BEST,
            fontsize=9, color="0.35")
    ax.axhline(REACH_HI, ls=":", lw=1, color="0.55")
    ax.text(0.62, REACH_HI - 0.02, "工作带可达上界 %.4f" % REACH_HI, fontsize=9, color="0.45")
    for c in cases:
        ax.plot(c["delta"], c["CL8"], "k*", ms=13)
        ax.annotate("%d" % c["idx"], (c["delta"], c["CL8"]),
                    textcoords="offset points", xytext=(7, 3), fontsize=10)
    ax.plot(0.349, 1.1019, "k^", ms=8)
    ax.set_xlabel("攻角带内升力损失 Δ（4° 或 10° 相对 8° 的最大掉幅，越小越稳）")
    ax.set_ylabel("设计攻角 8° 的升力系数 $C_L$")
    ax.set_title("图 3　三个随机种子各自搜出的「升力 与 带内稳健」折中曲线（星号=图 2 的五个案例）",
                 fontsize=11.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fn = "fig3_pareto_all.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "results/ga_front_{s42,s7,s2024}.csv")


# --------------------------------------------- 图 4 收敛（三子）+ 超体积
def fig_convergence():
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.2))
    colors = {"_s42": "#1f77b4", "_s7": "#d62728", "_s2024": "#2ca02c"}
    for tag, col in colors.items():
        h = _rows("ga_history%s.csv" % tag)
        if not h:
            continue
        g = np.array([int(r["gen"]) for r in h])
        bf = np.array([float(r["best_feasible_CL8"]) for r in h])
        axes[0].plot(g, bf, "-", lw=1.4, color=col, label=tag.strip("_"))
        axes[1].plot(g, np.array([int(r["n_feasible"]) for r in h]), "-", lw=1.4,
                     color=col, label=tag.strip("_"))
        z = os.path.join(RES, "ga_snaps%s.npz" % tag)
        if os.path.isfile(z):
            zz = np.load(z, allow_pickle=True)
            gs, hv = [], []
            for k in range(len(zz["gens"])):
                F, G = zz["F%d" % k], zz["G%d" % k]
                ok = np.all(G <= 0, axis=1)
                gs.append(int(zz["gens"][k]))
                hv.append(_hv(F[ok] if ok.any() else F))
            axes[2].plot(gs, hv, "-o", ms=3, color=col, label=tag.strip("_"))
    axes[0].axhline(DB_DESIGN_BEST, ls="--", color="0.4", lw=1)
    axes[0].text(2, DB_DESIGN_BEST + 0.004, "数据库设计点最优 %.4f" % DB_DESIGN_BEST,
                 fontsize=8.5, color="0.4")
    axes[0].set_xlabel("进化代数"); axes[0].set_ylabel("可行解里最优 $C_L(8°)$")
    axes[0].set_title("升力收敛：第 20 代前基本到位"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("进化代数"); axes[1].set_ylabel("可行解个数（种群 80）")
    axes[1].set_title("可行解数：很快占满整个种群"); axes[1].grid(alpha=0.3)
    axes[2].set_xlabel("进化代数"); axes[2].set_ylabel("折中前沿覆盖的面积（越大越好）")
    axes[2].set_title("前沿覆盖面积：上升后饱和 = 已收敛"); axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle("图 4　三次独立运行（不同随机种子）都在 20 代上下收敛", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fn = "fig4_convergence.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "ga_history_*.csv + ga_snaps_*.npz（覆盖面积由快照原始目标值重算）")


def _hv(F, ref=(0.5, 1.5)):
    """二维超体积：先剔被支配点，再按第一目标升序累加条带面积（口径与 03_weight_verify §3 一致）。"""
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


# --------------------------------------------- 图 5 梯度核对（方法可信度）
def fig_gradcheck():
    p = os.path.join(RES, "grad_check.npz")
    if not os.path.isfile(p):
        print("  (缺 grad_check.npz，跳过梯度图)")
        return
    z = np.load(p, allow_pickle=True)
    eps = np.asarray(z["fd_eps"], float)
    rel = np.asarray(z["fd_rel_by_eps"], float)      # (样本, 方法, ε)
    names = [str(x)[:26] for x in z["names"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    for i in range(rel.shape[0]):
        axes[0].loglog(eps[i], rel[i, 0], "-o", ms=4, lw=1.2, label=names[i])
    axes[0].set_xlabel("有限差分步长 ε")
    axes[0].set_ylabel("与自动微分的相对差")
    axes[0].set_title("步长越小越接近（说明链路真的可导）")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=7.5)
    a = np.asarray(z["grad"], float).ravel()
    b = np.asarray(z["fd_grad"], float).ravel()
    axes[1].scatter(b, a, s=8, alpha=0.5)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    axes[1].plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
    axes[1].set_xlabel("有限差分得到的梯度")
    axes[1].set_ylabel("自动微分得到的梯度")
    axes[1].set_title("两种算法逐维对照（45° 线上=一致）")
    axes[1].grid(alpha=0.3)
    fig.suptitle("图 5　可导性核对：几何改变对升力的影响，用两种算法互相验证", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fn = "fig5_gradcheck.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "results/grad_check.npz")


# --------------------------------------------- 图 6 五个案例的形状
def fig_case_shapes(cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(3.1 * len(cases), 3.9))
    for ax, c in zip(axes, cases):
        g = geom128(c["params"])
        x = np.linspace(0, 1, 64)
        up, lo = g[:64][::-1], g[64:][::-1]
        ax.fill_between(x, lo, up, color="#cfe0f5", edgecolor="none")
        ax.plot(x, up, color="crimson", lw=1.6)
        ax.plot(x, lo, color="navy", lw=1.6)
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.22, 0.30)
        ax.set_aspect("equal")
        ax.set_title("%s\nL=%.2f P=%.2f Q=%d T=%.2f"
                     % (c["label"], c["params"][0], c["params"][1],
                        int(c["params"][2]), c["params"][3]), fontsize=8.5)
        ax.set_xlabel("弦向位置 x/c", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("厚度方向 y/c", fontsize=9)
    fig.suptitle("图 6　五个案例翼型的剖面形状（红=上表面、蓝=下表面；Q=0 正常弯、Q=1 反弯）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fn = "fig6_case_shapes.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "cases_eval_s8.npz 的 geometry + 官方生成器（同口径重画）")


# --------------------------------------------- 图 7 带内升力：稳健性怎么看
def fig_band_lift(cases):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    a = np.array([4, 8, 10])
    for c in cases:
        y = np.array([c["CL4"], c["CL8"], c["CL10"]])
        axes[0].plot(a, y, "-o", lw=1.5, ms=5,
                     label="%d %s" % (c["idx"], c["label"][4:12]))
    axes[0].set_xticks(a)
    axes[0].set_xlabel("攻角（度）")
    axes[0].set_ylabel("升力系数 $C_L$")
    axes[0].set_title("工作带内三个攻角的升力")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    x = np.arange(len(cases))
    d = [c["delta"] for c in cases]
    col = ["#2ca02c" if c["feas"] else "#d62728" for c in cases]
    axes[1].bar(x, d, color=col)
    axes[1].axhline(0.37, ls="--", color="0.4", lw=1)
    axes[1].text(0.02, 0.385, "要超过数据库设计点最优，代价约 Δ≥0.37", fontsize=8.5, color="0.35")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["%d" % c["idx"] for c in cases])
    axes[1].set_xlabel("案例编号")
    axes[1].set_ylabel("带内升力损失 Δ（越小越稳）")
    axes[1].set_title("带内稳健性：Δ 越大，攻角偏离 8° 时掉得越狠")
    axes[1].grid(alpha=0.3, axis="y")
    for xi, c in zip(x, cases):
        axes[1].text(xi, c["delta"] + 0.01, ("%.3f" % c["delta"]), ha="center", fontsize=8)
    fig.suptitle("图 7　「设计点升力大」不等于「整个攻角带都好用」：案例 4 就是升力不低但 Δ 最大的反例",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fn = "fig7_band_lift.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "cases_eval_s8.csv")


def fold_cp(cp):
    """把 256 点闭环压力曲线折成两条表面的「弦向」曲线。

    口径（已核实）：0 与 255 都在前缘、约 127 在后缘；所以 `cp[:128]` 是前缘→后缘的一条表面，
    `cp[128:]` 是后缘→前缘的另一条。返回 (x/c, 表面A, 表面B)，两条都按前缘→后缘排列。
    **哪条是上/下表面不能按索引猜**，本函数返回的两条按"平均压力更低（吸力更强）"排序，
    画图时用文字说明这一点。
    """
    a = np.asarray(cp, float)[:128]
    b = np.asarray(cp, float)[128:][::-1]
    x = np.linspace(0, 1, 128)
    if a.mean() > b.mean():
        a, b = b, a
    return x, a, b


# --------------------------------------------- 图 8 五个案例的压力曲线对比
def fig_cp_all(cases):
    p = os.path.join(RES, "cases_eval_s8.npz")
    if not os.path.isfile(p):
        return
    z = np.load(p, allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    for i, c in enumerate(cases):
        x, a, b = fold_cp(np.asarray(z["cp8"])[i])
        axes[0].plot(x, a, lw=1.5, label="案例 %d" % c["idx"])
        axes[0].plot(x, b, lw=1.0, ls="--", color=axes[0].lines[-1].get_color())
    axes[0].set_xlabel("弦向位置 x/c（0=前缘，1=后缘）")
    axes[0].set_ylabel("压力系数 $C_p$")
    axes[0].set_title("8° 攻角：实线=压力较低（吸力）的那一面，虚线=另一面")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8, ncol=2)
    for i, c in enumerate(cases):
        cp = np.asarray(z["cp8"])[i]
        axes[1].plot(np.arange(256), cp, lw=1.5, label="案例 %d" % c["idx"])
    axes[1].axvspan(0, 25, color="0.9"); axes[1].axvspan(230, 255, color="0.9")
    axes[1].text(2, 0.6, "前缘附近\n（两段都在前缘）", fontsize=8, color="0.35")
    axes[1].set_xlabel("壁面点序号（0 与 255 都在前缘、约 127 在后缘）")
    axes[1].set_ylabel("压力系数 $C_p$")
    axes[1].set_title("原始排列：同一条曲线绕翼型一圈，所以两端都在前缘")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("图 8　五个案例在 8° 攻角下的压力分布（同一条曲线两种画法）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fn = "fig8_cp_all.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "cases_eval_s8.npz（cp8）；折面口径见脚本 fold_cp")


# ------------------------------------------------- 流场相关（依赖 cases_field.npz）
FIELD_CASES = {1: "case1_success", 2: "case2_boundary", 3: "case3_fail_mach",
               4: "case4_fail_robust", 5: "case5_interior"}


def _fields():
    p = os.path.join(RES, "cases_field.npz")
    if not os.path.isfile(p):
        print("  (缺 cases_field.npz——流场图跳过)")
        return None
    return np.load(p, allow_pickle=True)


def _draw_field(ax, z, key, cases_idx, aoa, title, vmin=None, vmax=None, cmap="RdBu_r",
                cbl="压力（与数据集同一口径）"):
    F = np.asarray(z[key], float)
    x, y = np.asarray(z["x"], float), np.asarray(z["y"], float)
    X, Y = np.meshgrid(x, y)
    im = ax.pcolormesh(X, Y, F, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    c = [c for c in case_table() if c["idx"] == cases_idx][0]
    s = surface(c["params"])
    ax.plot(s[:, 0], s[:, 1], "-", color="k", lw=1.0)
    ax.set_xlim(-0.35, 1.75); ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("弦向 x/c", fontsize=9); ax.set_ylabel("y/c", fontsize=9)
    ax.tick_params(labelsize=8)
    return im


def fig_case_detail(cases):
    """每个案例一张三联图：形状 / 流场 / 压力曲线（论文里随文插入的主图）。"""
    z = _fields()
    if z is None:
        return
    for c in cases:
        code = FIELD_CASES[c["idx"]]
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        # (a) 形状
        g = geom128(c["params"])
        x = np.linspace(0, 1, 64)
        up, lo = g[:64][::-1], g[64:][::-1]
        axes[0].fill_between(x, lo, up, color="#cfe0f5")
        axes[0].plot(x, up, color="crimson", lw=1.7, label="上表面")
        axes[0].plot(x, lo, color="navy", lw=1.7, label="下表面")
        axes[0].set_aspect("equal"); axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(-0.24, 0.32)
        axes[0].set_xlabel("弦向 x/c"); axes[0].set_ylabel("y/c")
        axes[0].set_title("(a) 翼型剖面：L=%.2f P=%.2f Q=%d T=%.2f"
                          % (c["params"][0], c["params"][1], int(c["params"][2]),
                             c["params"][3]), fontsize=10)
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
        # (b) 流场（压力）
        key = "p_%s_8" % code
        if key in z.files:
            im = _draw_field(axes[1], z, key, c["idx"], 8, "(b) 8° 攻角下的压力场")
            plt.colorbar(im, ax=axes[1], fraction=0.046)
        else:
            axes[1].text(0.5, 0.5, "缺 %s" % key, ha="center")
        # (c) 三个攻角的压力曲线（折面画法）
        p = os.path.join(RES, "cases_eval_s8.npz")
        if os.path.isfile(p):
            zz = np.load(p, allow_pickle=True)
            for k, lab in (("cp4", "4°"), ("cp8", "8°"), ("cp10", "10°")):
                xx, a, b = fold_cp(np.asarray(zz[k])[c["idx"] - 1])
                axes[2].plot(xx, a, lw=1.5, label="%s 吸力面" % lab)
            axes[2].set_xlabel("弦向 x/c"); axes[2].set_ylabel("压力系数 $C_p$")
        axes[2].set_title("(c) 三个攻角的压力分布（实线=吸力面）", fontsize=10)
        axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
        fig.suptitle("图 9-%d　%s：形状、压力场与压力曲线　"
                     "[$C_L(8°)$=%.3f，带内损失 Δ=%.3f，局部马赫代理=%.3f，分离代理=%.5f，%s]"
                     % (c["idx"], c["label"], c["CL8"], c["delta"], c["m"], c["sep"],
                        "满足全部约束" if c["feas"] else "不可行"), fontsize=11.5)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fn = "fig9_%d_case%d.png" % (c["idx"], c["idx"])
        SAVE(fig, os.path.join(FIGS, fn))
        _note(fn, "cases_eval_s8.npz + cases_field.npz")


def fig_fields_all(cases):
    """五个案例的流场并排（上排压力、下排速度大小），看设计差异如何反映在流场上。"""
    z = _fields()
    if z is None:
        return
    fig, axes = plt.subplots(2, len(cases), figsize=(3.4 * len(cases), 6.4))
    # 跨面板比较必须用**同一个色标**，否则会误判"哪个吸力更强"
    pk = ["p_%s_8" % FIELD_CASES[c["idx"]] for c in cases]
    vmin_p = min(float(np.nanmin(np.asarray(z[k], float))) for k in pk if k in z.files)
    vmax_p = max(float(np.nanmax(np.asarray(z[k], float))) for k in pk if k in z.files)
    mags = []
    for c in cases:
        ku, kv = ("u_%s_8" % FIELD_CASES[c["idx"]], "v_%s_8" % FIELD_CASES[c["idx"]])
        if ku in z.files and kv in z.files:
            mags.append(np.hypot(np.asarray(z[ku], float), np.asarray(z[kv], float)))
    vmin_u = min(float(np.nanmin(m)) for m in mags) if mags else None
    vmax_u = max(float(np.nanmax(m)) for m in mags) if mags else None
    for j, c in enumerate(cases):
        code = FIELD_CASES[c["idx"]]
        kp, ku = "p_%s_8" % code, "u_%s_8" % code
        kv = "v_%s_8" % code
        if kp in z.files:
            im = _draw_field(axes[0][j], z, kp, c["idx"], 8, "案例 %d：压力场" % c["idx"],
                             vmin=vmin_p, vmax=vmax_p)
            plt.colorbar(im, ax=axes[0][j], fraction=0.046)
        if ku in z.files and kv in z.files:
            mag = np.hypot(np.asarray(z[ku], float), np.asarray(z[kv], float))
            x, y = np.asarray(z["x"], float), np.asarray(z["y"], float)
            X, Y = np.meshgrid(x, y)
            im2 = axes[1][j].pcolormesh(X, Y, mag, cmap="viridis", shading="auto",
                                        vmin=vmin_u, vmax=vmax_u)
            s = surface(c["params"])
            axes[1][j].plot(s[:, 0], s[:, 1], "-", color="w", lw=1.0)
            axes[1][j].set_xlim(-0.35, 1.75); axes[1][j].set_ylim(-0.85, 0.85)
            axes[1][j].set_aspect("equal")
            axes[1][j].set_title("案例 %d：速度大小" % c["idx"], fontsize=9.5)
            axes[1][j].set_xlabel("弦向 x/c", fontsize=9)
            axes[1][j].tick_params(labelsize=8)
            plt.colorbar(im2, ax=axes[1][j], fraction=0.046)
    fig.suptitle("图 10　五个案例在 8° 攻角下的流场（上排压力、下排速度大小；"
                 "同一排共用色标，白线=翼型轮廓）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fn = "fig10_fields_all.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "cases_field.npz")


def fig_field_aoa():
    """攻角的影响：成功案例 4°/8°/10°，以及失败案例 3 在 8°/10° 的对比。"""
    z = _fields()
    if z is None:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    k1 = ["p_case1_success_%d" % a for a in (4, 8, 10) if "p_case1_success_%d" % a in z.files]
    v1 = (min(float(np.nanmin(np.asarray(z[k], float))) for k in k1),
          max(float(np.nanmax(np.asarray(z[k], float))) for k in k1)) if k1 else (None, None)
    k3 = ["p_case3_fail_mach_%d" % a for a in (8, 10) if "p_case3_fail_mach_%d" % a in z.files]
    v3 = (min(float(np.nanmin(np.asarray(z[k], float))) for k in k3),
          max(float(np.nanmax(np.asarray(z[k], float))) for k in k3)) if k3 else (None, None)
    for j, aoa in enumerate((4, 8, 10)):
        k = "p_case1_success_%d" % aoa
        if k in z.files:
            im = _draw_field(axes[0][j], z, k, 1, aoa, "成功案例：攻角 %d°" % aoa,
                             vmin=v1[0], vmax=v1[1])
            plt.colorbar(im, ax=axes[0][j], fraction=0.046)
    for j, aoa in enumerate((8, 10)):
        k = "p_case3_fail_mach_%d" % aoa
        if k in z.files:
            im = _draw_field(axes[1][j], z, k, 3, aoa, "失败案例：攻角 %d°" % aoa,
                             vmin=v3[0], vmax=v3[1])
            plt.colorbar(im, ax=axes[1][j], fraction=0.046)
    k = "p_case3_fail_mach_10"
    if k in z.files:
        cp = np.asarray(z[k], float)
        axes[1][2].axis("off")
        axes[1][2].text(0.0, 0.92, "为什么算失败：", fontsize=11, transform=axes[1][2].transAxes)
        axes[1][2].text(0.0, 0.72,
                        "① 10° 时前缘吸力峰极强\n    （压力系数最低约 −11.6），\n"
                        "    局部马赫数代理升到 0.643，\n    超过 0.6 的限制；\n\n"
                        "② 同一形状 10° 的升力只有 0.234，\n    远低于 8° 的 1.172 ——\n"
                        "    攻角带内性能塌陷。",
                        fontsize=10, transform=axes[1][2].transAxes, va="top")
    fig.suptitle("图 11　攻角对流场的影响：成功案例在带内变化平稳，失败案例在 10° 出现极端吸力峰",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fn = "fig11_field_aoa.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "cases_field.npz + cases_eval_s8.npz")


# --------------------------------------------- 图 12 真实样本锚定（模型可信度）
def fig_anchor():
    p = os.path.join(RES, "anchor_real.npz")
    if not os.path.isfile(p):
        print("  (缺 anchor_real.npz，跳过锚定图)")
        return
    z = np.load(p, allow_pickle=True)
    names = np.array([str(x) for x in z["names"]])
    gt = np.asarray(z["cl_official"], float)
    cm = np.asarray(z["cl_model"], float)
    man = json.load(open(os.path.join(ROOT, "03_code", "probe", "manifest.json"),
                         encoding="utf-8"))
    te = set(man["full_test"])
    m_te = np.array([n in te for n in names])
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
    lim = [gt.min() - 0.05, gt.max() + 0.05]
    axes[0].scatter(gt[~m_te], cm[~m_te], s=16, alpha=0.6, label="训练集内样本")
    axes[0].scatter(gt[m_te], cm[m_te], s=22, alpha=0.85, color="crimson",
                    label="官方留出集（模型没见过）")
    axes[0].plot(lim, lim, "--", color="0.4", lw=1)
    axes[0].set_xlabel("数据库给出的升力系数（真值）")
    axes[0].set_ylabel("模型给出的升力系数")
    axes[0].set_title("模型与真值几乎落在一条线上")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim); axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8.5, loc="upper left")
    rows = _rows("anchor_deciles.csv")
    if rows:
        b = [float(r["bias_mean_pct"]) for r in rows]
        axes[1].bar(np.arange(1, len(b) + 1), b, color="#1f77b4")
        axes[1].axhline(0, color="0.3", lw=1)
        axes[1].set_xticks(np.arange(1, len(b) + 1))
        axes[1].set_xlabel("真值升力从低到高分成十档（1=最低）")
        axes[1].set_ylabel("模型平均偏差（%）")
        axes[1].set_title("偏差只出现在升力接近 0 的档位")
        axes[1].grid(alpha=0.3, axis="y")
    nb = os.path.join(RES, "anchor_neighbour.npz")
    if os.path.isfile(nb):
        nz = np.load(nb, allow_pickle=True)
        go, cmm = np.asarray(nz["cl_official"], float), np.asarray(nz["cl_model"], float)
        axes[2].scatter(go, cmm, s=40, color="crimson",
                        label="设计点附近最近的 12 个真实样本")
        if "ga_best_model_CL" in nz.files:
            bc = float(nz["ga_best_model_CL"])
            axes[2].axhline(bc, ls="--", lw=1.3, color="crimson",
                            label="本文最优设计的模型升力 %.3f（真值未验证）" % bc)
        lo = min(go.min(), cmm.min()) - 0.05
        hi = max(max(go.max(), cmm.max()), float(nz["ga_best_model_CL"])) + 0.05
        axes[2].plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
        axes[2].set_xlabel("数据库给出的升力系数")
        axes[2].set_ylabel("模型给出的升力系数")
        axes[2].set_title("设计点邻域的真值核对（邻域并不紧）")
        axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("图 12　模型可信度核对：用数据库里已有的真实结果反过来检查模型", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fn = "fig12_anchor.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "anchor_real.npz + anchor_deciles.csv + anchor_neighbour.npz")


# --------------------------------------------- 图 13 反演：目标/反演/模型地板
def fig_inversion(pick=("45.931", "47.789")):
    files = sorted(glob.glob(os.path.join(RES, "invert_*.npz")))
    if not files:
        print("  (无 invert_*.npz，跳过反演图)")
        return
    sel = [f for f in files if any(k in os.path.basename(f) for k in pick)][:2]
    if not sel:
        return
    fig, axes = plt.subplots(len(sel), 3, figsize=(15.4, 4.3 * len(sel)))
    if len(sel) == 1:
        axes = axes[None, :]
    for i, f in enumerate(sel):
        z = np.load(f, allow_pickle=True)
        key = os.path.basename(f).replace("invert_", "").replace(".npz", "")
        tgt = np.asarray(z["target_cp"], float).ravel()
        bc = np.asarray(z["best_cp"], float).ravel()
        at = np.asarray(z["cp_at_true"], float).ravel()
        idx = np.arange(tgt.size)
        axes[i][0].plot(idx, tgt, "-", color="crimson", lw=1.6, label="目标压力（数据库真值）")
        axes[i][0].plot(idx, bc, "--", color="navy", lw=1.6,
                        label="反演解的压力（相对差 %.3f）" % float(z["best_cp_relL2"]))
        if at.size == tgt.size:
            axes[i][0].plot(idx, at, ":", color="0.35", lw=1.4,
                            label="用真形状算出来的压力（相对差 %.3f）"
                                  % float(z["cp_true_relL2"]))
        axes[i][0].set_xlabel("壁面点序号（0 与 255 都在前缘）")
        axes[i][0].set_ylabel("压力系数 $C_p$")
        axes[i][0].set_title("%s：三条压力曲线" % key[:30], fontsize=10)
        axes[i][0].grid(alpha=0.3); axes[i][0].legend(fontsize=8)
        h = np.asarray(z["dist_history"], float).ravel()
        axes[i][1].plot(np.arange(h.size), h, "-o", ms=3, color="darkgreen",
                        label="反演搜索的最好结果")
        axes[i][1].axhline(float(z["cp_true_relL2"]), ls=":", color="0.35", lw=1.4,
                           label="模型地板（真形状处）%.3f" % float(z["cp_true_relL2"]))
        axes[i][1].set_xlabel("反演迭代代数"); axes[i][1].set_ylabel("压力曲线相对差")
        axes[i][1].set_title("搜索收敛，并且穿过了地板线", fontsize=10)
        axes[i][1].grid(alpha=0.3); axes[i][1].legend(fontsize=8)
        tp = np.asarray(z["true_params"], float)
        rp = np.asarray(z["best_params"], float)
        lbl = ["L\n(弯度大小)", "P\n(弯度位置)", "Q\n(正/反弯)", "T\n(厚度)"]
        rng = np.array([3.989, 4.964, 1.0, 14.981])
        base = np.array([0.004, 3.005, 0.0, 5.008])
        xx = np.arange(4)
        tv, rv = (tp - base) / rng, (rp - base) / rng
        axes[i][2].bar(xx - 0.19, tv, 0.36, label="真形状", color="#1f77b4")
        axes[i][2].bar(xx + 0.19, rv, 0.36, label="反演解", color="#d62728")
        axes[i][2].set_xticks(xx); axes[i][2].set_xticklabels(lbl, fontsize=8.5)
        axes[i][2].set_ylim(-0.12, 1.25)
        axes[i][2].set_ylabel("参数在允许范围内的位置（0=最小，1=最大）")
        axes[i][2].set_title("参数恢复：压力拟合好了，形状并没找回来", fontsize=10)
        axes[i][2].grid(alpha=0.3, axis="y"); axes[i][2].legend(fontsize=8.5, loc="lower right")
        for k in range(4):
            # 柱子高度为 0 时看不见，所以一律把原始数值写在柱子上/下
            axes[i][2].text(k - 0.19, max(tv[k], 0.0) + 0.03, "%.2f" % tp[k],
                            ha="center", fontsize=8, color="#1f77b4")
            axes[i][2].text(k + 0.19, max(rv[k], 0.0) + 0.03, "%.2f" % rp[k],
                            ha="center", fontsize=8, color="#d62728")
            if abs(tp[k] - rp[k]) > 0.15 * rng[k]:
                axes[i][2].annotate("差异大", (k, 1.15), ha="center", fontsize=9,
                                    color="crimson")
    fig.suptitle("图 13　压力曲线反演：目标、反演解、以及「用真形状也达不到」的模型地板",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fn = "fig13_inversion.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "results/invert_*.npz")


# --------------------------------------------- 图 14 反演的多解性
def fig_multisolution():
    files = sorted(glob.glob(os.path.join(RES, "invert_*.npz")))
    if not files:
        return
    tols = [1.01, 1.02, 1.05, 1.10]
    names, counts = [], {t: [] for t in tols}
    spreads = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        d = np.asarray(z["cp_relL2"], float)
        P = np.asarray(z["params"], float)
        r = d / d.min()
        names.append(os.path.basename(f).replace("invert_", "")[:12])
        for t in tols:
            counts[t].append(int((r <= t).sum()))
        near = P[r <= 1.01]
        lo = np.array([0.004, 3.005, 0.0, 5.008])
        hi = np.array([3.993, 7.969, 1.0, 19.989])
        if len(near) > 1:
            spreads.append((near - lo) / (hi - lo))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
    x = np.arange(len(names))
    w = 0.2
    for k, t in enumerate(tols):
        axes[0].bar(x + (k - 1.5) * w, counts[t], w,
                    label="压力差在最优值 %.0f%% 以内" % ((t - 1) * 100))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=60, ha="right", fontsize=7.5)
    axes[0].set_ylabel("同一压力曲线下的「不同形状」个数（共 60 个解）")
    axes[0].set_title("几乎一样的压力，对应几十个不同形状")
    axes[0].grid(alpha=0.3, axis="y"); axes[0].legend(fontsize=8.5)
    if spreads:
        axes[1].boxplot(np.vstack(spreads), tick_labels=["L", "P", "Q", "T"])
        axes[1].set_ylabel("参数在允许范围内的位置（0=最小，1=最大）")
        axes[1].set_title("这些「等价形状」的参数散布")
        axes[1].grid(alpha=0.3)
    fig.suptitle("图 14　反演的多解性：压力分布反推几何不是一个唯一可解的问题", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fn = "fig14_multisolution.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "results/invert_*.npz（cp_relL2 与 params）")


# --------------------------------- 图 15 新设计与数据库里最好的真实翼型对照
DB_ROWS = [("本文新设计（案例 1）", 1, "case1_success", (0.7789, 1.3512, 1.5579), 0.5723),
           ("数据库设计点最优", None, "db_designbest", None, None),
           ("数据库工作带最优", None, "db_bandbest", None, None)]


def fig_db_compare(cases):
    """新设计 vs 数据库里两个最好的真实翼型：形状、8° 压力场、带内升力。"""
    z = _fields()
    pz = os.path.join(RES, "cases_eval_db.npz")
    if z is None or not os.path.isfile(pz):
        print("  (缺 cases_field.npz 或 cases_eval_db.npz，跳过对照图)")
        return
    zd = np.load(pz, allow_pickle=True)
    dp = np.asarray(zd["params"], float)          # 三行：数据库设计点最优/带内最优/本文案例1
    rows = [("本文新设计（案例 1）", dp[2], zd, 2, "case1_success"),
            ("数据库设计点最优", dp[0], zd, 0, "db_designbest"),
            ("数据库工作带最优", dp[1], zd, 1, "db_bandbest")]
    keys = ["p_%s_8" % r[4] for r in rows]
    vmin = min(float(np.nanmin(np.asarray(z[k], float))) for k in keys if k in z.files)
    vmax = max(float(np.nanmax(np.asarray(z[k], float))) for k in keys if k in z.files)
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.0))
    for j, (lab, pr, _zd, i, code) in enumerate(rows):
        g = geom128(pr)
        x = np.linspace(0, 1, 64)
        axes[0][j].fill_between(x, g[64:][::-1], g[:64][::-1], color="#cfe0f5")
        axes[0][j].plot(x, g[:64][::-1], color="crimson", lw=1.7)
        axes[0][j].plot(x, g[64:][::-1], color="navy", lw=1.7)
        axes[0][j].set_aspect("equal"); axes[0][j].set_xlim(-0.05, 1.05)
        axes[0][j].set_ylim(-0.22, 0.32)
        axes[0][j].set_xlabel("弦向 x/c"); axes[0][j].set_ylabel("y/c")
        axes[0][j].set_title("%s\nL=%.2f P=%.2f Q=%d T=%.2f"
                             % (lab, pr[0], pr[1], int(pr[2]), pr[3]), fontsize=9.5)
        axes[0][j].grid(alpha=0.25)
        k = "p_%s_8" % code
        if k in z.files:
            im = _draw_field(axes[1][j], z, k, 1, 8, "8° 攻角压力场",
                             vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=axes[1][j], fraction=0.046)
    fig.suptitle("图 15　本文新设计与数据库里两个最好的真实翼型的对照"
                 "（上排形状、下排同一色标下的 8° 压力场）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fn = "fig15_db_compare.png"
    SAVE(fig, os.path.join(FIGS, fn))
    _note(fn, "cases_eval_db.npz + cases_field.npz")


def write_index():
    p = os.path.join(FIGS, "REPORT_FIG_INDEX.md")
    os.makedirs(FIGS, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("# 报告插图 → 原始数据 索引\n\n")
        f.write("> 由 `03_code/inversion/plot_report_figs.py` 自动生成；"
                "插图与报告同目录（便于 Markdown 直接显示）；"
                "每张图都能由 `results/` 下的数据重画。\n\n")
        f.write("| 图 | 原始数据/来源 |\n|---|---|\n")
        for fn, cap in MAKE:
            f.write("| `%s` | %s |\n" % (fn, cap))
    print("wrote", p, "（%d 张）" % len(MAKE))


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    cases = case_table()
    if not cases:
        print("!! 缺 cases_eval_s8.csv，先跑案例评估")
        raise SystemExit(1)
    fig_workflow()
    fig_design_space(cases)
    fig_pareto_all(cases)
    fig_convergence()
    fig_gradcheck()
    fig_case_shapes(cases)
    fig_band_lift(cases)
    fig_cp_all(cases)
    fig_case_detail(cases)
    fig_fields_all(cases)
    fig_field_aoa()
    fig_anchor()
    fig_inversion()
    fig_multisolution()
    fig_db_compare(cases)
    write_index()
    print("完成（报告图全部）")




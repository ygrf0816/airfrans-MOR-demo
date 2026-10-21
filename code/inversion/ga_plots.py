#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二阶段的所有图：**只读已落盘的数据**画图（不现场推理），并生成"图 → 原始数据"索引。

原则（用户定的铁律）：
  1) 能可视化的都可视化；
  2) 每张图背后的原始数据必须已落盘（CSV / npz），图必须能由这些数据重画。

输入（都在 04_results/inversion/results/）：
  ga_front<TAG>.csv / ga_all<TAG>.csv / ga_history<TAG>.csv / ga_front<TAG>.npz
  ga_snaps<TAG>.npz（每若干代的种群快照，供画"前沿演化"）
  grid_all<TAG>.csv
  invert_<sample>.npz（反演：目标/还原 Cp、收敛历史）
输出：04_results/inversion/figs/*.png  +  FIG_DATA_INDEX.md

用法（本地或工作站都行，只要有 matplotlib）：
  python ga_plots.py --tag _s42
  python ga_plots.py --tag _s42 --grid-tag _full
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "04_results", "inversion", "results")
FIGS = os.path.join(ROOT, "04_results", "inversion", "figs")
sys.path.insert(0, os.path.join(ROOT, "03_code", "explain"))
try:
    import plot_style  # 复用第一阶段的统一样式（字体/字号）
    SAVE = plot_style.save
except Exception:      # 工作站上没有 plot_style 时的兜底
    def SAVE(fig, path):
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)

DB_BEST = 1.1464      # 数据库内 5 位族最优 CL
REACH_HI = 1.4225     # 可达上界
INDEX = []


def _rows(name):
    p = os.path.join(RES, name)
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add(figfile, src, desc):
    INDEX.append((figfile, src, desc))
    print("  ->", figfile, "|", desc)


# ------------------------------------------------------------------ GA 图
def _dedup(rows):
    """同一个解在旧版产物里可能重复出现（前沿里出现过 idx 相同的两行）——按参数去重。"""
    seen, out = set(), []
    for r in rows:
        k = tuple(round(float(r[x]), 9) for x in ("L", "P", "Q", "T"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def plot_pareto(tag):
    allr = _dedup(_rows("ga_all%s.csv" % tag))
    if not allr:
        return
    front = [r for r in allr if r["rank"] == "0"]
    feas = [r for r in allr if r["feasible"] == "True"]
    inf = [r for r in allr if r["feasible"] != "True"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if inf:
        ax.scatter([float(r["delta"]) for r in inf], [float(r["CL8"]) for r in inf],
                   s=14, c="0.6", label="不可行解")
    if feas:
        ax.scatter([float(r["delta"]) for r in feas], [float(r["CL8"]) for r in feas],
                   s=18, c="#1f77b4", label="可行解")
    if front:
        fr = sorted(front, key=lambda r: float(r["delta"]))
        ax.plot([float(r["delta"]) for r in fr], [float(r["CL8"]) for r in fr],
                "-o", ms=3, lw=1.0, color="crimson", label="帕累托前沿")
    snap = os.path.join(RES, "ga_snaps%s.npz" % tag)
    if os.path.isfile(snap):
        z = np.load(snap, allow_pickle=True)
        for k in range(0, len(z["gens"]), max(1, len(z["gens"]) // 4)):
            F = z["F%d" % k] if ("F%d" % k) in z.files else None
            if F is None:
                continue
            ax.scatter(F[:, 1], -F[:, 0], s=6, alpha=0.35,
                       label="第 %d 代种群" % int(z["gens"][k]))
    ax.axhline(DB_BEST, ls="--", lw=1, color="0.3")
    ax.text(0.02, DB_BEST + 0.01, "数据库内最优 %.4f" % DB_BEST, fontsize=9, color="0.3")
    ax.axhline(REACH_HI, ls=":", lw=1, color="0.5")
    ax.text(0.02, REACH_HI + 0.01, "可达上界 %.4f（超出即不可达）" % REACH_HI,
            fontsize=9, color="0.5")
    ax.set_xlabel("带内损失 Δ（4°/10° 相对 8° 的最大掉幅）")
    ax.set_ylabel("$C_L(8°)$")
    ax.set_title("帕累托前沿：升力 vs 带内稳健（%s）" % tag.strip("_"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    p = os.path.join(FIGS, "ga_pareto%s.png" % tag)
    SAVE(fig, p)
    add(os.path.basename(p), "ga_all%s.csv + ga_snaps%s.npz" % (tag, tag),
        "帕累托前沿（升力 vs 带内损失），叠加若干代种群快照与两条参照线")


REF = [0.5, 1.5]      # 超体积参考点（与 ga_design.py 的 REF 相同）


def hv_fix(F, ref=REF):
    """正确的二维超体积（最小化）：先剔被支配点，再按 f1 升序累加条带面积。

    history CSV 里的 `hypervolume` 列是**旧口径**（前沿越密数值越小，方向是反的），
    所以这里一律用落盘快照里的原始 F 重算；没有快照时才退回 history 列。
    """
    F = np.asarray(F, float)
    F = F[np.all(F <= ref, axis=1)]
    if len(F) == 0:
        return 0.0
    nd = [p for p in F
          if not np.any(np.all(F <= p, axis=1) & np.any(F < p, axis=1))]
    pts = np.array(nd)[np.argsort(np.array(nd)[:, 0])]
    hv = 0.0
    for i, (f1, f2) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
        hv += (nxt - f1) * (ref[1] - f2)
    return float(hv)


def plot_convergence(tag):
    h = _rows("ga_history%s.csv" % tag)
    if not h:
        return
    g = [int(r["gen"]) for r in h]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(g, [float(r["best_feasible_CL8"]) for r in h], "-o", ms=3)
    ax[0].axhline(DB_BEST, ls="--", color="0.4")
    ax[0].set_title("可行解里最优 $C_L(8°)$")
    ax[0].set_xlabel("代"); ax[0].set_ylabel("$C_L$"); ax[0].grid(alpha=0.3)
    ax[1].plot(g, [float(r["min_delta_feasible"]) for r in h], "-o", ms=3, color="darkgreen")
    ax[1].set_title("可行解里最小带内损失 Δ")
    ax[1].set_xlabel("代"); ax[1].set_ylabel("Δ"); ax[1].grid(alpha=0.3)
    ax[2].plot(g, [int(r["n_feasible"]) for r in h], "-o", ms=3, color="purple")
    ax[2].set_title("可行解个数")
    ax[2].set_xlabel("代"); ax[2].set_ylabel("个数"); ax[2].grid(alpha=0.3)
    snap = os.path.join(RES, "ga_snaps%s.npz" % tag)
    src = "ga_history%s.csv" % tag
    if os.path.isfile(snap):
        z = np.load(snap, allow_pickle=True)
        gs, hvs = [], []
        for k in range(len(z["gens"])):
            F, G = z["F%d" % k], z["G%d" % k]
            ok = np.all(G <= 0, axis=1)
            gs.append(int(z["gens"][k]))
            hvs.append(hv_fix(F[ok] if ok.any() else F))
        a = ax[2].twinx()
        a.plot(gs, hvs, "--s", ms=3, color="orange", lw=1.2)
        a.set_ylabel("超体积 HV（修正口径）", color="orange")
        src = "ga_history%s.csv + ga_snaps%s.npz（HV 由快照原始 F 重算）" % (tag, tag)
    elif "hypervolume" in h[0]:
        a = ax[2].twinx()
        a.plot(g, [float(r["hypervolume"]) for r in h], "--", color="orange", lw=1)
        a.set_ylabel("超体积 HV（history 旧口径）", color="orange")
    fig.suptitle("进化搜索收敛过程（%s）" % tag.strip("_"))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(FIGS, "ga_convergence%s.png" % tag)
    SAVE(fig, p)
    add(os.path.basename(p), src,
        "逐代：最优可行 CL8 / 最小 Δ / 可行解数 / 超体积（HV 用修正口径重算）")


def plot_params(tag):
    allr = _dedup(_rows("ga_all%s.csv" % tag))
    front = [r for r in allr if r["rank"] == "0"]
    if not front:
        return
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for k, (key, lab) in enumerate([("L", "L（设计升力系数 $c_{l,des}$=0.15L）"),
                                    ("P", "P（最大弯度位置 $x_f$=P/20）"),
                                    ("T", "T（相对厚度 $t_c$=T/100）")]):
        v = [float(r[key]) for r in front]
        ax[k].hist(v, bins=16, color="#1f77b4", alpha=0.85)
        ax[k].set_title(lab); ax[k].grid(alpha=0.3); ax[k].set_ylabel("解数")
    q = [r["Q"] for r in front]
    ax[3].bar([str(x) for x in (0, 1)], [q.count("0"), q.count("1")], color="#d62728")
    ax[3].set_title("Q（0 正常弯 / 1 反弯）")
    fig.suptitle("帕累托前沿解集在参数空间的分布（%s）" % tag.strip("_"))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = os.path.join(FIGS, "ga_solset%s.png" % tag)
    SAVE(fig, p)
    add(os.path.basename(p), "ga_all%s.csv（rank=0 的行）" % tag,
        "前沿解集在四个设计参数上的分布（含 Q 的取值计数）")


def plot_cp(tag, topk=5):
    p = os.path.join(RES, "ga_front%s.npz" % tag)
    if not os.path.isfile(p):
        return
    z = np.load(p, allow_pickle=True)
    cl8 = np.asarray(z["CL8"], float)
    order = np.argsort(-cl8)[:topk]
    keys = [(k, "%s°" % a) for k, a in (("cp4", "4"), ("cp8", "8"), ("cp10", "10"))
            if k in z.files]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.0), squeeze=False)
    for j, (k, lab) in enumerate(keys):
        cp = np.asarray(z[k], float)
        idx = np.arange(cp.shape[1])          # 下标即采样点序号，不是弦向坐标
        for i in order:
            axes[0][j].plot(idx, cp[i], lw=1.2, label="CL8=%.3f" % cl8[i])
        axes[0][j].set_title("Cp 曲线（攻角 %s°）" % lab)
        axes[0][j].set_xlabel("壁面点序号（0 与 %d 都在前缘，约 %d 在后缘）"
                              % (cp.shape[1] - 1, (cp.shape[1] - 1) // 2))
        axes[0][j].set_ylabel("$C_p$")
        axes[0][j].grid(alpha=0.3); axes[0][j].legend(fontsize=8)
    fig.suptitle("前沿前 %d 个解的 Cp 曲线（%s）" % (topk, tag.strip("_")))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    q = os.path.join(FIGS, "ga_cp_top%s.png" % tag)
    SAVE(fig, q)
    add(os.path.basename(q), "ga_front%s.npz（cp4/cp8/cp10）" % tag,
        "前沿前 %d 个解的 Cp 曲线（三个攻角；横轴为壁面点序号，非弦向坐标）" % topk)


def plot_shapes(tag, topk=5):
    p = os.path.join(RES, "ga_front%s.npz" % tag)
    if not os.path.isfile(p):
        return
    z = np.load(p, allow_pickle=True)
    G = np.asarray(z["geometry"], float)
    cl8 = np.asarray(z["CL8"], float)
    order = np.argsort(-cl8)[:topk]
    x = np.linspace(0, 1, 64)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i in order:
        g = G[i]
        # 几何向量是余弦站位、TE→LE 顺序（见 ga_design.py 口径说明）：反转让 x 从 LE 到 TE
        ax.plot(x, g[:64][::-1], color="crimson", lw=1.3, alpha=0.85)
        ax.plot(x, g[64:][::-1], color="navy", lw=1.3, alpha=0.85)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color="crimson"), Line2D([], [], color="navy")],
              ["上表面 y/c", "下表面 y/c"], fontsize=9)
    ax.set_xlabel("x/c（前缘→后缘）"); ax.set_ylabel("y/c")
    ax.set_title("前沿前 %d 个解的形状（%s）" % (topk, tag.strip("_")))
    ax.grid(alpha=0.3); ax.set_aspect("equal")
    q = os.path.join(FIGS, "ga_shapes%s.png" % tag)
    SAVE(fig, q)
    add(os.path.basename(q), "ga_front%s.npz（geometry，64+64）" % tag,
        "前沿前 %d 个解的实际形状（上下表面 y/c）" % topk)


# ------------------------------------------------------------------ 网格图
def plot_grid(tag):
    rows = _rows("grid_all%s.csv" % tag)
    if not rows:
        return
    Ls = sorted({round(float(r["L"]), 4) for r in rows})
    Ts = sorted({round(float(r["T"]), 4) for r in rows})
    for Q in ("0", "1"):
        sub = [r for r in rows if (abs(float(r["Q"])) < 0.5) == (Q == "0")]
        if not sub:
            continue
        M = np.full((len(Ts), len(Ls)), np.nan)     # 可行解里最优 CL8
        D = np.full((len(Ts), len(Ls)), np.nan)     # 带内损失 Δ（同格取最小）
        F = np.full((len(Ts), len(Ls)), np.nan)     # 可行比例
        for i in range(len(Ts)):
            for j in range(len(Ls)):
                cell = [r for r in sub
                        if round(float(r["T"]), 4) == Ts[i] and round(float(r["L"]), 4) == Ls[j]]
                if not cell:
                    continue
                ok = [r for r in cell if r["feasible"] == "True"]
                F[i, j] = len(ok) / len(cell)
                if ok:
                    M[i, j] = max(float(r["CL8"]) for r in ok)
                    D[i, j] = min(float(r["delta"]) for r in ok)
        fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.4))
        for ax, A, ttl, cm, lo, hi in (
                (axes[0], M, "$C_L(8°)$（可行解内最优）", "viridis", DB_BEST, REACH_HI),
                (axes[1], D, "带内损失 Δ（同格取最小）", "magma", None, None),
                (axes[2], F, "可行比例（同格 P 上）", "cividis", 0.0, 1.0)):
            im = ax.pcolormesh(np.array(Ls), np.array(Ts), A, cmap=cm, shading="nearest")
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("L（设计升力系数 $c_{l,des}$=0.15L）")
            ax.set_ylabel("T（相对厚度 $t_c$=T/100）")
            ax.set_title(ttl)
            if cm == "viridis":
                ax.contour(np.array(Ls), np.array(Ts), np.nan_to_num(A, nan=-9),
                           levels=[DB_BEST], colors="w", linewidths=1.4)
        fig.suptitle("参数网格穷举地图（Q=%s，%s）—— 空白格 = 该格内无可行解；边界案例取自可行域边缘"
                     % (Q, tag.strip("_")))
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        p = os.path.join(FIGS, "grid_heatmap_q%s%s.png" % (Q, tag))
        SAVE(fig, p)
        add(os.path.basename(p), "grid_all%s.csv" % tag,
            "网格穷举的 CL8 / Δ / 可行比例地图（Q=%s），用于定位可行域边界与边界案例" % Q)


# ------------------------------------------------------------- 反演（S6/S7）
def _invert_files():
    import glob
    return sorted(glob.glob(os.path.join(RES, "invert_*.npz")))


def plot_inversion(tag="", rel=1.10):
    """Cp 反演：目标 vs 反演 Cp、收敛历史、以及"近优解参数散布"（多解性）。"""
    fs = _invert_files()
    if not fs:
        print("  (没有 invert_*.npz，跳过反演图)")
        return
    names, bestd, spread = [], [], []
    for p in fs:
        z = np.load(p, allow_pickle=True)
        tgt = np.asarray(z["target_cp"], float).ravel()
        bc = np.asarray(z["best_cp"], float).ravel()
        if bc.size != tgt.size:            # 旧版产物把"还原 Cp"存成了标量，不能用
            print("  !! 跳过 %s：best_cp 不是曲线（size=%d）" % (os.path.basename(p), bc.size))
            continue
        h = np.asarray(z["dist_history"], float).ravel()
        d = np.asarray(z["cp_relL2"], float).ravel()
        P = np.asarray(z["params"], float)
        dmin = float(z["best_cp_relL2"])
        floor = (float(z["cp_true_relL2"]) if "cp_true_relL2" in z.files
                 else float("nan"))
        base = os.path.basename(p)[:-4]
        nm = base.replace("invert_", "")
        idx = np.arange(tgt.size)
        fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.0))
        ax[0].plot(idx, tgt, "-", color="crimson", lw=1.5, label="目标 Cp（真实样本）")
        ax[0].plot(idx, bc, "--", color="navy", lw=1.5,
                   label="反演解 Cp（相对 L2=%.3f）" % dmin)
        if "cp_at_true" in z.files and np.asarray(z["cp_at_true"]).size == tgt.size:
            ax[0].plot(idx, np.asarray(z["cp_at_true"], float), ":", color="0.35", lw=1.3,
                       label="真参数处的模型 Cp（地板 %.3f）" % floor)
        ax[0].set_xlabel("壁面点序号（0 与 255 在前缘）"); ax[0].set_ylabel("$C_p$")
        ax[0].set_title("目标 vs 反演 Cp"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax[1].plot(np.arange(h.size), h, "-o", ms=3, color="darkgreen",
                   label="反演搜索：最优 Cp 距离")
        if floor == floor:          # 不是 NaN
            ax[1].axhline(floor, ls=":", color="0.35", lw=1.3,
                          label="模型地板（真参数处）%.3f" % floor)
        ax[1].set_xlabel("代"); ax[1].set_ylabel("最优 Cp 相对 L2 距离")
        ax[1].set_title("反演收敛 vs 模型地板"); ax[1].grid(alpha=0.3)
        ax[1].legend(fontsize=8)
        fig.suptitle("反演检验：%s" % nm)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        q = os.path.join(FIGS, "invert_%s.png" % nm[:40])
        SAVE(fig, q)
        add(os.path.basename(q), "invert_%s.npz" % nm[:32],
            "目标 Cp 与反演解 Cp 的对比 + 反演收敛历史")
        names.append(nm)
        bestd.append(dmin)
        near = P[d <= max(rel * dmin, dmin + 1e-9)]
        spread.append(near)

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.2))
    ax[0].bar(np.arange(len(names)), bestd, color="#1f77b4")
    ax[0].set_xticks(np.arange(len(names)))
    ax[0].set_xticklabels([n[:14] for n in names], rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel("Cp 相对 L2 距离（越小越准）")
    ax[0].set_title("各目标样本的反演精度"); ax[0].grid(alpha=0.3, axis="y")
    LO = np.array([0.004, 3.005, 0.0, 5.008])
    HI = np.array([3.993, 7.969, 1.0, 19.989])
    data = []
    for s in spread:
        if s.size:
            data.append(((s - LO) / (HI - LO)))     # 归一到 [0,1] 看散布
    if data:
        ax[1].boxplot(np.vstack(data), tick_labels=["L", "P", "Q", "T"])
    ax[1].set_ylabel("归一化参数（0=下界, 1=上界）")
    ax[1].set_title("近优解（距离 ≤ 最优×%.2f）的参数散布 = 多解性" % rel)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Cp 反演：精度与多解性")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    q = os.path.join(FIGS, "invert_ambiguity.png")
    SAVE(fig, q)
    add(os.path.basename(q), "invert_*.npz（cp_relL2, params）",
        "各目标反演精度 + 近优解参数散布（反演问题的多解性）")


# ------------------------------------------------------------- 真实锚定（S5）
def plot_anchor():
    p = os.path.join(RES, "anchor_real.npz")
    if not os.path.isfile(p):
        print("  (没有 anchor_real.npz，跳过锚定图)")
        return
    z = np.load(p, allow_pickle=True)
    gt = np.asarray(z["cl_official"], float)
    cA = np.asarray(z["cl_model"], float)
    cB = (np.asarray(z["cl_model_u62"], float)
          if "cl_model_u62" in z.files else None)
    fig, axes = plt.subplots(1, 2 if cB is not None else 1,
                             figsize=(6.2 * (2 if cB is not None else 1), 5.4),
                             squeeze=False)
    lim = [min(gt.min(), cA.min()) - 0.05, max(gt.max(), cA.max()) + 0.05]
    for j, (cl, ttl) in enumerate([(cA, "口径A：每个样本用各自 $U_\\infty$, AoA"),
                                   (cB, "口径B：全部固定 $U_\\infty$=62 m/s")]):
        if cl is None:
            continue
        ax = axes[0][j]
        ax.scatter(gt, cl, s=14, alpha=0.55, c=50 * np.ones_like(gt), cmap="viridis")
        ax.plot(lim, lim, "--", color="0.4", lw=1)
        rho = float(np.corrcoef(gt, cl)[0, 1])
        ax.set_xlabel("官方 $C_L$（真实样本）"); ax.set_ylabel("模型 $C_L$")
        ax.set_title("%s\nPearson=%.4f" % (ttl, rho))
        ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.suptitle("真实样本锚定：模型 $C_L$ vs 官方 $C_L$（511 个五位族样本）")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    q = os.path.join(FIGS, "anchor_scatter.png")
    SAVE(fig, q)
    add(os.path.basename(q), "anchor_real.npz",
        "模型 vs 官方 CL 散点（两种风速口径）+ 45 度参考线")

    h = _rows("anchor_deciles.csv")
    nb = os.path.join(RES, "anchor_neighbour.npz")
    if h or os.path.isfile(nb):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
        if h:
            b = [float(r["bias_mean_pct"]) for r in h]
            axes[0].bar(np.arange(1, len(b) + 1), b, color="#1f77b4")
            axes[0].axhline(0, color="0.3", lw=1)
            axes[0].set_xticks(np.arange(1, len(b) + 1))
            axes[0].set_xlabel("官方 CL 的十分位区间（1=最低）")
            axes[0].set_ylabel("模型平均偏差（%）")
            axes[0].set_title("偏差随升力水平的变化"); axes[0].grid(alpha=0.3, axis="y")
        if os.path.isfile(nb):
            nz = np.load(nb, allow_pickle=True)
            go, cm = np.asarray(nz["cl_official"], float), np.asarray(nz["cl_model"], float)
            axes[1].scatter(go, cm, s=34, color="crimson", label="GA 最优解邻域的真实样本")
            if "ga_best_model_CL" in nz.files:
                bc = float(nz["ga_best_model_CL"])
                axes[1].axhline(bc, ls="--", lw=1.2, color="crimson",
                                label="GA 最优解的模型 CL=%.4f（真值未验证）" % bc)
            lo = min(go.min(), cm.min()) - 0.03
            hi = max(go.max(), cm.max(), bc if "ga_best_model_CL" in nz.files else 0.0) + 0.03
            axes[1].plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
            axes[1].set_xlabel("官方 $C_L$"); axes[1].set_ylabel("模型 $C_L$")
            axes[1].set_title("设计点邻域的真值核对（%d 个最近样本）" % go.size)
            axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
        fig.suptitle("锚定细化：分位偏差与邻域核对")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        q = os.path.join(FIGS, "anchor_detail.png")
        SAVE(fig, q)
        add(os.path.basename(q), "anchor_deciles.csv + anchor_neighbour.npz",
            "模型偏差随升力水平的变化 + GA 最优解参数邻域内的真值核对")


# --------------------------------------------------------- 案例图（S8 用）
def plot_cases(tag=""):
    """成功/边界/失败三类案例：每个案例一张图（形状 + 三个攻角的 Cp + 关键指标）。"""
    p = os.path.join(RES, "cases_eval%s.npz" % tag)
    if not os.path.isfile(p):
        print("  (没有 cases_eval%s.npz，跳过案例图)" % tag)
        return
    z = np.load(p, allow_pickle=True)
    P = np.asarray(z["params"], float)
    G = np.asarray(z["geometry"], float)
    feas = np.asarray(z["feasible"]).astype(bool)
    x = np.linspace(0, 1, 64)
    for i in range(len(P)):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
        g = G[i]
        axes[0].plot(x, g[:64][::-1], color="crimson", lw=1.5, label="上表面 y/c")
        axes[0].plot(x, g[64:][::-1], color="navy", lw=1.5, label="下表面 y/c")
        axes[0].set_xlabel("x/c（前缘→后缘）"); axes[0].set_ylabel("y/c")
        axes[0].set_title("形状：L=%.3f P=%.3f Q=%d T=%.3f"
                          % (P[i][0], P[i][1], P[i][2], P[i][3]))
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9); axes[0].set_aspect("equal")
        for k, lab in (("cp4", "4°"), ("cp8", "8°"), ("cp10", "10°")):
            if k in z.files:
                axes[1].plot(np.arange(np.asarray(z[k]).shape[1]),
                             np.asarray(z[k])[i], lw=1.4, label=lab)
        axes[1].set_xlabel("壁面点序号（0 与 255 在前缘）"); axes[1].set_ylabel("$C_p$")
        axes[1].set_title("三个攻角的 Cp（%s）" % ("可行" if feas[i] else "不可行"))
        axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9, title="攻角")
        fig.suptitle("案例 %d：$C_L(8°)$=%.4f  Δ=%.4f  $m_{local}$=%.3f  分离代理=%.5f"
                     % (i + 1, float(np.asarray(z["CL8"])[i]),
                        float(np.asarray(z["delta"])[i]),
                        float(np.asarray(z["m_max"])[i]),
                        float(np.asarray(z["sep"])[i])))
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        q = os.path.join(FIGS, "case_%d%s.png" % (i + 1, tag))
        SAVE(fig, q)
        add(os.path.basename(q), "cases_eval%s.npz（params/cp4/cp8/cp10/geometry）" % tag,
            "案例 %d：形状 + 三攻角 Cp + 关键指标（可行性由约束判据给出）" % (i + 1))


def write_index():
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(ROOT, "04_results", "inversion", "FIG_DATA_INDEX.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# 第二阶段图表 → 原始数据 索引\n\n")
        f.write("> 铁律：能可视化的都可视化；**每张图的原始数据都已落盘**，图可由数据重画。\n")
        f.write("> 画图脚本：`03_code/inversion/ga_plots.py`（只读落盘数据，不现场推理）。\n\n")
        f.write("| 图 | 原始数据 | 说明 |\n|---|---|---|\n")
        for a, b, c in INDEX:
            f.write("| `figs/%s` | `results/%s` | %s |\n" % (a, b, c))
    print("wrote", p, "（%d 张图）" % len(INDEX))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="单个 tag，如 _s42")
    ap.add_argument("--tags", default=None,
                    help="逗号分隔的多个 tag，如 _s42,_s7,_s2024（索引汇总到同一份）")
    ap.add_argument("--grid-tag", default=None)
    ap.add_argument("--invert", action="store_true", help="同时画反演图（invert_*.npz）")
    ap.add_argument("--anchor", action="store_true", help="同时画真实锚定图（anchor_*）")
    ap.add_argument("--cases", type=str, default="", help="案例 tag 后缀（cases_eval<tag>.npz）")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    tags = ([t.strip() for t in a.tags.split(",") if t.strip()] if a.tags
            else [a.tag or "_s42"])
    for t in tags:
        print("画图（tag=%s）" % t)
        plot_pareto(t)
        plot_convergence(t)
        plot_params(t)
        plot_cp(t, a.topk)
        plot_shapes(t, a.topk)
    if a.grid_tag:
        plot_grid(a.grid_tag)
    if a.anchor:
        plot_anchor()
    if a.invert:
        plot_inversion()
    if a.cases:
        plot_cases(a.cases)
    write_index()

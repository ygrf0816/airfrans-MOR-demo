#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二阶段主方法：在 NACA 生成参数上用多目标进化搜索（NSGA-II 型）设计翼型。

设计变量（5 位族，范围由 1000 样本实测覆盖定界）：
    L ∈ [0.004, 3.993]   camber = 0.15*L
    P ∈ [3.005, 7.969]   x_f    = P/20
    Q ∈ {0, 1}           0 正常弯 / 1 反弯
    T ∈ [5.008, 19.989]  t_c    = T/100
工况：U∞ = 62 m/s 固定；AoA 设计点 8°，工作带 4°–10°。

目标向量（都按最小化写）：
    f1 = −CL(8°)                     升力最大
    f2 = Δ = max(CL(8°)−CL(4°), CL(8°)−CL(10°))   带内损失最小
约束（约束支配：可行解优先）：
    g1: m_local(8°) ≤ 0.6            （Cp 链可算的代理）
    g2: 分离代理 ≤ 阈值              （吸力面后段压力恢复不出现长平坦段）
    g3: 参数在范围内（由变量边界保证）
    g4: 几何合法性（厚度>0、不自交；由生成器天然满足，仍逐点复核）

关键实现口径（都已核实，不要改）：
  * 模型几何输入 = 128 维：64 个**余弦站位**（`_chord_nodes`：beta=pi*linspace(1,0,64,endpoint=False)，
    即 x 从 1 降到 ~0.0006，TE→LE 顺序），每站位取该 x 带内表面点的 max/min y（弦长归一化），
    前 64 = 上表面、后 64 = 下表面 —— 与 `geometry_input.geometric_vector` 完全一致。
  * 数据里的翼型**未按攻角旋转**（弦线水平、TE 在 y=0），生成器输出同款坐标系。
  * 升力 = 压力项积分，对 Cp 是线性的；权重 w(256) 由 `official_metrics.anchored_surface` 同款口径构造
    （grad_check.py 已验证：与参考积分相对差 1e-8）。

用法（在工作站上，用 pytorch 环境的 python）：
    python ga_design.py --selftest        # 先跑自检：生成几何 + 权重口径是否正确
    python ga_design.py --anchor          # 模型 vs 官方 CL 在 511 个真实样本上的排序核对
    python ga_design.py --run             # 跑 NSGA-II
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/home/zch/airfrans"
PRED = f"{ROOT}/prediction"
OUT = f"{ROOT}/04_results/inversion/results"
POD = f"{ROOT}/pod"
sys.path.insert(0, PRED)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_B_transformer import BBackbone  # noqa: E402
from input_side import input_vector  # noqa: E402
import naca_generator as NG  # noqa: E402  (官方源码，与本文件同目录)

N_ST = 64
U_INF = 62.0
BOUNDS = np.array([[0.004, 3.993],      # L
                   [3.005, 7.969],      # P
                   [0.0, 1.0],          # Q（二值，取整）
                   [5.008, 19.989]])    # T
TAG = ""            # 由 --tag 设置，让多种子/多配置的输出并存
AOA_DESIGN = 8.0
AOA_BAND = (4.0, 10.0)
M_LOCAL_MAX = 0.6
FLAT_THRESH = 0.05       # 分离代理阈值（见 sep_proxy）


# ---------------------------------------------------------------- 几何生成
def chord_nodes(n=N_ST):
    """与 geometry_input._chord_nodes 逐字一致：x 从 1 降到 ~0，余弦加密。"""
    beta = np.pi * np.linspace(1, 0, n, endpoint=False)
    return (1 - np.cos(beta)) / 2.0


def gen_geometry(params, nb=400):
    """(L,P,Q,T) -> 128 维几何向量（与数据集 geometry_input 同口径）。"""
    L, P, Q, T = params
    pos = NG.naca_generator((L, P, round(Q), T), nb_samples=nb,
                            cosine_spacing=True, verbose=False)
    X, Y = pos[:, 0], pos[:, 1]
    x0, x1 = X.min(), X.max()
    c = x1 - x0
    xs = chord_nodes(N_ST) * c + x0
    vec = np.zeros(2 * N_ST)
    for i, xx in enumerate(xs):
        band = np.abs(X - xx) < (c / N_ST) * 0.5
        ys = Y[band]
        if ys.size >= 2:
            vec[i] = ys.max() / c
            vec[N_ST + i] = ys.min() / c
        elif ys.size == 1:
            vec[i] = vec[N_ST + i] = ys[0] / c
    return vec


def gen_surface(params, nb=994):
    """(L,P,Q,T) -> 前缘锚定的闭合轮廓 (X(N,2), 外法向 n(N,2), 弧长微元 dl(N))。

    与 official_metrics.anchored_surface 同款口径：前缘 = x 最小点，起点为前缘，
    前半走一个表面；法向由轮廓切线旋转 90° 得到（外法向 = 指向离开翼型一侧）。
    """
    L, P, Q, T = params
    pos = NG.naca_generator((L, P, round(Q), T), nb_samples=nb,
                            cosine_spacing=True, verbose=False)
    P2 = pos[:, :2]
    # 闭环：去掉末尾重复点（生成器首尾都是 TE）
    if np.allclose(P2[0], P2[-1]):
        P2 = P2[:-1]
    N = len(P2)
    le = int(np.argmin(P2[:, 0]))
    X = np.roll(P2, -le, axis=0)                      # 前缘在 0
    nxt = np.roll(X, -1, axis=0)
    prv = np.roll(X, 1, axis=0)
    tang = nxt - prv
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-15)
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]])   # 切线转 -90°
    # 定朝向：这里**反过来取内法向**才与 official_metrics.anchored_surface 用的
    # vtp 法向一致（自检实测：外法向给出的 CL 符号相反、量级相同 0.26%）。
    cen = X.mean(axis=0)
    inward = -np.sign(((X - cen) * nrm).sum(axis=1))
    nrm = nrm * inward[:, None]
    d = np.linalg.norm(np.diff(np.vstack([X, X[:1]]), axis=0), axis=1)
    return X, nrm, d


def build_weight_256(params, aoa_deg, u_inf=U_INF):
    """Cp(256) -> CL 的线性权重（与 grad_check.py / official_metrics 同口径）。"""
    X, nrm, dl = gen_surface(params)
    a = np.deg2rad(aoa_deg)
    coef = -np.sin(a) * nrm[:, 0] + np.cos(a) * nrm[:, 1]
    s994 = np.cumsum(dl) / dl.sum()
    s994 = np.concatenate([[0.0], s994[:-1]])
    s256 = np.linspace(0, 1, 256)
    w = np.zeros(256)
    seg = s256[1] - s256[0]
    for j in range(len(s994)):
        sj = s994[j]
        if sj <= 0:
            w[0] += coef[j] * dl[j]
        elif sj >= 1:
            w[255] += coef[j] * dl[j]
        else:
            i = min(int(sj / seg), 254)
            frac = (sj - s256[i]) / seg
            w[i] += coef[j] * dl[j] * (1 - frac)
            w[i + 1] += coef[j] * dl[j] * frac
    # CL = Σ_j coef_j·dl_j·Cp(s994_j)：压力项积分的 q=0.5U∞² 与 2/U∞² 正好相消，
    # 所以权重就是 w（不要再多乘 2/U∞² —— 自检时多乘了一次，CL 直接小 1922 倍）。
    return w


def sep_proxy(cp):
    """分离代理：压力恢复平坦段长度（越小越好；实际窗口口径见下，别按名字想当然）。"平坦段"长度（越小越好）。

    已核对的实际口径：
      * 256 点是**闭环**：0 与 255 都在前缘、约 127 在后缘；`cp[:128]` 与 `cp[128:]`
        各是一条表面但方向相反。这里把 `cp[:128]` 反转后与 `cp[128:]` 配对，
        于是**两半方向一致（都是 TE→LE）**；再配 `x=linspace(0,1,128)`（LE→TE），
        掩码 [0.3,0.95] 实际对应**物理 x/c∈[0.05,0.70]**，不是名字暗示的后段。
      * 取平均 Cp 更负的一侧（吸力面），统计 |dCp/dx| < FLAT_THRESH 的最长连续段长度。
      * 取值**离散**：一格 = 1/127 ≈ 0.007874，所以 p90 阈值 0.007874 等价于"最多 1 格平坦"。
    该定义已在 511 个真实样本上标定（calibration.json），GA 与阈值口径一致；
    上述方向/窗口是已知口径怪癖，改动会让已跑结果不可比，故**保持不动**。
    """
    cp = np.asarray(cp).ravel()
    half = cp[:128][::-1], cp[128:]          # 两半都折成 LE->TE
    if half[0].mean() < half[1].mean():
        s = half[0]
    else:
        s = half[1]
    x = np.linspace(0, 1, 128)
    m = (x >= 0.3) & (x <= 0.95)
    xs, cs = x[m], s[m]
    grad = np.abs(np.diff(cs) / np.diff(xs))
    flat = grad < FLAT_THRESH
    best = cur = 0
    for f in flat:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return float(best * (xs[1] - xs[0]))


# ---------------------------------------------------------------- 模型
def load_model():
    st = torch.load(f"{PRED}/04_results/prediction/runs/full_B_official/best_model.pt",
                    map_location="cpu")
    ns = st["norm_stats"]
    model = BBackbone(dim_in=133, dim_out=int(ns["pod_r"]), hidden=256, n_layers=4)
    model.load_state_dict(st["model_state"])
    model.eval()
    return model, ns


def make_inputs(model, ns, params_list, aoa_list, geom_list=None, u_list=None):
    """(N,4) 参数 + (N,) 攻角 -> 标准化后的 133 维输入张量。

    u_list 给定时用每个样本各自的风速（锚定真实样本时必须这样，否则会把风速差异
    混进"模型偏差"里）；给 None 时用设计点 U∞=62 m/s。
    """
    smean = np.asarray(ns["smean"]).ravel()
    sstd = np.asarray(ns["sstd"]).ravel()
    gmean = np.asarray(ns["gmean"]).ravel()
    gstd = np.asarray(ns["gstd"]).ravel()
    scal, geo = [], []
    for k, (pr, aoa) in enumerate(zip(params_list, aoa_list)):
        L, P, Q, T = pr
        u = U_INF if u_list is None else float(u_list[k])
        scal.append([u, aoa, 0.15 * L, T / 100.0, round(Q)])
        geo.append(geom_list[k] if geom_list is not None else gen_geometry(pr))
    scal = (np.array(scal) - smean) / sstd
    geo = (np.array(geo) - gmean) / gstd
    return torch.tensor(np.concatenate([scal, geo], axis=1), dtype=torch.float32)


def eval_batch(model, ns, params_list, aoa_list, geom_list=None, u_list=None):
    """返回 dict(CL=(N,), cp=(N,256), m_local=(N,), sep=(N,))。"""
    X = make_inputs(model, ns, params_list, aoa_list, geom_list, u_list)
    with torch.no_grad():
        alpha = model(X).numpy()
    Phi = np.asarray(ns["pod_Phi"])
    mu = np.asarray(ns["pod_mu"]).ravel()
    cp = mu[None, :] + alpha @ Phi.T
    cl = np.array([float(np.dot(build_weight_256(pr, a), c))
                   for pr, a, c in zip(params_list, aoa_list, cp)])
    m_loc = (U_INF / 343.0) * np.sqrt(np.clip(1.0 - cp.min(axis=1), 0.0, None))
    sep = np.array([sep_proxy(c) for c in cp])
    return {"CL": cl, "cp": cp, "m_local": m_loc, "sep": sep}


# ---------------------------------------------------------------- 自检 / 锚定
def selftest(model, ns):
    """用真实样本核对：参数->几何->权重->CL 这一套口径是否与既有链路一致。"""
    rows, zv = [], np.load(f"{POD}/geomvec_1000.npz", allow_pickle=True)
    gidx = {n: i for i, n in enumerate(zv["names"].tolist())}
    wz = np.load(f"{POD}/wallcp_1000.npz", allow_pickle=True)
    wsz = {n: i for i, n in enumerate(wz["names"].tolist())}
    oz = np.load(f"{POD}/official_coefs_1000.npz", allow_pickle=True)
    osz = {n: i for i, n in enumerate(oz["names"].tolist())}
    picks = [n for n in zv["names"].tolist()
             if len(str(n).replace("airFoil2D_SST_", "").split("_")) == 6][:3]
    print("=== 自检（真实样本：生成几何 vs 数据几何；生成权重 CL vs 真值 Cp 的 CL）===")
    for nm in picks:
        f = str(nm).replace("airFoil2D_SST_", "").split("_")
        L, P, Q, T = [float(x) for x in f[2:]]
        aoa = float(f[1])
        g_gen = gen_geometry((L, P, Q, T))
        g_dat = np.asarray(zv["G"][gidx[str(nm)]], dtype=float).ravel()
        dmax = float(np.abs(g_gen - g_dat).max())
        cp_true = np.asarray(wz["CP"][wsz[str(nm)]], dtype=float)
        cl_w = float(np.dot(build_weight_256((L, P, Q, T), aoa), cp_true))
        cl_off = float(oz["CL"][osz[str(nm)]])
        print("  %-40s 几何maxdiff=%.3e  权重CL=%.4f  官方CL=%.4f  差=%.1f%%"
              % (str(nm)[:40], dmax, cl_w, cl_off, 100 * (cl_w - cl_off) / cl_off))
    print("判据：几何 maxdiff 应 ~1e-3 量级（点密度不同导致的箱内极值差）；")
    print("      权重 CL 与官方 CL 相对差应 ~1–2%（与预测阶段 '压力项升力 vs 官方差 1.38%' 一致）。")


def _real_5digit(zv, ozm):
    """从 geomvec/official 两个 npz 里按**名字**对齐取出 511 个 5 位族真实样本。"""
    names, gt, pf, aoas, us, geom = [], [], [], [], [], []
    for i, n in enumerate(zv["names"].tolist()):
        f = str(n).replace("airFoil2D_SST_", "").split("_")
        if len(f) != 6:
            continue
        names.append(str(n)); gt.append(ozm[str(n)])
        us.append(float(f[0])); aoas.append(float(f[1]))
        pf.append([float(x) for x in f[2:]])
        geom.append(np.asarray(zv["G"][i], dtype=float))
    return names, np.array(gt), np.array(pf), np.array(aoas), np.array(us), geom


def anchor(model, ns):
    """511 个真实 5 位族样本：模型 CL vs 官方 CL 的排序一致性 + 偏差口径对比。

    两种口径都算，用来分辨"模型偏差"和"把风速固定成 62 m/s 带来的偏差"：
      A) 每个样本用**自己的** (U∞, AoA) —— 公平锚定；
      B) 全部固定 U∞=62 —— 与 GA 设计点同口径，但混入了风速差异。
    """
    import csv
    from scipy.stats import spearmanr
    zv = np.load(f"{POD}/geomvec_1000.npz", allow_pickle=True)
    oz = np.load(f"{POD}/official_coefs_1000.npz", allow_pickle=True)
    ozm = {str(n): float(c) for n, c in zip(oz["names"], oz["CL"])}
    names, gt, pf, aoas, us, geom = _real_5digit(zv, ozm)

    def report(tag_, cl):
        rel = (cl - gt) / gt
        rho = spearmanr(cl, gt).statistic
        r2 = 1.0 - np.sum((cl - gt) ** 2) / np.sum((gt - gt.mean()) ** 2)
        print("  [%s] Spearman=%.4f  R2=%.4f  平均偏差=%+.2f%%  中位=%+.2f%%  "
              "MAE=%.4f  最大|偏差|=%.1f%%"
              % (tag_, rho, r2, 100 * rel.mean(), 100 * np.median(rel),
                 np.abs(cl - gt).mean(), 100 * np.abs(rel).max()))
        return {"rho": float(rho), "r2": float(r2), "bias_mean": float(rel.mean()),
                "bias_med": float(np.median(rel)), "mae": float(np.abs(cl - gt).mean())}

    print("=== 真实锚定（n=%d 个 5 位族样本）===" % len(gt))
    res_u = eval_batch(model, ns, pf, aoas, geom, u_list=us)
    res_f = eval_batch(model, ns, pf, aoas, geom)
    sA = report("A 各自风速", res_u["CL"])
    sB = report("B 固定62m/s", res_f["CL"])

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/anchor_real.npz", names=np.array(names),
                        cl_model=res_u["CL"], cl_model_u62=res_f["CL"],
                        cl_official=gt, aoas=aoas, u_inf=us, params=pf)
    # 分位数表：模型 vs 官方（口径 A），落盘成小表，供报告直接引用
    qs = np.quantile(gt, np.linspace(0, 1, 11))
    with open(f"{OUT}/anchor_deciles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "cl_lo", "cl_hi", "n", "cl_official_mean",
                    "cl_model_mean", "bias_mean_pct"])
        for k in range(10):
            m = (gt >= qs[k]) & (gt <= qs[k + 1] if k == 9 else gt < qs[k + 1])
            if not m.any():
                continue
            bl = 100 * ((res_u["CL"][m] - gt[m]) / gt[m]).mean()
            w.writerow([k + 1, round(float(qs[k]), 4), round(float(qs[k + 1]), 4),
                        int(m.sum()), round(float(gt[m].mean()), 4),
                        round(float(res_u["CL"][m].mean()), 4), round(float(bl), 3)])
    with open(f"{OUT}/anchor_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["convention", "n", "spearman", "r2", "bias_mean_pct",
                    "bias_median_pct", "mae"])
        for tag_, s in (("A_per_sample_Uinf_AoA", sA), ("B_fixed_U62", sB)):
            w.writerow([tag_, len(gt), round(s["rho"], 4), round(s["r2"], 4),
                        round(100 * s["bias_mean"], 3), round(100 * s["bias_med"], 3),
                        round(s["mae"], 4)])
    print("  -> anchor_real.npz / anchor_deciles.csv / anchor_summary.csv")


def neighbour_check(model, ns, k=12, front_csv=None, aoa_band=(4.0, 10.0)):
    """S5 邻域真值核对：GA 最优解在**参数空间**的 k 个最近真实样本上，模型误差有多大。

    这是回答"贴着搜索边界的解，绝对升力到底能不能信"的唯一可行办法：
    拿不到新形状的 CFD，但可以给出该参数邻域内模型 vs 官方 CL 的局部偏差。
    邻域只在**工作带内**的真实样本里找（AoA∈aoa_band），否则容易挑到低攻角、CL≈0 的样本，
    对比就没有意义；每个样本用**它自己的** (U∞, AoA) 评估模型。
    """
    import csv
    zv = np.load(f"{POD}/geomvec_1000.npz", allow_pickle=True)
    oz = np.load(f"{POD}/official_coefs_1000.npz", allow_pickle=True)
    ozm = {str(n): float(c) for n, c in zip(oz["names"], oz["CL"])}
    names, gt, pf, aoas, us, geom = _real_5digit(zv, ozm)
    keep = np.where((aoas >= aoa_band[0]) & (aoas <= aoa_band[1]))[0]
    print("  可选邻域样本（AoA∈[%.0f,%.0f]）= %d / %d"
          % (aoa_band[0], aoa_band[1], len(keep), len(aoas)))

    front_csv = front_csv or f"{OUT}/ga_front_s42.csv"
    rows = [r for r in csv.DictReader(open(front_csv, encoding="utf-8"))
            if r["feasible"] == "True"]
    if not rows:
        print("  !! 前沿 CSV 里没有可行解，跳过邻域核对：", front_csv)
        return
    best = max(rows, key=lambda r: float(r["CL8"]))
    b = np.array([float(best[x]) for x in ("L", "P", "Q", "T")])
    LO, HI = BOUNDS[:, 0], BOUNDS[:, 1]
    dist = np.linalg.norm((pf[keep] - LO) / (HI - LO) - (b - LO) / (HI - LO), axis=1)
    pos = np.argsort(dist)[:k]
    idx = keep[pos]
    dselp = dist[pos]

    cl = eval_batch(model, ns, [list(pf[i]) for i in idx],
                    [float(aoas[i]) for i in idx],
                    [geom[i] for i in idx],
                    u_list=[float(us[i]) for i in idx])["CL"]
    rel = (cl - gt[idx]) / gt[idx]
    print("=== 邻域真值核对（GA 最优解 L=%.3f P=%.3f Q=%d T=%.3f 的 %d 个最近真实样本）==="
          % (b[0], b[1], b[2], b[3], k))
    print("  最近样本归一化距离 = %.3f ~ %.3f" % (dselp.min(), dselp.max()))
    print("  局部偏差：平均 %+.2f%%  中位 %+.2f%%  最大|偏差| %.1f%%"
          % (100 * rel.mean(), 100 * np.median(rel), 100 * np.abs(rel).max()))
    with open(f"{OUT}/anchor_neighbour.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "u_inf", "aoa", "L", "P", "Q", "T",
                    "norm_dist", "cl_official", "cl_model", "rel_pct"])
        for r_, i in enumerate(idx):
            w.writerow([r_ + 1, names[i], us[i], aoas[i]] + [round(x, 4) for x in pf[i]]
                       + [round(float(dselp[r_]), 4), round(float(gt[i]), 4),
                          round(float(cl[r_]), 4), round(100 * float(rel[r_]), 3)])
    np.savez_compressed(f"{OUT}/anchor_neighbour.npz", idx=idx, dist=dselp,
                        cl_model=cl, cl_official=gt[idx], params=pf[idx],
                        ga_best_params=b, ga_best_model_CL=float(best["CL8"]))
    print("  GA 最优解本身：模型 CL8=%.4f（真值未验证，仅供与邻域真值同图对照）"
          % float(best["CL8"]))
    print("  -> anchor_neighbour.csv / anchor_neighbour.npz")


# ---------------------------------------------------------------- 评估与校准
def eval_pop(model, ns, P, sep_max):
    """一次评估整个种群（三个攻角一起喂）。返回 objectives/constraints/明细。"""
    n = len(P)
    aoas = np.array([AOA_BAND[0], AOA_DESIGN, AOA_BAND[1]])
    pr = np.repeat(P, 3, axis=0)
    aa = np.tile(aoas, n)
    geom = [gen_geometry(p) for p in pr]
    r = eval_batch(model, ns, pr, aa, geom)
    CL = r["CL"].reshape(n, 3)
    m = r["m_local"].reshape(n, 3)
    s = r["sep"].reshape(n, 3)
    cl4, cl8, cl10 = CL[:, 0], CL[:, 1], CL[:, 2]
    delta = np.maximum(cl8 - cl4, cl8 - cl10)
    F = np.column_stack([-cl8, delta])
    m_max = m.max(axis=1)
    sep_worst = s.max(axis=1)
    G = np.column_stack([m_max - M_LOCAL_MAX, sep_worst - sep_max])
    return {"F": F, "G": G, "CL4": cl4, "CL8": cl8, "CL10": cl10,
            "delta": delta, "m_max": m_max, "sep": sep_worst,
            "cp4": r["cp"][0::3], "cp8": r["cp"][1::3], "cp10": r["cp"][2::3]}


def calibrate_thresholds(model, ns):
    """用 511 个真实 5 位族样本定 `sep_max`（分离代理阈值）与 m_local 参考分布。"""
    zv = np.load(f"{POD}/geomvec_1000.npz", allow_pickle=True)
    names = [str(n) for n in zv["names"].tolist()]
    keep = [i for i, n in enumerate(names)
            if len(n.replace("airFoil2D_SST_", "").split("_")) == 6]
    pf = [[float(x) for x in names[i].replace("airFoil2D_SST_", "").split("_")[2:]]
          for i in keep]
    aoas = [float(names[i].replace("airFoil2D_SST_", "").split("_")[1]) for i in keep]
    geom = [np.asarray(zv["G"][i], float) for i in keep]
    r = eval_batch(model, ns, pf, aoas, geom)
    sep = r["sep"]
    mloc = r["m_local"]
    thr = float(np.percentile(sep, 90))
    info = {"sep_p50": float(np.percentile(sep, 50)),
            "sep_p90": thr, "sep_p99": float(np.percentile(sep, 99)),
            "sep_max_observed": float(sep.max()),
            "mloc_p50": float(np.percentile(mloc, 50)),
            "mloc_p99": float(np.percentile(mloc, 99)),
            "n_real": int(len(sep)), "sep_max_used": thr}
    with open(f"{OUT}/calibration.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("=== 校准（%d 个真实 5 位族样本）===" % len(sep))
    print("  分离代理 sep: p50=%.4f p90=%.4f p99=%.4f max=%.4f -> 阈值取 p90=%.4f"
          % (info["sep_p50"], info["sep_p90"], info["sep_p99"],
             info["sep_max_observed"], thr))
    print("  m_local: p50=%.3f p99=%.3f" % (info["mloc_p50"], info["mloc_p99"]))
    return thr


# ---------------------------------------------------------------- NSGA-II 核心
def nds(F, G):
    """约束支配 + 非支配排序。返回 (rank, violation)。"""
    n = len(F)
    viol = np.clip(G, 0, None).sum(axis=1)
    feas = viol <= 0
    rank = np.full(n, -1)
    order = sorted(range(n), key=lambda i: (not feas[i], viol[i]))
    fronts, cur = [], []
    remaining = set(order)
    while remaining:
        cur = []
        for i in sorted(remaining):
            dom = False
            for j in remaining:
                if i == j:
                    continue
                if dominates(F[j], G[j], viol[j], feas[j], F[i], G[i], viol[i], feas[i]):
                    dom = True
                    break
            if not dom:
                cur.append(i)
        for i in cur:
            remaining.discard(i)
            rank[i] = len(fronts)
        fronts.append(cur)
    return rank, viol, feas


def dominates(fa, ga, va, feas_a, fb, gb, vb, feas_b):
    """a 是否支配 b（最小化 F；约束：可行优先，都不可行时违反量小的优先）。"""
    if feas_a and not feas_b:
        return True
    if not feas_a and feas_b:
        return False
    if not feas_a and not feas_b:
        if va < vb - 1e-12:
            return True
        if va > vb + 1e-12:
            return False
    return bool(np.all(fa <= fb)) and bool(np.any(fa < fb))


def crowding(F, front):
    if len(front) <= 2:
        return {i: np.inf for i in front}
    d = {i: 0.0 for i in front}
    for k in range(F.shape[1]):
        s = sorted(front, key=lambda i: F[i, k])
        d[s[0]] = d[s[-1]] = np.inf
        rng = F[s[-1], k] - F[s[0], k] + 1e-12
        for a, b in zip(s[1:-1], s[2:]):
            d[a] += (F[b, k] - F[a, k]) / rng
    return d


def hv2d(F, ref):
    """二维超体积（最小化）：非支配点集相对参考点 ref 覆盖的面积。

    修正记录（本地独立复算发现）：旧版按"相邻点的条带"累加、条带高度用的是**上一个点**的
    f2 之差，导致前沿越密数值越小（实测 0.030 → 0.0035，看着像"越优化越差"）。
    正确做法：先剔掉被支配点，再按 f1 升序，第 i 段宽度 = f1_{i+1} - f1_i（末段到 ref1），
    高度 = ref2 - f2_i。用落盘快照复算后：2.037 → 2.421（单调上升后饱和），符合预期。
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


def sbx(p1, p2, lo, hi, rng, eta=15.0, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    for k in range(len(p1)):
        if rng.random() > 0.5:
            continue
        if abs(p1[k] - p2[k]) < 1e-12:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[k] = 0.5 * ((1 + beta) * p1[k] + (1 - beta) * p2[k])
        c2[k] = 0.5 * ((1 - beta) * p1[k] + (1 + beta) * p2[k])
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def poly_mut(p, lo, hi, rng, eta=20.0):
    q = p.copy()
    for k in range(len(p)):
        if rng.random() > 1.0 / len(p):
            continue
        u = rng.random()
        d = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        q[k] = p[k] + d * (hi[k] - lo[k])
    return np.clip(q, lo, hi)


def nsga2(model, ns, pop=60, gens=60, seed=42):
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    thr_file = f"{OUT}/calibration.json"
    sep_max = (json.load(open(thr_file, encoding="utf-8"))["sep_max_used"]
               if os.path.exists(thr_file) else 0.5)
    print("使用 sep_max=%.4f" % sep_max)
    X = rng.uniform(lo, hi, size=(pop, 4))
    # 用 511 个真实样本里的 10 个做"热启动"，保证初始种群含可行区
    zv = np.load(f"{POD}/geomvec_1000.npz", allow_pickle=True)
    real = []
    for n in zv["names"].tolist():
        f = str(n).replace("airFoil2D_SST_", "").split("_")
        if len(f) == 6:
            real.append([float(x) for x in f[2:]])
    if len(real) >= 10:
        X[:10] = np.array(real)[rng.choice(len(real), 10, replace=False)]
    hist = []
    snaps = []
    REF = np.array([0.5, 1.5])       # 超体积参考点（比所有解都差）
    for g in range(gens):
        ev = eval_pop(model, ns, X, sep_max)
        rank, viol, feas = nds(ev["F"], ev["G"])
        fronts = []
        for r in range(rank.max() + 1):
            fronts.append(list(np.where(rank == r)[0]))
        cd = {}
        for fr in fronts:
            cd.update(crowding(ev["F"], fr))
        hv = hv2d(ev["F"][feas] if feas.any() else ev["F"], REF)
        if g % max(1, gens // 10) == 0 or g == gens - 1:
            snaps.append({"gen": g, "params": X.copy(),
                          "F": ev["F"].copy(), "G": ev["G"].copy()})
        hist.append({"gen": g, "hypervolume": hv, "n_feasible": int(feas.sum()),
                     "best_CL8": float(ev["CL8"].max()),
                     "best_feasible_CL8": float(ev["CL8"][feas].max()) if feas.any() else np.nan,
                     "min_delta_feasible": float(ev["delta"][feas].min()) if feas.any() else np.nan,
                     "front0": len(fronts[0]) if fronts else 0})
        if g == gens - 1:
            break
        # 选择
        def better(i, j):
            if rank[i] != rank[j]:
                return rank[i] < rank[j]
            return cd[i] > cd[j]
        pool = list(range(pop))
        rng.shuffle(pool)
        children = []
        while len(children) < pop:
            i, j = rng.choice(pool, 2, replace=False)
            a = i if better(i, j) else j
            i2, j2 = rng.choice(pool, 2, replace=False)
            b = i2 if better(i2, j2) else j2
            c1, c2 = sbx(X[a], X[b], lo, hi, rng)
            children += [poly_mut(c1, lo, hi, rng), poly_mut(c2, lo, hi, rng)]
        children = np.array(children[:pop])
        M = np.vstack([X, children])
        ev2 = eval_pop(model, ns, M, sep_max)
        rank2, viol2, feas2 = nds(ev2["F"], ev2["G"])
        fronts2 = [list(np.where(rank2 == r)[0]) for r in range(rank2.max() + 1)]
        cd2 = {}
        for fr in fronts2:
            cd2.update(crowding(ev2["F"], fr))
        newX, cnt = [], 0
        for fr in fronts2:
            fr = sorted(fr, key=lambda i: -cd2[i])
            for i in fr:
                if cnt >= pop:
                    break
                newX.append(M[i]); cnt += 1
            if cnt >= pop:
                break
        X = np.array(newX)
    ev = eval_pop(model, ns, X, sep_max)
    rank, viol, feas = nds(ev["F"], ev["G"])
    return X, ev, rank, viol, feas, np.array(hist), sep_max, snaps


def save_ga(X, ev, rank, viol, feas, hist, sep_max, seed, pop, gens, snaps=None):
    import csv
    front = np.where((rank == 0))[0]
    rows = []
    for i in range(len(X)):
        L, P, Q, T = X[i]
        rows.append({"idx": i, "rank": int(rank[i]), "feasible": bool(feas[i]),
                     "violation": float(viol[i]), "L": L, "P": P, "Q": round(float(Q)),
                     "T": T, "camber": 0.15 * L, "x_f": P / 20, "t_c": T / 100,
                     "CL4": ev["CL4"][i], "CL8": ev["CL8"][i], "CL10": ev["CL10"][i],
                     "delta": ev["delta"][i], "m_max": ev["m_max"][i], "sep": ev["sep"][i]})
    rows.sort(key=lambda r: (-r["feasible"], -r["CL8"]))
    with open(f"{OUT}/ga_all{TAG}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(f"{OUT}/ga_front{TAG}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows([r for r in rows if r["rank"] == 0])
    with open(f"{OUT}/ga_history{TAG}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist.tolist())
    if snaps:
        d = {"gens": np.array([s0["gen"] for s0 in snaps])}
        for k, s0 in enumerate(snaps):
            d["F%d" % k] = s0["F"]
            d["G%d" % k] = s0["G"]
            d["params%d" % k] = s0["params"]
        np.savez_compressed(f"{OUT}/ga_snaps{TAG}.npz", **d)
        print("  -> ga_snaps%s.npz（%d 个快照，供画前沿演化）" % (TAG, len(snaps)))
    fi = front
    np.savez_compressed(f"{OUT}/ga_front{TAG}.npz",
                        params=X[fi], geometry=np.array([gen_geometry(p) for p in X[fi]]),
                        cp4=ev["cp4"][fi], cp8=ev["cp8"][fi], cp10=ev["cp10"][fi],
                        CL8=ev["CL8"][fi], delta=ev["delta"][fi],
                        m_max=ev["m_max"][fi], sep=ev["sep"][fi],
                        seed=seed, pop=pop, gens=gens, sep_max=sep_max)
    best = max([r for r in rows if r["feasible"]], key=lambda r: r["CL8"], default=None)
    print("=== 帕累托前沿 %d 个解；可行解 %d 个 ===" % (len(fi), int(feas.sum())))
    if best:
        print("  最优可行解: CL8=%.4f (CL4=%.3f CL10=%.3f Δ=%.4f) m_max=%.3f sep=%.4f"
              % (best["CL8"], best["CL4"], best["CL10"], best["delta"],
                 best["m_max"], best["sep"]))
        print("  参数: L=%.3f P=%.3f Q=%d T=%.3f -> camber=%.4f x_f=%.3f t_c=%.4f"
              % (best["L"], best["P"], best["Q"], best["T"],
                 best["camber"], best["x_f"], best["t_c"]))
    print("  参照：数据库内 5 位族最优 CL=1.1464（可达上界 1.4225）")
    print("  -> ga_all.csv / ga_front.csv / ga_history.csv / ga_front.npz")


def grid_scan(model, ns, nL=12, nP=8, nT=10, sep_max=None):
    """补充实验 S4：参数网格穷举（参考上界 + 边界案例地图）。"""
    import csv
    if sep_max is None:
        f = f"{OUT}/calibration.json"
        sep_max = json.load(open(f, encoding="utf-8"))["sep_max_used"] if os.path.exists(f) else 0.5
    Ls = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], nL)
    Ps = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], nP)
    Ts = np.linspace(BOUNDS[3, 0], BOUNDS[3, 1], nT)
    combos = [[L, P, Q, T] for L in Ls for P in Ps for Q in (0, 1) for T in Ts]
    print("网格点数 = %d（L%d × P%d × Q2 × T%d）" % (len(combos), nL, nP, nT))
    rows = []
    B = 200
    for s in range(0, len(combos), B):
        chunk = combos[s:s + B]
        ev = eval_pop(model, ns, np.array(chunk), sep_max)
        for k, pr in enumerate(chunk):
            rows.append({"L": pr[0], "P": pr[1], "Q": pr[2], "T": pr[3],
                         "camber": 0.15 * pr[0], "x_f": pr[1] / 20, "t_c": pr[3] / 100,
                         "CL4": ev["CL4"][k], "CL8": ev["CL8"][k], "CL10": ev["CL10"][k],
                         "delta": ev["delta"][k], "m_max": ev["m_max"][k], "sep": ev["sep"][k],
                         "feasible": bool(ev["G"][k][0] <= 0 and ev["G"][k][1] <= 0)})
        print("  %d/%d" % (min(s + B, len(combos)), len(combos)), flush=True)
    rows.sort(key=lambda r: (-r["feasible"], -r["CL8"]))
    with open(f"{OUT}/grid_all{TAG}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    top = [r for r in rows if r["feasible"]][:10]
    print("=== 穷举参考上界（可行解里 CL8 最高 10 个）===")
    for r in top[:5]:
        print("  CL8=%.4f Δ=%.4f m=%.3f sep=%.4f | L=%.2f P=%.2f Q=%d T=%.2f"
              % (r["CL8"], r["delta"], r["m_max"], r["sep"],
                 r["L"], r["P"], r["Q"], r["T"]))
    print("  -> grid_all.csv（%d 行，含可行标志，可作边界案例地图）" % len(rows))


def invert_one(model, ns, target_name, target_aoa, target_cp, pop=60, gens=40, seed=0,
               true_params=None):
    """S6 干净验证：给定真实样本的 Cp 曲线（藏起形状），反推 (L,P,Q,T)。

    true_params 给定时会额外算出**模型在真参数处的 Cp 距离**——这是本次反演的"地板"：
    模型自己都偏了那么多，反演再好也不可能更接近。有了它才能把
    "反演搜索没做好" 和 "模型本身在这个形状上就不准" 分开。

    注意：搜索期 Q 当连续量（好爬），**出结果前吸附到 0/1**（几何生成用的是 round(Q)）；
    所以报告的参数一定是合法的五位族参数。
    """
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    X = rng.uniform(lo, hi, size=(pop, 4))

    def fit(P):
        n = len(P)
        geom = [gen_geometry(p) for p in P]
        r = eval_batch(model, ns, [list(p) for p in P], [target_aoa] * n, geom)
        cp = r["cp"]
        d = np.linalg.norm(cp - np.asarray(target_cp)[None, :], axis=1) / np.linalg.norm(target_cp)
        return d, r["CL"], cp

    floor, cp_true = None, None
    if true_params is not None:
        dt, clt, cpt = fit(np.array([true_params], float))
        floor, cp_true = float(dt[0]), cpt[0]
        print("  模型在真参数处的 Cp 距离（反演地板）=%.4f" % floor)

    hist = []
    best = None
    for g in range(gens):
        d, cl, _ = fit(X)
        hist.append(float(d.min()))
        idx = np.argsort(d)
        if best is None or d[idx[0]] < best[0]:
            best = (float(d[idx[0]]), X[idx[0]].copy(), float(cl[idx[0]]))
        elite = X[idx[: max(4, pop // 5)]]
        kids = []
        while len(kids) < pop - len(elite):
            a, b = elite[rng.integers(len(elite))], elite[rng.integers(len(elite))]
            c1, c2 = sbx(a, b, lo, hi, rng, pc=0.9)
            kids += [poly_mut(c1, lo, hi, rng), poly_mut(c2, lo, hi, rng)]
        X = np.vstack([elite, np.array(kids[: pop - len(elite)])])
    # Q 是**二值**设计变量（几何生成时本来就取 round(Q)）；搜索期让它连续只是为了好爬，
    # 出结果前必须**吸附到 0/1**，否则报告里的参数不合法（曾出现 Q=0.348 这种写法）。
    X = np.array(X, float)
    X[:, 2] = np.round(X[:, 2])
    d, cl, cp_all = fit(X)
    i = int(np.argmin(d))
    best_cp = cp_all[i]                      # 必须是曲线本身，不是标量距离
    np.savez_compressed(f"{OUT}/invert_{target_name[:32]}.npz",
                        target_cp=np.asarray(target_cp), best_cp=best_cp,
                        params=X, cp_relL2=d, CL=cl, cp_all=cp_all,
                        dist_history=np.array(hist),
                        best_params=X[i], best_cp_relL2=float(d[i]),
                        true_params=(np.asarray(true_params, float)
                                     if true_params is not None else np.zeros(0)),
                        cp_true_relL2=(floor if floor is not None else np.nan),
                        cp_at_true=(cp_true if cp_true is not None else np.zeros(0)))
    return {"target": target_name, "target_aoa": target_aoa,
            "recovered_params": X[i].tolist(), "cp_relL2": float(d[i]),
            "CL_recovered": float(cl[i]),
            "true_params": (list(true_params) if true_params is not None else None),
            "cp_true_relL2": floor,
            "dist_history_best": best[0]}


def eval_cases(model, ns, pts, sep_max, tag=""):
    """S8 用：把指定参数点（成功/边界/失败三类候选）在三攻角上评估并全部落盘。

    pts: [[L,P,Q,T], ...]
    落盘 cases_eval{tag}.npz / .csv：参数、三个攻角的 Cp、CL、Δ、m_max、sep、可行标志、几何。
    """
    import csv
    pts = np.asarray(pts, float)
    n = len(pts)
    aoas = np.array([AOA_BAND[0], AOA_DESIGN, AOA_BAND[1]])
    pr = np.repeat(pts, 3, axis=0)
    aa = np.tile(aoas, n)
    geom = [gen_geometry(p) for p in pr]
    r = eval_batch(model, ns, pr, aa, geom)
    CL = r["CL"].reshape(n, 3)
    m = r["m_local"].reshape(n, 3)
    s = r["sep"].reshape(n, 3)
    cp = r["cp"].reshape(n, 3, -1)
    cl4, cl8, cl10 = CL[:, 0], CL[:, 1], CL[:, 2]
    delta = np.maximum(cl8 - cl4, cl8 - cl10)
    m_max, sep = m.max(axis=1), s.max(axis=1)
    feas = (m_max <= M_LOCAL_MAX) & (sep <= sep_max)
    np.savez_compressed(
        f"{OUT}/cases_eval{tag}.npz", params=pts, cp4=cp[:, 0], cp8=cp[:, 1], cp10=cp[:, 2],
        CL4=cl4, CL8=cl8, CL10=cl10, delta=delta, m_max=m_max, sep=sep,
        feasible=feas, geometry=np.array([gen_geometry(p) for p in pts]), sep_max=sep_max)
    with open(f"{OUT}/cases_eval{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "L", "P", "Q", "T", "CL4", "CL8", "CL10", "delta",
                    "m_max", "sep", "feasible", "binding"])
        for i in range(n):
            bind = []
            if m_max[i] > M_LOCAL_MAX - 0.02:
                bind.append("m_local")
            if sep[i] > sep_max * 0.99:
                bind.append("sep")
            w.writerow([i + 1] + [round(float(x), 4) for x in pts[i]]
                       + [round(float(x), 4) for x in (cl4[i], cl8[i], cl10[i], delta[i])]
                       + [round(float(m_max[i]), 4), round(float(sep[i]), 6),
                          bool(feas[i]), "|".join(bind) or "-"])
    print("=== 案例点评估（%d 个）===" % n)
    for i in range(n):
        print("  #%d L=%.3f P=%.3f Q=%d T=%.3f | CL8=%.4f Δ=%.4f m=%.3f sep=%.5f %s"
              % (i + 1, pts[i][0], pts[i][1], pts[i][2], pts[i][3], cl8[i], delta[i],
                 m_max[i], sep[i], "可行" if feas[i] else "不可行"))
    print("  -> cases_eval%s.npz / cases_eval%s.csv" % (tag, tag))
    return {"params": pts, "CL8": cl8, "delta": delta, "feasible": feas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--neighbour", action="store_true",
                    help="S5 邻域真值核对（GA 最优解附近的真实样本）")
    ap.add_argument("--front-csv", type=str, default=None,
                    help="--neighbour 用的前沿 CSV，默认 results/ga_front_s42.csv")
    ap.add_argument("--cases", type=str, default=None,
                    help="S8：评估案例点，格式 \"L,P,Q,T;L,P,Q,T;...\"")
    ap.add_argument("--cases-tag", type=str, default="",
                    help="案例产物的 tag 后缀（如 _s8）")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--invert", type=str, default=None,
                    help="干净验证：给定真实样本名（藏形状，用它的真实 Cp 反推参数）")
    ap.add_argument("--invert-list", type=str, default=None,
                    help="多个目标样本名，逗号分隔（S6 批量反演）")
    ap.add_argument("--invert-gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    TAG = a.tag
    os.makedirs(OUT, exist_ok=True)
    model, ns = load_model()
    if a.selftest:
        selftest(model, ns)
    if a.anchor:
        anchor(model, ns)
    if a.neighbour:
        neighbour_check(model, ns, front_csv=a.front_csv)
    if a.cases:
        pts = [[float(x) for x in g.split(",")] for g in a.cases.split(";") if g.strip()]
        f = f"{OUT}/calibration.json"
        sep_max = (json.load(open(f, encoding="utf-8"))["sep_max_used"]
                   if os.path.exists(f) else 0.5)
        eval_cases(model, ns, pts, sep_max, tag=a.cases_tag)
    if a.calibrate:
        calibrate_thresholds(model, ns)
    if a.run:
        X, ev, rank, viol, feas, hist, sep_max, snaps = nsga2(
            model, ns, pop=a.pop, gens=a.gen, seed=a.seed)
        save_ga(X, ev, rank, viol, feas, hist, sep_max, a.seed, a.pop, a.gen, snaps)
    if a.grid:
        grid_scan(model, ns)
    if a.invert or a.invert_list:
        wz = np.load(f"{POD}/wallcp_1000.npz", allow_pickle=True)
        wm = {str(n): i for i, n in enumerate(wz["names"].tolist())}
        todo = ([x.strip() for x in a.invert_list.split(",") if x.strip()]
                if a.invert_list else [a.invert])
        for nm in todo:
            tgt = np.asarray(wz["CP"][wm[nm]], dtype=float)
            aoa = float(nm.replace("airFoil2D_SST_", "").split("_")[1])
            fld = nm.replace("airFoil2D_SST_", "").split("_")
            tp = ([float(x) for x in fld[2:]] if len(fld) == 6 else None)
            out = invert_one(model, ns, nm, aoa, tgt, pop=a.pop,
                             gens=a.invert_gens, seed=a.seed, true_params=tp)
            with open(f"{OUT}/invert_{nm[:32]}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            print("=== 反演（目标=%s, AoA=%.3f）===" % (nm[:40], aoa))
            print("  Cp 相对误差=%.4f（模型地板 %.4f）  反演 CL=%.4f"
                  % (out["cp_relL2"], out["cp_true_relL2"] or float("nan"),
                     out["CL_recovered"]))
            if out["true_params"]:
                print("  真参数:", [round(x, 3) for x in out["true_params"]])
            print("  反演参数:", [round(x, 3) for x in out["recovered_params"]], flush=True)
    print("done", time.strftime("%H:%M:%S"))

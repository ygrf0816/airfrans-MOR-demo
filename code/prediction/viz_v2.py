"""可视化（重写）：每个任务随机抽 6 个测试样本，只读这 6 个文件，用已训好权重推理。

输出：
  1) Cp 曲线图   {task}_cp.png
     3 列子图，横轴统一弦向 x/c（上下表面分线），同一行 3 列分别是：
        真值(CFD) | A 链路 | B 链路(POD 升维)
     每格标题给该样本 AIM 的 Cp 平均绝对误差，方便横向比较。
     ⚠ 真值/POD 基的曲线在 x=0 附近差异来自前缘采样密度（CFD 表面点极密、
       64 线性站位会在前缘把多个点并成一格），非模型误差，标题已注明。

  2) 流场云图  {task}_field_{u,v,p,nu_t}.png
     每种场量单独一张图（4 张），6 行（6 个样本）× 3 列：
        真值 | A 链路 | B 链路(POD 升维)
     误差不单独画，改为在子图标题里给平均相对误差。

B 链路只输出壁面 Cp（256 点，前缘锚定、弧长序），所以流场云图里 "B 链路"
一列对体积场无法逐点给出 —— 这一列对 Cp 场有效；对 u/v/p/nu_t 体积场，
B 链路按设计不预测体积场，该列留空并标注 "B: N/A (只建模壁面 Cp)"。
"""
import os, sys, json, random, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from runkit import RunDir  # noqa: E402
from train_A_gnn import OfficialGraphSAGE, build_radius_graph, SUBSAMP  # noqa: E402
from train_B_transformer import BBackbone, load_pod_basis, cp_from_alpha, POD_R  # noqa: E402
from train_B_field import FieldMLP  # noqa: E402
from geometry_input import load_all_geometric, input_vector_5scalar  # noqa: E402
from single_sample import load_sample, TASK_ROOT  # noqa: E402
import b_cp_to_xchord as bx  # noqa: E402

FIGDIR = os.path.join(_HERE, "04_results", "prediction", "figs")
os.makedirs(FIGDIR, exist_ok=True)
WALLCP_NPZ = "/home/zch/airfrans/pod/wallcp_1000.npz"
VOL_FIELDS_NPZ = "/home/zch/airfrans/pod/vol_fields_1000.npz"
BFIELD_BASIS_NPZ = "/home/zch/airfrans/pod/bfield_basis.npz"
TASKS = ["full", "scarce", "reynolds", "aoa"]
N_SAMP = 6
SEED = 20261004
FIELD_NAMES = ["u", "v", "p", "nu_t"]
POD_R_FIELD = 32

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_names(task, n=N_SAMP):
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    test_key = "full_test" if task == "scarce" else task + "_test"
    rng = random.Random(SEED + len(task) * 131)
    return rng.sample(man[test_key], n)


def a_coef_norm(task):
    meta = json.load(open(os.path.join(RunDir(task, "A").dir, "eval_metrics.json"),
                          encoding="utf-8"))
    c = meta["coef_norm"]
    return (np.asarray(c["mean_in"], np.float32), np.asarray(c["std_in"], np.float32) + 1e-8,
            np.asarray(c["mean_out"], np.float32), np.asarray(c["std_out"], np.float32) + 1e-8)


def b_stats(task):
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    tn = man[task + "_train"]
    n_val = max(1, int(0.1 * len(tn)))
    tr_eff = tn[:len(tn) - n_val]
    scal = np.stack([input_vector_5scalar(n) for n in tr_eff])
    smean, sstd = scal.mean(0, keepdims=True), scal.std(0, keepdims=True)
    sstd[sstd == 0] = 1
    geo = load_all_geometric(tr_eff, TASK_ROOT)
    gmean, gstd = geo.mean(0, keepdims=True), geo.std(0, keepdims=True)
    gstd[gstd == 0] = 1
    return smean, sstd, gmean, gstd


# ---- B 全场版（train_B_field.FieldMLP + 均匀网格 POD 基）----
# 与 train_B_field.py 训练时完全一致：用 task 训练样本(前 90%)逐场做 SVD，
# 网格真值来自 vol_fields_1000.npz，用 A 链路的官方 coef_norm 标准化。
_FIELD_FIELDS = ["u", "v", "p", "nu_t"]
_BFIELD_CACHE = {}

def _bf_cache(task, device="cpu"):
    """B 全场版：直接从 pod/bfield_basis.npz 读 POD 基与网格几何，不再重算 SVD。
    与 b_field_basis 脚本落盘口径一致：Phi_<task> (4,31066,32)，mu_<task> (4,1,31066)，
    几何向量仍走 geomvec_1000.npz 缓存（见 load_all_geometric）。"""
    if task not in _BFIELD_CACHE:
        bd = np.load(BFIELD_BASIS_NPZ)
        phi_key = None
        for cand in bd.files:
            if cand == "Phi_" + task or cand.startswith("Phi_" + task + "."):
                phi_key = cand
                break
        assert phi_key is not None, f"Phi key for task '{task}' not in bfield_basis.npz"
        mu_key = phi_key.replace("Phi", "mu")
        Phi = [np.asarray(bd[phi_key][k], dtype=np.float64) for k in range(len(bd[phi_key]))]
        Mu = [np.asarray(bd[mu_key][k], dtype=np.float64) for k in range(len(bd[mu_key]))]
        assert all(p.shape[0] == 31066 for p in Phi), "bfield basis ncell mismatch"
        cp = np.load(VOL_FIELDS_NPZ, allow_pickle=True)
        xs = np.linspace(float(cp["x0"]), float(cp["x1"]), int(cp["nx"]))
        ys = np.linspace(float(cp["y0"]), float(cp["y1"]), int(cp["ny"]))
        GX, GY = np.meshgrid(xs, ys, indexing="xy")
        gxy = np.column_stack([GX.ravel(), GY.ravel()])[cp["mask"]]
        rd = RunDir(task, "B_field")
        m = FieldMLP(133, 4 * POD_R_FIELD).to(device)
        st = torch.load(os.path.join(rd.dir, "best_model.pt"), map_location=device)
        m.load_state_dict(st["model_state"])
        m.eval()
        smean, sstd, gmean, gstd = b_stats(task)
        _BFIELD_CACHE[task] = dict(model=m, Phi=Phi, Mu=Mu, gxy=gxy,
                                   smean=smean, sstd=sstd, gmean=gmean, gstd=gstd)
    return _BFIELD_CACHE[task]

def b_field_predict(names, task, device="cpu"):
    """B 全场版：给定样本名列表 -> 每样本在均匀网格上的 (ncell,4) 重构（物理量）。
    输入向量构造与 train_B_field.build_inputs 严格一致：
    5 标量与 128 几何各自用 tr 的 mean/std 归一化后拼接。"""
    c = _bf_cache(task, device)
    m, Phi, Mu = c["model"], c["Phi"], c["Mu"]
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    tr_all = man[task + "_train"]
    tr_eff = tr_all[:int(0.9 * len(tr_all))]
    # 与 build_inputs 一致：标量/几何分开归一化
    s_tr = np.stack([input_vector_5scalar(n) for n in tr_eff]).astype(np.float64)
    s_te = np.stack([input_vector_5scalar(n) for n in names]).astype(np.float64)
    m_, s_ = s_tr.mean(0, keepdims=True), s_tr.std(0, keepdims=True); s_[s_ == 0] = 1.0
    g_tr = load_all_geometric(tr_eff, TASK_ROOT, 64)
    g_te = load_all_geometric(names, TASK_ROOT, 64)
    gm, gs = g_tr.mean(0, keepdims=True), g_tr.std(0, keepdims=True); gs[gs == 0] = 1.0
    Xn = np.concatenate([(s_te - m_) / s_, (g_te - gm) / gs], axis=1).astype(np.float32)
    with torch.no_grad():
        pred = m(torch.from_numpy(Xn)).numpy()
    out = np.empty((len(names), Phi[0].shape[0], 4))
    for k in range(4):
        out[:, :, k] = pred[:, k * POD_R_FIELD:(k + 1) * POD_R_FIELD] @ Phi[k].T + Mu[k]
    return out


_A_CACHE, _B_CACHE = {}, {}


def a_predict(sample, task, device="cuda:1"):
    """A 链路单样本推理 -> 4 场量（物理量）。返回 (pred(N,4), true(N,4))。"""
    if task not in _A_CACHE:
        rd = RunDir(task, "A")
        m = OfficialGraphSAGE().to(device)
        st = torch.load(os.path.join(rd.dir, "best_model.pt"), map_location=device)
        m.load_state_dict(st["model_state"], strict=False)
        m.eval()
        _A_CACHE[task] = m
    model = _A_CACHE[task]
    mi, si, mo, so = a_coef_norm(task)

    x = torch.from_numpy(sample["x"])
    xn = (x - torch.from_numpy(mi)) / torch.from_numpy(si)
    P = sample["pos"]
    N = P.shape[0]
    pred = np.zeros((N, 4)); cnt = np.zeros(N, dtype=int)
    seen = np.zeros(N, dtype=bool)
    rng = random.Random(SEED)
    with torch.no_grad():
        for _ in range(200):
            if seen.all():
                break
            sel = rng.sample(range(N), min(SUBSAMP, N))
            it = torch.tensor(sel, dtype=torch.long)
            e = build_radius_graph(P[sel], device)
            o = model(xn[it].to(device), e.to(device)).cpu().numpy()
            pred[sel] += o; cnt[sel] += 1; seen[sel] = True
    pred = pred / np.maximum(cnt, 1)[:, None]
    pred = pred * so + mo
    return pred, sample["y"].astype(np.float64)


def b_predict(sample, task, device="cuda:1"):
    """B 链路 -> 256 点 Cp（前缘锚定弧长序）与 POD 同口径真值。"""
    if task not in _B_CACHE:
        man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
        pod = load_pod_basis(man[task + "_train"], wallcp_npz=WALLCP_NPZ)
        rd = RunDir(task, "B_official")
        m = BBackbone(dim_in=133, dim_out=POD_R, hidden=256, n_layers=4).to(device)
        st = torch.load(os.path.join(rd.dir, "best_model.pt"), map_location=device)
        m.load_state_dict(st["model_state"])
        m.eval()
        smean, sstd, gmean, gstd = b_stats(task)
        _B_CACHE[task] = (pod, m, (smean, sstd, gmean, gstd))
    pod, m, (smean, sstd, gmean, gstd) = _B_CACHE[task]

    nm = sample["name"]
    x1_ = ((np.asarray(input_vector_5scalar(nm)) - smean.ravel()) / sstd.ravel()
           ).ravel().astype(np.float32)
    x2_ = ((load_all_geometric([nm], TASK_ROOT)[0] - gmean.ravel()) / gstd.ravel()
           ).astype(np.float32)
    X = np.concatenate([x1_, x2_])
    with torch.no_grad():
        alpha = m(torch.from_numpy(X[None]).to(device)).cpu().numpy()
    cp_pred = cp_from_alpha(alpha, pod["Phi"].astype(np.float64),
                            pod["mu"].astype(np.float64))[0]
    names = pod["names_all"]
    cp_true = pod["CP_all"][names.index(nm)].astype(np.float64)
    return cp_pred, cp_true


# ---------------- 画图 1：Cp 曲线（3 列：真值 | A | B） ----------------

def _cfd_surface_cp(sample, n_out=64):
    """CFD 真值表面 Cp -> 弦向 x/c 上下表面（64 站位）。"""
    S = sample["surf"]
    P = sample["pos"][S]
    q = 0.5 * sample["par"]["U_inf"] ** 2
    cp = sample["y"][S, 2] / q
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    xs = np.linspace(0, 1, n_out)
    up = np.full(n_out, np.nan); lo = np.full(n_out, np.nan)
    for i, xx in enumerate(xs * (x1 - x0) + x0):
        band = np.abs(P[:, 0] - xx) < (x1 - x0) / n_out * 0.5
        if band.sum() < 2:
            continue
        ys = P[band, 1]
        up[i] = cp[band][np.argmax(ys)]
        lo[i] = cp[band][np.argmin(ys)]
    return xs, up, lo


def fig_cp(task, samples, device):
    """3 列 × 6 行：真值(CFD) | A 链路 | B 链路(POD)。横轴统一 x/c。"""
    fig, axes = plt.subplots(6, 3, figsize=(15, 18))
    for r, s in enumerate(samples):
        nm = s["name"]
        xs_t, up_t, lo_t = _cfd_surface_cp(s)
        pred, _ = a_predict(s, task, device)
        # A 预测的表面 Cp，同一站位口径
        S = s["surf"]
        P = s["pos"][S]
        q = 0.5 * s["par"]["U_inf"] ** 2
        cp_a_surf = pred[S, 2] / q
        x0, x1 = P[:, 0].min(), P[:, 0].max()
        n_out = 64
        xs = np.linspace(0, 1, n_out)
        up_a = np.full(n_out, np.nan); lo_a = np.full(n_out, np.nan)
        for i, xx in enumerate(xs * (x1 - x0) + x0):
            band = np.abs(P[:, 0] - xx) < (x1 - x0) / n_out * 0.5
            if band.sum() < 2:
                continue
            ys = P[band, 1]
            up_a[i] = cp_a_surf[band][np.argmax(ys)]
            lo_a[i] = cp_a_surf[band][np.argmin(ys)]
        cp_b, _ = b_predict(s, task, device)
        xs_b, up_b, lo_b, _, _ = bx.b_curve_to_xchord(nm, TASK_ROOT, cp_b)

        def mae(a, b_):
            m = ~(np.isnan(a) | np.isnan(b_))
            return float(np.mean(np.abs(a[m] - b_[m]))) if m.any() else float("nan")

        mae_a = mae(up_a, up_t)
        mae_a_lo = mae(lo_a, lo_t)
        mae_b = mae(up_b, up_t)
        mae_b_lo = mae(lo_b, lo_t)

        panels = [
            (up_t, lo_t, None, None, "true (CFD)"),
            (up_a, lo_a, mae_a, mae_a_lo, "A-chain"),
            (up_b, lo_b, mae_b, mae_b_lo, "B-chain (POD)"),
        ]
        for c, (pu, pl, m1, m2, title) in enumerate(panels):
            ax = axes[r, c]
            ax.plot(xs_t, up_t, "k-", lw=1.1, alpha=0.8)
            ax.plot(xs_t, lo_t, "k--", lw=1.1, alpha=0.8)
            ax.plot(xs, pu, "r-", lw=1.3)
            ax.plot(xs, pl, "r--", lw=1.3)
            if m1 is None:
                t = "%s" % title
            else:
                t = "%s  MAE: up=%.3f lo=%.3f" % (title, m1, m2)
            ax.set_title(t, fontsize=9)
            ax.set_ylim(-2.5, 1.5)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.set_ylabel("Cp")
            if c == 0:
                ax.text(0.02, 0.06, nm[:26], transform=ax.transAxes,
                        fontsize=6, alpha=0.7)
    for c, lab in enumerate(["true (CFD)", "A-chain (GNN)", "B-chain (POD)"]):
        axes[-1, c].set_xlabel("x/c")
    fig.suptitle(
        "task=%s  Cp vs x/c   black=true   red=prediction   "
        "(x=0 附近真值差异来自前缘采样密度，非模型误差)" % task, fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "%s_cp.png" % task)
    fig.savefig(p, dpi=130); plt.close(fig)
    print("[fig] %s" % p, flush=True)


# ---------------- 画图 2：流场云图（每种场一张，3 列：真值 | A | B） ----------------

def fig_field(task, samples, device, field_idx, field_name):
    """6 行(样本) × 3 列(真值 | A链路 | B全场版)。
    真值/A 是逐 CFD 点，B 全场版在 256×128 均匀网格上重构，用同色标 vmin/vmax 对比。
    误差以 A 列标题里的平均相对误差给出（B 为网格域值，不作逐点误差标注）。"""
    ns = len(samples)
    names = [s["name"] for s in samples]
    bf = b_field_predict(names, task)   # (ns, ncell, 4) 物理量
    gfxy = _BFIELD_CACHE[task]["gxy"]
    fig, axes = plt.subplots(ns, 3, figsize=(13.5, 3.0 * ns))
    if ns == 1:
        axes = axes[None, :]
    for r, s in enumerate(samples):
        pred, true = a_predict(s, task, device)
        P = s["pos"]
        t = true[:, field_idx]
        pr = pred[:, field_idx]
        bfv = bf[r, :, field_idx]      # (ncell,) B 网格重构值
        rel = np.linalg.norm(pr - t) / max(np.linalg.norm(t), 1e-12)
        vmin, vmax = np.nanmin(t), np.nanmax(t)

        ax = axes[r, 0]
        sc = ax.scatter(P[:, 0], P[:, 1], c=t, s=0.4, cmap="viridis",
                        vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_ylabel("%s\n%s" % (field_name, s["name"][:20]), fontsize=6)
        if r == 0:
            ax.set_title("true (CFD)", fontsize=10)

        ax = axes[r, 1]
        sc = ax.scatter(P[:, 0], P[:, 1], c=pr, s=0.4, cmap="viridis",
                        vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046)
        if r == 0:
            ax.set_title("A-chain (GNN)", fontsize=10)
        ax.set_xlabel("mean rel.err = %.3f" % rel, fontsize=8)

        ax = axes[r, 2]
        sc = ax.scatter(gfxy[:, 0], gfxy[:, 1], c=bfv, s=0.4, cmap="viridis",
                        vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046)
        if r == 0:
            ax.set_title("B-chain (POD full-field)", fontsize=10)
            ax.set_xlabel("uniform 256x128 grid (coarse near wall)", fontsize=8)

        for c in range(3):
            axes[r, c].set_aspect("equal")
    fig.suptitle("task=%s  field=%s   true | A-chain | B full-field (POD on uniform grid)"
                 % (task, field_name), fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "%s_field_%s.png" % (task, field_name))
    fig.savefig(p, dpi=110); plt.close(fig)
    print("[fig] %s" % p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--nsamp", type=int, default=N_SAMP)
    ap.add_argument("--only", default=None, choices=["cp", "field"])
    args = ap.parse_args()

    for task in args.tasks:
        names = pick_names(task, args.nsamp)
        print("[viz] %s: %s" % (task, names), flush=True)
        samples = [load_sample(nm) for nm in names]
        print("[viz] %s: loaded %d" % (task, len(samples)), flush=True)
        if args.only in (None, "cp"):
            fig_cp(task, samples, args.device)
        if args.only in (None, "field"):
            for fi, fn in zip(range(4), FIELD_NAMES):
                fig_field(task, samples, args.device, fi, fn)
    print("[viz] ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

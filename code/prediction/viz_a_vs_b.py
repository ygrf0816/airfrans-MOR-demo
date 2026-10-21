"""可视化：对每个任务从测试集随机抽 6 个样本，画 真值 / A链路 / B链路 三条 Cp 曲线。

  A 链路：用 train_A_gnn.py 训练好的 best_model.pt（官方 GraphSAGE 全字段），
          对同一个样本逐点预测 [u,v,p,nu_t]，取表面点的静压 p 算 Cp（口径同官方）。
  B 链路：用 train_B_transformer.py 训练好的 best_model.pt（POD 降维），
          输出 256 点 Cp 曲线（前缘锚定、弧长归一化）后换算到弦向 x/c 与官方叠画。
  真值：数据集 airfrans.dataset.load 表面点（.vtp），按弦向 x 分上下表面算 Cp。
  Cp 定义统一：Cp = p/(0.5 U^2)（p 为数据集里的静压 p/ρ 分量，三条曲线同一基准）。

输出：04_results/prediction/figs/{task}_6samples_cp.png（2行×3列，每格 1 样本）
"""
import os, sys, json, random
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from runkit import RunDir
from train_A_gnn import (OfficialGraphSAGE, build_radius_graph, SUBSAMP)
from train_B_transformer import BBackbone, load_pod_basis, cp_from_alpha, POD_R
from geometry_input import load_all_geometric, input_vector_5scalar, decode_sim


def b_Uinf(name):
    return decode_sim(name)["U_inf"]

FIGDIR = os.path.normpath(os.path.join(_HERE, "..", "..",
                                       "04_results", "prediction", "figs"))
os.makedirs(FIGDIR, exist_ok=True)
TASK_ROOT = "/home/user/airfrans/data/Dataset"
WALLCP_NPZ = "/home/user/airfrans/pod/wallcp_1000.npz"
TASKS = ["full", "scarce", "reynolds", "aoa"]
N_SAMP = 6
SEED = 20261004

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_test_samples(task, n=N_SAMP):
    import airfrans
    _, names = airfrans.dataset.load(TASK_ROOT, task, train=False)
    rng = random.Random(SEED + len(task) * 131)
    return rng.sample(list(names), n)


def _surface_cp(name, p_wall, task_root=TASK_ROOT, n_out=64):
    """表面点静压 p_wall -> 弦向 x/c 上下表面 Cp = p / (0.5 U^2)。
    真值、A 预测都走这里，保证同一基准。
    """
    import vtkxml2 as vx
    fp = os.path.join(task_root, name, name + "_aerofoil.vtp")
    pts, _, _, _, _ = vx.read_vtkxml(fp)
    X = pts[:, :2]
    x0, x1 = X[:, 0].min(), X[:, 0].max()
    q = 0.5 * b_Uinf(name) ** 2
    xs = np.linspace(0, 1, n_out)
    up = np.full(n_out, np.nan); lo = np.full(n_out, np.nan)
    for i, xx in enumerate(xs * (x1 - x0) + x0):
        band = np.abs(X[:, 0] - xx) < (x1 - x0) / n_out * 0.5
        if band.sum() < 2:
            continue
        ys = X[band, 1]; pp = p_wall[band] / q
        imax, imin = ys.argmax(), ys.argmin()
        up[i] = pp[imax]; lo[i] = pp[imin]
    return xs, up, lo


def true_cp(name):
    """真值：直接从 .vtp 表面点取官方数据集的静压 p（P2 数据源口径）。"""
    import vtkxml2 as vx
    fp = os.path.join(TASK_ROOT, name, name + "_aerofoil.vtp")
    _, pdata, _, _, _ = vx.read_vtkxml(fp)
    p = pdata["p"]
    return _surface_cp(name, p)


def a_predict(name, task, device="cuda:1", cache={}):
    """A 链路：单样本官方 Infer_test 口径（反复采 32k 直到全覆盖）推理。
    airfrans.dataset.load 返回原始 ndarray（N,12），这里按官方列布局自己拆解：
      0:2=pos, 7:11=目标(u,v,p,nu_t), 11=surface bool。
    归一化口径必须和训练时一致：coef_norm 在 task 的 train 部分上算。
    """
    import airfrans
    key = ("load", task)
    if key not in cache:
        data_tr, _ = airfrans.dataset.load(TASK_ROOT, task, train=True)
        data_te, names_te = airfrans.dataset.load(TASK_ROOT, task, train=False)
        cache[key] = (data_tr, data_te, list(names_te))
    data_tr, data_te, names_te = cache[key]

    k2 = ("coef", task)
    if k2 not in cache:
        n = 0
        sx = np.zeros(7); sxx = np.zeros(7)
        for b in data_tr:
            b = np.asarray(b, dtype=np.float64)[:, 0:7]
            sx += b.sum(0); sxx += (b * b).sum(0); n += b.shape[0]
        mean_in = (sx / n).astype(np.float32)
        std_in = np.sqrt(np.maximum(sxx / n - (sx / n) ** 2, 0)).astype(np.float32) + 1e-8
        cache[k2] = (mean_in, std_in)
    mean_in, std_in = cache[k2]

    idx = names_te.index(name)
    a = np.asarray(data_te[idx])
    N = a.shape[0]
    P = a[:, 0:2].astype(np.float32)
    S = a[:, 11].astype(bool)
    x_norm = (a[:, 0:7].astype(np.float32) - mean_in) / std_in

    model = _a_model(task, device)
    pred = np.zeros((N, 4)); cnt = np.zeros(N, dtype=int)
    seen = np.zeros(N, dtype=bool)
    rng = random.Random(SEED)
    xt = torch.from_numpy(x_norm)
    for _ in range(200):
        if seen.all():
            break
        sel = rng.sample(range(N), min(SUBSAMP, N))
        it = torch.tensor(sel, dtype=torch.long)
        e = build_radius_graph(P[sel], device)
        o = model(xt[it].to(device), e.to(device)).cpu().numpy()
        pred[sel] += o; cnt[sel] += 1; seen[sel] = True
    pred = pred / np.maximum(cnt, 1)
    coef = _a_coef(task)
    std_out = np.asarray(coef["std_out"], dtype=np.float32) + 1e-8
    mean_out = np.asarray(coef["mean_out"], dtype=np.float32)
    p_dn = pred[S, 2] * std_out[2] + mean_out[2]
    return _surface_cp(name, p_dn)


_MODEL_CACHE, _COEF_CACHE = {}, {}


def _a_model(task, device):
    if task not in _MODEL_CACHE:
        rd = RunDir(task, "A")
        m = OfficialGraphSAGE().to(device)
        st = torch.load(os.path.join(rd.dir, "best_model.pt"), map_location=device)
        m.load_state_dict(st["model_state"], strict=False)
        m.eval()
        _MODEL_CACHE[task] = m
    return _MODEL_CACHE[task]


def _a_coef(task):
    if task not in _COEF_CACHE:
        meta = json.load(open(os.path.join(RunDir(task, "A").dir,
                                           "eval_metrics.json"),
                              encoding="utf-8"))
        _COEF_CACHE[task] = meta["coef_norm"]
    return _COEF_CACHE[task]


def b_predict(names, task, device="cuda:1", task_root=TASK_ROOT):
    """B 链路：POD 降维 256 点 Cp（前缘锚定）-> 弦向 x/c 上下表面。"""
    import b_cp_to_xchord
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"),
                        encoding="utf-8"))
    train_names = man[task + "_train"]
    pod = load_pod_basis(train_names, wallcp_npz=WALLCP_NPZ)
    rd = RunDir(task, "B_official")
    m = BBackbone(dim_in=133, dim_out=POD_R, hidden=256, n_layers=4).to(device)
    st = torch.load(os.path.join(rd.dir, "best_model.pt"), map_location=device)
    m.load_state_dict(st["model_state"])
    m.eval()

    # 归一化口径和 train_B_transformer.train_official 完全一致
    n_val = max(1, int(0.1 * len(train_names)))
    tr_eff = train_names[:len(train_names) - n_val]
    scal = np.stack([input_vector_5scalar(n) for n in tr_eff])
    smean, sstd = scal.mean(0, keepdims=True), scal.std(0, keepdims=True)
    sstd[sstd == 0] = 1
    geo = load_all_geometric(tr_eff, TASK_ROOT)
    gmean, gstd = geo.mean(0, keepdims=True), geo.std(0, keepdims=True)
    gstd[gstd == 0] = 1

    out = {}
    geom_cache = load_all_geometric(names, TASK_ROOT)   # 一次性读 6 个 vtp
    for k, nm in enumerate(names):
        x1_ = ((np.asarray(input_vector_5scalar(nm)) - smean) / sstd).ravel()\
            .astype(np.float32)
        x2_ = ((geom_cache[k] - gmean.ravel()) / gstd.ravel()).astype(np.float32)
        X = np.concatenate([x1_, x2_])
        with torch.no_grad():
            alpha = m(torch.from_numpy(X[None]).to(device)).cpu().numpy()
        cp = cp_from_alpha(alpha, pod["Phi"].astype(np.float64),
                           pod["mu"].astype(np.float64))[0]
        # cp 已是 Cp（wall_cp 里已除 q），只借几何换算到 x/c，不再除 q
        xs, up, lo, _, _ = b_cp_to_xchord.b_curve_to_xchord(nm, TASK_ROOT, cp)
        out[nm] = (xs, up, lo)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    args = ap.parse_args()

    for task in args.tasks:
        print(f"[viz] {task}: pick {N_SAMP} test samples...", flush=True)
        names = pick_test_samples(task)
        print(f"[viz] {task}: B chain predict...", flush=True)
        bres = b_predict(names, task, args.device)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, nm in enumerate(names):
            ax = axes[i // 3][i % 3]
            xs_t, up_t, lo_t = true_cp(nm)
            xs_a, up_a, lo_a = a_predict(nm, task, args.device)
            xs_b, up_b, lo_b = bres[nm]
            ax.plot(xs_t, up_t, "k-", lw=1.4, label="真值·上")
            ax.plot(xs_t, lo_t, "k--", lw=1.4, label="真值·下")
            ax.plot(xs_a, up_a, "r-", lw=1, alpha=0.9, label="A链路·上")
            ax.plot(xs_a, lo_a, "r--", lw=1, alpha=0.9, label="A链路·下")
            ax.plot(xs_b, up_b, "b-", lw=1, alpha=0.9, label="B链路(MOR升维)·上")
            ax.plot(xs_b, lo_b, "b--", lw=1, alpha=0.9, label="B链路(MOR升维)·下")
            ax.set_title(nm, fontsize=9)
            ax.set_xlabel("x/c")
            ax.set_ylabel("Cp")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, ncol=2)
        fig.suptitle(f"task={task}  真值(黑) vs A链路(红) vs B链路MOR升维(蓝)，"
                     f"6 随机测试样本", fontsize=13)
        fig.tight_layout()
        out = os.path.join(FIGDIR, f"{task}_6samples_cp.png")
        fig.savefig(out, dpi=150)
        print(f"[viz] {task}: saved {out}", flush=True)
    print("[viz] ALL_DONE")


if __name__ == "__main__":
    main()

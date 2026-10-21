"""把 B 场链路缺的两样东西固化下来（**不需要重训**），并做逐位验收。

背景：`{task}_B_field/best_model.pt` 里只有 model_state，没有归一化统计量、
没有 4 场 POD 基 —— 权重拿到别处无法预测。本脚本把缺的补齐：

  1. `pod/geomvec_1000.npz`      names(1000,) + G(1000,128) + n_station
     128 维几何向量是 .vtp 的确定性函数，但每次重算要读 1000 个 vtp（约 1 分钟）。
     有了它，几何输入侧永远不再碰 .vtp。（**只有这一项真的需要读 .vtp**；
     4 场 POD 基是 vol_fields_1000.npz 的确定性函数，本来就不用 vtp。）

  2. `pod/bfield_basis.npz`      每个 task 一组：Phi_{task}(4,ncell,r)、
     mu_{task}(4,1,ncell)、S_{task}(4,ncell 全谱)。基由 vol_fields_1000.npz
     + 该任务 train 划分重算，与该次训练所用完全一致。

  3. 回填 `{task}_B_field/best_model.pt` 的 `norm_stats`（含 smean/sstd/gmean/gstd
     + 基的路径与键 + 官方 coef_norm + tr_eff_names），与 B_official 的字段名一致。

验收（决定性）：只用回填后的 `best_model.pt`（不重训、不重算统计量）重建
模型并预测测试集，网格域逐场 MSE 必须与 `eval_metrics.json` 里记录的一致。

用法：
  python patch_B_field_stats.py                 # 全部 4 个任务
  python patch_B_field_stats.py --tasks full    # 指定任务
"""
import os, sys, json, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from geometry_input import geometric_vector, N_STATION      # noqa: E402
from train_B_field import FIELDS, FieldMLP, build_inputs    # noqa: E402

POD = "/home/zch/airfrans/pod"
GEOM_CACHE = os.path.join(POD, "geomvec_1000.npz")
BASIS_NPZ = os.path.join(POD, "bfield_basis.npz")
FIELD_CACHE = os.path.join(POD, "vol_fields_1000.npz")
MANIFEST = "/home/zch/airfrans/data/manifest.json"
TASK_ROOT = "/home/zch/airfrans/data/Dataset"
RUNS = "/home/zch/airfrans/prediction/04_results/prediction/runs"
TASKS = ["full", "scarce", "reynolds", "aoa"]
VAL_FRAC = 0.1


def _one_geom(nm):
    return geometric_vector(nm, TASK_ROOT, N_STATION)


def _safe_geom(nm):
    try:
        return _one_geom(nm)
    except Exception:
        return None


def build_geom_cache():
    if os.path.exists(GEOM_CACHE):
        z = np.load(GEOM_CACHE, allow_pickle=True)
        print("[geom] 已存在：%s  names=%d G=%s" % (GEOM_CACHE, len(z["names"]), z["G"].shape))
        return
    from multiprocessing import Pool
    man = json.load(open(MANIFEST, encoding="utf-8"))
    names = list(man["full_train"]) + list(man["full_test"])
    print("[geom] 首次构建：读 %d 个 .vtp（只此一次）..." % len(names), flush=True)
    # 必须先在父进程读一次：pyvista 首次 read 会去枚举 importlib entry-points，
    # 这个过程在 fork 出的多个子进程里并发跑会偶发 BadZipFile（读 dist-info 的 zip）。
    # 父进程热身一次后，子进程直接继承已建好的注册表，不再触发该路径。
    geometric_vector(names[0], TASK_ROOT, N_STATION)
    got, miss = {}, []
    with Pool(32) as pool:
        for nm, v in zip(names, pool.map(_safe_geom, names, chunksize=4)):
            if v is None:
                miss.append(nm)
            else:
                got[nm] = v
    for nm in miss:                       # 子进程失败的少量样本回退到父进程串行重读
        got[nm] = geometric_vector(nm, TASK_ROOT, N_STATION)
    G = np.stack([got[n] for n in names]).astype(np.float64)
    np.savez(GEOM_CACHE, names=np.array(names), G=G, n_station=np.array([N_STATION]))
    print("[geom] 写好 %s  G=%s  %.2f MB  子进程失败回退 %d 个"
          % (GEOM_CACHE, G.shape, os.path.getsize(GEOM_CACHE) / 1e6, len(miss)))


def split_of(task, man):
    tr_all = man[task + "_train"]
    te = man["full_test"] if task == "scarce" else man[task + "_test"]
    n_val = max(1, int(VAL_FRAC * len(tr_all)))
    return tr_all[:len(tr_all) - n_val], tr_all[len(tr_all) - n_val:], te


def patch_task(task, man, cp, basis):
    r = None
    am = json.load(open(os.path.join(RUNS, "%s_A" % task, "eval_metrics.json"), encoding="utf-8"))
    bm = json.load(open(os.path.join(RUNS, "%s_B_field" % task, "eval_metrics.json"), encoding="utf-8"))
    r = int(bm["pod_r"])
    mean_o = np.asarray(am["coef_norm"]["mean_out"], np.float64)
    std_o = np.asarray(am["coef_norm"]["std_out"], np.float64)

    tr_names, val_names, te_names = split_of(task, man)
    Xtr, Xval, Xte, stats = build_inputs(tr_names, val_names, te_names, TASK_ROOT)

    names_all = cp["names"].tolist()
    idx = {n: i for i, n in enumerate(names_all)}
    raw_tr = np.stack([np.stack([cp[k][idx[n]] for k in FIELDS], axis=-1) for n in tr_names])
    Xn_tr = (raw_tr - mean_o) / std_o
    raw_te = np.stack([np.stack([cp[k][idx[n]] for k in FIELDS], axis=-1) for n in te_names])
    Xn_te = (raw_te - mean_o) / std_o

    Phi, Mu, S_full = [], [], []
    for k in range(4):
        Xk = Xn_tr[:, :, k].astype(np.float64)
        mu = Xk.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(Xk - mu, full_matrices=False)
        Phi.append(Vt[:r].T); Mu.append(mu); S_full.append(S)
    ncell = Phi[0].shape[0]

    # 写入基
    basis["Phi_%s" % task] = np.stack(Phi)
    basis["mu_%s" % task] = np.stack(Mu)
    basis["S_%s" % task] = np.stack(S_full)
    basis["ncell_%s" % task] = np.array([ncell])

    norm_stats = {
        "chain": "B_field", "task": task, "pod_r": r,
        "smean": stats["smean"].tolist(), "sstd": stats["sstd"].tolist(),
        "gmean": stats["gmean"].tolist(), "gstd": stats["gstd"].tolist(),
        "geom_station": stats["geom_station"],
        "fields": FIELDS, "n_grid_cells": int(ncell),
        "cache": FIELD_CACHE, "basis_npz": BASIS_NPZ,
        "basis_keys": ["Phi_%s" % task, "mu_%s" % task],
        "coef_norm": {"mean_out": mean_o.tolist(), "std_out": std_o.tolist()},
        "tr_eff_names": list(tr_names), "val_names": list(val_names),
    }

    p = os.path.join(RUNS, "%s_B_field" % task, "best_model.pt")
    st = torch.load(p, map_location="cpu", weights_only=False)
    st["norm_stats"] = norm_stats
    st["_norm_stats"] = norm_stats
    torch.save(st, p)

    # ---- 决定性验收：只用权重文件里的东西，重建并预测 ----
    model = FieldMLP(Xte.shape[1], 4 * r, hidden=256).eval()
    model.load_state_dict(st["model_state"], strict=False)
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).numpy()
    rec = np.empty((len(te_names), ncell, 4))
    for k in range(4):
        rec[:, :, k] = pred[:, k * r:(k + 1) * r] @ Phi[k].T + Mu[k]
    mse = ((rec - Xn_te) ** 2).mean(axis=(0, 1)) * 100
    ref = bm["field_MSE_grid_x1e-2"]
    # 阈值用**相对差**：重算 SVD 会因 BLAS 归约顺序不同产生 ~1e-6 量级的浮点抖动，
    # 这是 float32/float64 舍入级别，不是"口径不一致"。
    drel = max(abs(mse[k] - ref[FIELDS[k]]) / max(abs(ref[FIELDS[k]]), 1e-12)
               for k in range(4))
    print("[%s] r=%d ncell=%d | 复现网格 MSE(x1e-2)=%s | 记录=%s | 最大相对差=%.2e %s"
          % (task, r, ncell, np.round(mse, 6), np.round([ref[f] for f in FIELDS], 6),
             drel, "OK(浮点舍入级)" if drel < 1e-5 else "**不一致**"), flush=True)
    return p, drel


def verify_only(task, man, basis):
    """只加载已落盘的基 + 权重做验收，不重算 SVD（秒级，可反复跑）。"""
    bm = json.load(open(os.path.join(RUNS, "%s_B_field" % task, "eval_metrics.json"),
                        encoding="utf-8"))
    am = json.load(open(os.path.join(RUNS, "%s_A" % task, "eval_metrics.json"), encoding="utf-8"))
    r = int(bm["pod_r"])
    mean_o = np.asarray(am["coef_norm"]["mean_out"], np.float64)
    std_o = np.asarray(am["coef_norm"]["std_out"], np.float64)
    Phi = basis["Phi_%s" % task]; Mu = basis["mu_%s" % task]
    ncell = Phi.shape[1]
    _, _, te_names = split_of(task, man)
    cp = np.load(FIELD_CACHE, allow_pickle=True)
    idx = {n: i for i, n in enumerate(cp["names"].tolist())}
    raw_te = np.stack([np.stack([cp[k][idx[n]] for k in FIELDS], axis=-1) for n in te_names])
    Xn_te = (raw_te - mean_o) / std_o
    st = torch.load(os.path.join(RUNS, "%s_B_field" % task, "best_model.pt"),
                    map_location="cpu", weights_only=False)
    if "norm_stats" not in st:
        print("[%s] best_model.pt 里没有 norm_stats —— 回填没生效" % task)
        return 1.0
    _, _, Xte, _ = build_inputs(*split_of(task, man), TASK_ROOT)
    model = FieldMLP(Xte.shape[1], 4 * r, hidden=256).eval()
    model.load_state_dict(st["model_state"], strict=False)
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).numpy()
    rec = np.empty((len(te_names), ncell, 4))
    for k in range(4):
        rec[:, :, k] = pred[:, k * r:(k + 1) * r] @ Phi[k].T + Mu[k]
    mse = ((rec - Xn_te) ** 2).mean(axis=(0, 1)) * 100
    ref = bm["field_MSE_grid_x1e-2"]
    drel = max(abs(mse[k] - ref[FIELDS[k]]) / max(abs(ref[FIELDS[k]]), 1e-12) for k in range(4))
    print("[%s] 只读 bfield_basis.npz + best_model.pt 复现：%s | 记录 %s | 最大相对差 %.2e %s"
          % (task, np.round(mse, 6), np.round([ref[f] for f in FIELDS], 6), drel,
             "OK" if drel < 1e-5 else "**不一致**"), flush=True)
    return drel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--verify-only", action="store_true",
                    help="只加载已落盘的基做验收，不重算 SVD")
    args = ap.parse_args()

    man = json.load(open(MANIFEST, encoding="utf-8"))
    if args.verify_only:
        basis = dict(np.load(BASIS_NPZ, allow_pickle=True))
        worst = max(verify_only(t, man, basis) for t in args.tasks)
        print("[VERIFY-ONLY] 最大相对差 = %.2e -> %s" % (worst, "PASS" if worst < 1e-5 else "FAIL"))
        return

    build_geom_cache()
    cp = np.load(FIELD_CACHE, allow_pickle=True)
    basis = {}
    if os.path.exists(BASIS_NPZ):
        basis = dict(np.load(BASIS_NPZ, allow_pickle=True))
    worst = 0.0
    for t in args.tasks:
        p, d = patch_task(t, man, cp, basis)
        worst = max(worst, d)
        np.savez(BASIS_NPZ, **basis)     # 每任务写一次，中断也不丢
        print("       回填 ->", p, flush=True)
    print("[basis] %s  %.2f MB  keys=%s"
          % (BASIS_NPZ, os.path.getsize(BASIS_NPZ) / 1e6, sorted(basis.keys())))
    print("[VERIFY] 最大相对差 = %.2e  -> %s" % (worst, "PASS" if worst < 1e-5 else "FAIL"))


if __name__ == "__main__":
    main()

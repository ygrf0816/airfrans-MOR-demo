"""把缺失的归一化统计量补进已训练的权重文件（不需要重训）。

背景：训练脚本原先只存 model_state，没存归一化统计量，导致权重加载后
无法对新样本做与训练一致的归一化，等于不可用。

已验证（误差严格为 0）：用重算的统计量 + 已存权重，能逐位复现
eval_metrics.json 里记录的 cp_relL2_test。所以这里的补写是等价还原，
不是近似 —— 补完的权重与"训练时直接存统计量"的结果一致。

做法：
  1. 重算 A 的 coef_norm（其实 eval_metrics.json 里已有，直接读）
  2. 重算 B 的 smean/sstd/gmean/gstd + POD 基 Phi/mu
  3. 写回 best_model.pt 的 _norm_stats / coef_norm 字段
  4. 补写 hyperparams.json
"""
import os, sys, json, shutil
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from runkit import RunDir  # noqa: E402
from geometry_input import load_all_geometric, input_vector_5scalar  # noqa: E402
from train_B_transformer import load_pod_basis, POD_R, WALLCP_NPZ_DEFAULT  # noqa: E402
import train_A_gnn as A  # noqa: E402
import train_B_transformer as B  # noqa: E402

ROOT = "/home/user/airfrans"
TASK_ROOT = os.path.join(ROOT, "data", "Dataset")
TASKS = ["full", "scarce", "reynolds", "aoa"]
BACKUP = os.path.join(ROOT, "prediction", "04_results", "prediction", "_weights_backup_before_stats")


def backup(path):
    os.makedirs(BACKUP, exist_ok=True)
    dst = os.path.join(BACKUP, os.path.relpath(path, os.path.join(ROOT, "prediction")))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(path, dst)


def patch_A(task):
    rd = RunDir(task, "A")
    p = os.path.join(rd.dir, "best_model.pt")
    meta_path = os.path.join(rd.dir, "eval_metrics.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    coef = meta["coef_norm"]
    st = torch.load(p, map_location="cpu")
    if "coef_norm" in st:
        print("[A/%s] already patched, skip" % task); return
    backup(p)
    st["coef_norm"] = coef
    torch.save(st, p)
    # hyperparams
    hp = {
        "task": task, "chain": "A", "kind": "sage_official",
        "enc": A.ENC, "dec": A.DEC, "hidden": A.HIDDEN, "nb_hidden": A.NB_HIDDEN,
        "radius": A.RADIUS, "max_neighbors": A.MAX_NEIGHBORS,
        "subsample": A.SUBSAMP, "lambda_surf": A.LAMBDA_SURF,
        "n_epochs": meta.get("n_epochs"), "lr": 1e-3,
        "seed": 0, "optimizer": "Adam (no weight_decay), OneCycleLR",
        "input": "7-dim point (x,y,Ux,Uy,sdf,nx,ny)",
        "output": "4-dim point (u,v,p,nu_t)",
        "train_samples": meta.get("train_samples"),
        "val_samples": meta.get("val_samples"),
        "test_samples": meta.get("test_samples"),
        "_patched": "norm stats backfilled by patch_norm_stats.py",
    }
    rd.save_hyperparams(hp)
    print("[A/%s] patched coef_norm + hyperparams" % task)


def patch_B(task):
    rd = RunDir(task, "B_official")
    p = os.path.join(rd.dir, "best_model.pt")
    st = torch.load(p, map_location="cpu")
    if "norm_stats" in st:
        print("[B/%s] already patched, skip" % task); return
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    tn = man[task + "_train"]
    n_val = max(1, int(0.1 * len(tn)))
    tr_eff = tn[:len(tn) - n_val]
    # 5 标量统计量
    scal = np.stack([input_vector_5scalar(n) for n in tr_eff])
    smean, sstd = scal.mean(0, keepdims=True), scal.std(0, keepdims=True)
    sstd[sstd == 0] = 1.0
    # 几何统计量
    geo = load_all_geometric(tr_eff, TASK_ROOT)
    gmean, gstd = geo.mean(0, keepdims=True), geo.std(0, keepdims=True)
    gstd[gstd == 0] = 1.0
    # POD 基
    pod = load_pod_basis(tn, wallcp_npz=WALLCP_NPZ_DEFAULT)
    backup(p)
    st["norm_stats"] = {
        "smean": smean.tolist(), "sstd": sstd.tolist(),
        "gmean": gmean.tolist(), "gstd": gstd.tolist(),
        "pod_Phi": pod["Phi"].tolist(), "pod_mu": pod["mu"].tolist(),
        "pod_r": POD_R, "wallcp_npz": WALLCP_NPZ_DEFAULT,
        "tr_eff_names": list(tr_eff),
    }
    torch.save(st, p)
    meta = json.load(open(os.path.join(rd.dir, "eval_metrics.json"), encoding="utf-8"))
    rd.save_hyperparams({
        "task": task, "chain": "B_official",
        "pod_r": POD_R, "hidden": 256, "dim_in": 133, "n_layers": 4,
        "val_frac": 0.1, "lr": 3e-4, "wd": 1e-4, "batch": 128,
        "n_epochs": meta.get("n_epochs"), "seed": 0,
        "n_geometric": 128, "geom_station": 64,
        "wallcp_npz": WALLCP_NPZ_DEFAULT,
        "train_n_eff": meta.get("train_n_eff"), "val_n": meta.get("val_n"),
        "test_n": meta.get("test_n"),
        "input": "5 freestream scalars + 128 geometric (y/c curve)",
        "output": "r-dim POD coords -> Cp curve (256 pts, LE-anchored)",
        "_patched": "norm stats backfilled by patch_norm_stats.py",
    })
    print("[B/%s] patched norm_stats (Phi %s) + hyperparams"
          % (task, np.shape(pod["Phi"])))


if __name__ == "__main__":
    print("TASK_ROOT exists:", os.path.isdir(TASK_ROOT))
    for t in TASKS:
        patch_A(t)
        patch_B(t)
    print("PATCH_ALL_DONE")

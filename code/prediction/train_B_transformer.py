"""B 链路（官方口径重写）：(5 来流标量 + 128 维几何向量) -> POD 低维坐标 -> 还原 Cp -> 官方 C_L/C_D 评估。

设计对齐官方 AirfRANS（可回溯 01_initial/）：
  1. camber 口径改为官方：4 位 M/100（M 是 hundredths of chord），5 位 0.15*L。
  2. 几何进模型：翼型表面弦向 64 站上下表面 y/c 曲线（128 维），不是靠文件名猜 camber。
  3. 骨干：MLP（残差 + LayerNorm + GELU），不再用 5-token 自注意力（无 token 序列结构）。
  4. 评估：还原 Cp 后按官方目标口径算 C_L/C_D 相对误差 + Spearman，
     对标官方 GraphSAGE full C_L relerr=0.517。

官方 4 任务：full / scarce / reynolds / aoa
止损点：full 任务 val 误差 > 2 x POD 截断误差 -> 停，不继续外推任务
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from input_side import standardize, INPUT_FEATURE_NAMES  # noqa: F401, E402
from geometry_input import load_all_geometric, input_vector_5scalar, N_STATION  # noqa: E402
from runkit import RunDir, smoke_gate, is_nan  # noqa: E402

# ROOT 解析：本地仓库布局是 <repo>/03_code/prediction/，故 _HERE/../.. 是仓库根；
# 但工作站上是 /home/user/airfrans/prediction/，_HERE/../.. 会算成 /home/user（错）。
# 这里改成逐个候选探测，谁存在用谁，避免路径写死后在另一个环境直接报错。
def _resolve_root():
    """返回含 04_results 的仓库根。
    本地布局: <root>/03_code/prediction/ 与 <root>/04_results
    工作站:   /home/user/airfrans/{prediction,data,pod}，评估产物在 prediction/04_results
    故两边都可能有 04_results，两个位置都探。
    """
    cands = [
        os.path.normpath(os.path.join(_HERE, "..", "..")),          # 本地仓库根
        "/home/user/airfrans",                                         # 工作站根
        _HERE,                                                        # 就是 prediction/ 本身
    ]
    for c in cands:
        for sub in (os.path.join(c, "04_results"),
                    os.path.join(c, "prediction", "04_results")):
            if os.path.isdir(sub):
                # 必须返回**真正包含 04_results 的那一层**：
                # 之前命中 c/prediction/04_results 时返回了 c，导致
                # judge_path = c/04_results/... 仍然不存在（工作站上就是踩这个坑）。
                return os.path.dirname(sub)
    return cands[0]


ROOT = _resolve_root()

# 壁面 Cp 全量 npz：优先仓库内 04_results/mor，其次工作站 pod 目录
_ANS_CANDS = [
    os.path.join(ROOT, "04_results", "mor", "p2_wallcp_anchored.npz"),
    "/home/user/airfrans/pod/wallcp_1000.npz",
]
for _c in _ANS_CANDS:
    if os.path.exists(_c):
        WALLCP_NPZ_DEFAULT = _c
        break
else:
    WALLCP_NPZ_DEFAULT = _ANS_CANDS[0]
MOR_OUT = os.path.normpath(os.path.join(ROOT, "04_results", "mor"))
POD_R = 8
SEED = 0

# TASK_ROOT：翼型表面 .vtp 数据源（P2 铁律：Dataset 是唯一数据源）
_TASKROOT_CANDS = [
    os.path.join(ROOT, "02_dataset", "Dataset"),
    os.path.join(ROOT, "data", "Dataset"),
    "/home/user/airfrans/data/Dataset",
]
for _c in _TASKROOT_CANDS:
    if os.path.isdir(_c):
        TASK_ROOT_DEFAULT = _c
        break
else:
    TASK_ROOT_DEFAULT = _TASKROOT_CANDS[0]


def load_pod_basis(task_train_names, wallcp_npz=None):
    """按本任务 train 集合重做 SVD -> 自洽 POD 基 + 全量 alpha + 全谱 S。"""
    src = wallcp_npz or WALLCP_NPZ_DEFAULT
    cp = np.load(src, allow_pickle=True)
    CP_all = cp["CP"]; names_all = cp["names"].tolist()
    idx = [names_all.index(n) for n in task_train_names]
    CPtr = CP_all[idx]
    mu = CPtr.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(CPtr - mu, full_matrices=False)
    Phi = Vt[:POD_R].T
    alpha_tr = (CPtr - mu) @ Phi
    return {"CP_all": CP_all, "names_all": names_all, "Phi": Phi, "mu": mu,
            "alpha_train": alpha_tr, "S_full": S}


def cp_from_alpha(alpha, Phi, mu):
    """POD 坐标 -> 还原 Cp 曲线（256 点）。"""
    return alpha @ Phi.T + mu


class BBackbone(nn.Module):
    """(5 来流标量 + 128 几何) -> r 维 POD 坐标。MLP 残差骨干。

    官方逐点 GNN 用 ~30K 参数；本链路是"几何压缩向量 -> 低维 POD 坐标"，
    输入 133 维 -> 输出 8 维，MLP 256/4L ~200K 参数与官方量级相当但更宽。
    """

    def __init__(self, dim_in=133, dim_out=8, hidden=256, n_layers=4, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(dim_in, hidden), nn.LayerNorm(hidden), nn.GELU()]
        for i in range(n_layers):
            layers += [nn.Linear(hidden, hidden)]
            if i < n_layers - 1:
                layers += [nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        self.tower = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim_out))

    def forward(self, x):
        # 残差旁路 res_proj 已并入 tower 首层（tower 首层 Linear(dim_in,hidden) 即残差），
        # 故 forward 直接过 tower + head，避免 (B,133) 与 (B,256) 维度冲突。
        h = self.tower(x)
        return self.head(h)


def build_loader(Xtr, al_tr, Xte, al_te, batch=128):
    Xtr = np.asarray(Xtr, dtype=np.float32); al_tr = np.asarray(al_tr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32); al_te = np.asarray(al_te, dtype=np.float32)
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(al_tr))
    te = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(al_te))
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True),
            torch.utils.data.DataLoader(te, batch_size=batch))


def train_official(task, train_names, test_names, r=POD_R, lr=1e-3, wd=1e-4,
                   n_epochs=80, batch=128, device="cuda", init_from=None,
                   wallcp_npz=None, patience=12, hidden=256, task_root=None,
                   n_geometric=128, geom_station=64, val_frac=0.1,
                   official_coefs=None, seed=SEED):
    """训练官方口径 B 链路；返回 (model, eval_metrics, rd)。

    输入 = concat(5 来流标量 z-score, 128 几何向量 z-score) -> POD r 坐标。
    几何向量来自翼型表面弦向 y/c 曲线（不同翼型自然区分）。

    验证/早停：train 末尾 val_frac 切出真验证集（官方 main.py 口径），
    测试集只用于最终评估，绝不参与早停/best_model 选择（修复 1：
    之前直接用 test 早停，B 无独立验证集，已废弃那 4 个旧结果）。
    """
    rd = RunDir(task, "B_official")
    # 可复现：同配置不同种子的 cp_relL2 可差 ~20%（0.245/0.270/0.311），
    # 单次跑不足以支撑结论；官方口径是跑 5 次报 mean±std，这里用 --seed 支持多跑。
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # 0) 验证集切分（train 末尾 10%，官方 main.py 口径；scarce 任务 train 本就少，
    #    val_frac=0.1 时至少保 1 个 val）
    n_val = max(1, int(val_frac * len(train_names)))
    n_tr_eff = max(1, len(train_names) - n_val)
    tr_eff_names = train_names[:n_tr_eff]
    val_names = train_names[n_tr_eff:]
    # 1) POD 基 + 坐标
    #    ⚠ 基必须只用 train_eff（去掉 val 的那部分）算。
    #    之前这里传的是完整 train_names，而下面第 160/172/186 行的标量、几何统计量
    #    和训练坐标都用 tr_eff_names —— 等于 val 参与了 POD 基的构造，
    #    val 上的早停因此偏乐观（典型的"定义了 train 减 10%、却用完整 train 建矩阵"）。
    #    修好后 val 才真正独立；test 本来就没进基，所以测试指标的有效性不变。
    pod = load_pod_basis(tr_eff_names, wallcp_npz)
    Phi, mu = pod["Phi"], pod["mu"]
    # 2) 5 来流标量（官方 camber 口径）z-score（train_eff 算 mean/std，val/test 复用，
    #    避免 val 信息泄漏进标准化统计）
    scal_tr = np.stack([input_vector_5scalar(n) for n in tr_eff_names])
    scal_val = np.stack([input_vector_5scalar(n) for n in val_names])
    scal_te = np.stack([input_vector_5scalar(n) for n in test_names])
    mean_s, std_s = scal_tr.mean(axis=0, keepdims=True), scal_tr.std(axis=0, keepdims=True)
    std_s[std_s == 0] = 1.0
    sXtr = ((scal_tr - mean_s) / std_s).astype(np.float32)
    sXval = ((scal_val - mean_s) / std_s).astype(np.float32)
    sXte = ((scal_te - mean_s) / std_s).astype(np.float32)
    # sm 变量已内联替代（不再调用 standardize()），保留接口以防后续调用
    sm = {"Xtr": sXtr, "mean": mean_s, "std": std_s, "Xte": sXte}
    # 3) 几何 128 向量 z-score（按 train_eff 统计，val/test 复用）
    tr_root = task_root or TASK_ROOT_DEFAULT
    geo_tr = load_all_geometric(tr_eff_names, tr_root, geom_station)
    geo_val = load_all_geometric(val_names, tr_root, geom_station)
    geo_te = load_all_geometric(test_names, tr_root, geom_station)
    gmean, gstd = geo_tr.mean(axis=0, keepdims=True), geo_tr.std(axis=0, keepdims=True)
    gstd[gstd == 0] = 1.0
    sGeo_tr = ((geo_tr - gmean) / gstd).astype(np.float32)
    sGeo_val = ((geo_val - gmean) / gstd).astype(np.float32)
    sGeo_te = ((geo_te - gmean) / gstd).astype(np.float32)
    # 4) 拼接输入（5 标量 + 128 几何 = 133 维）
    Xtr = np.concatenate([sXtr, sGeo_tr], axis=1).astype(np.float32)
    Xval = np.concatenate([sXval, sGeo_val], axis=1).astype(np.float32)
    Xte = np.concatenate([sXte, sGeo_te], axis=1).astype(np.float32)
    # 5) POD 坐标目标（基只用 train_eff 算，val/test 坐标复用同一基/mu）
    alpha_all = ((pod["CP_all"] - mu) @ Phi)
    al_tr = alpha_all[[pod["names_all"].index(n) for n in tr_eff_names]].astype(np.float32)
    al_val = alpha_all[[pod["names_all"].index(n) for n in val_names]].astype(np.float32)
    cp_test_idx = [pod["names_all"].index(n) for n in test_names]
    al_te = alpha_all[cp_test_idx].astype(np.float32)
    # 6) 模型
    dim_in = sXtr.shape[1] + sGeo_tr.shape[1]
    model = BBackbone(dim_in=dim_in, dim_out=r, hidden=hidden, n_layers=4).to(device)
    if init_from:
        state = torch.load(init_from, map_location=device)
        model.load_state_dict(state["model_state"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tr_ld, val_ld = build_loader(Xtr, al_tr, Xval, al_val, batch)
    crit = nn.MSELoss()
    best_vl, best_state = float("inf"), None
    bad_epochs = 0
    for ep in range(n_epochs):
        model.train()
        tl_sum = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tl_sum += loss.item()
        model.eval()
        # 早停/选 best_model 只用验证集（train 末尾 10%），测试集不参与（修复 1）
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                vl_sum += crit(model(xb), yb).item()
                vl_n += 1
        vl = vl_sum / max(vl_n, 1)
        rd.append_loss(ep, tl_sum / len(tr_ld), vl,
                       extra={"pod_r": r, "dim_in": dim_in, "lr": lr})
        if vl < best_vl - 1e-4:
            best_vl, bad_epochs = vl, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{task}] B_official early stop @ep{ep} (best val={best_vl:.4f})")
                break
        if is_nan(vl):
            print(f"[{task}] B_official NaN @ep{ep} -> abort"); break
    # 所有权重依赖的配置一起存（修复：之前只存 model_state，
    # 加载后没有归一化统计量就无法对新样本做与训练一致的归一化，等于不可用）
    rd.save_hyperparams({
        "task": task, "chain": "B_official",
        "pod_r": r, "hidden": hidden, "dim_in": dim_in, "n_layers": 4,
        "val_frac": val_frac, "lr": lr, "wd": wd, "batch": batch,
        "n_epochs": n_epochs, "patience": patience, "seed": int(seed),
        "n_geometric": n_geometric, "geom_station": geom_station,
        "wallcp_npz": wallcp_npz, "official_coefs": official_coefs,
        "task_root": tr_root,
        "train_n_eff": int(len(Xtr)), "val_n": int(len(val_names)),
        "test_n": int(len(Xte)),
        "input": "5 freestream scalars + 128 geometric (y/c curve)",
        "output": "r-dim POD coords -> Cp curve (256 pts, LE-anchored)",
    })
    rd.save_best({"model_state": best_state, "task": task,
                  "norm_stats": {
                      # B 输入归一化（train_eff 上算）
                      "smean": mean_s.tolist(), "sstd": std_s.tolist(),
                      "gmean": gmean.tolist(), "gstd": gstd.tolist(),
                      # POD 基与均值（还原 Cp 必需）
                      "pod_Phi": Phi.tolist(), "pod_mu": mu.tolist(),
                      "pod_r": r, "wallcp_npz": wallcp_npz,
                      "tr_eff_names": list(tr_eff_names),
                  }},
                 best_vl, ep)
    model.load_state_dict(best_state, strict=False)
    model.eval()
    # 7) 统一评估口径（修复 3）：
    #    截断误差 = 同测试集、同公式的逐样本 relL2（之前是全局能量比 sqrt(Σ_{k≥r}S²/ΣS²)，
    #    和 cp_relL2 不是一个量纲，ratio 不成立，已废弃）。
    #    对照基线 3 个：均值场(α≡0) / rank-r POD 截断 / ridge 回归（train_eff 拟合）。
    with torch.no_grad():
        pred_te = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
    cp_pred = cp_from_alpha(pred_te, Phi.astype(np.float64), mu.astype(np.float64))
    cp_true = pod["CP_all"][cp_test_idx].astype(np.float64)

    def per_sample_cp_relL2(cp_a, cp_b):
        """逐样本 relL2 取均值（分子=‖cp_a-cp_b‖₂逐样本，分母=‖cp_b‖₂逐样本），同 test 集。"""
        d = np.linalg.norm(cp_a - cp_b, axis=1)
        base = np.linalg.norm(cp_b, axis=1)
        return float((d / np.maximum(base, 1e-12)).mean())

    # (a) 代理模型本身
    cp_relL2 = per_sample_cp_relL2(cp_pred, cp_true)
    # (b) 均值场基线：α≡0 -> Cp=mu
    cp_meanfield = np.tile(mu.astype(np.float64), (len(cp_true), 1))
    meanfield_relL2 = per_sample_cp_relL2(cp_meanfield, cp_true)
    # (c) rank-r POD 截断基线（oracle 投影：真值 POD 坐标截到 r 维再投影回 Cp）
    cp_trunc = cp_from_alpha(al_te.astype(np.float64), Phi.astype(np.float64), mu.astype(np.float64))
    trunc_relL2 = per_sample_cp_relL2(cp_trunc, cp_true)
    # (d) ridge 回归基线（train_eff 拟合 POD 坐标，同 133 维输入）
    from sklearn.linear_model import Ridge
    idx_te = cp_test_idx
    alpha_all_te = ((pod["CP_all"] - mu) @ Phi)[idx_te].astype(np.float64)
    idx_tr = [pod["names_all"].index(n) for n in tr_eff_names]
    alpha_all_tr = ((pod["CP_all"] - mu) @ Phi)[idx_tr].astype(np.float64)
    ridge = Ridge(alpha=1.0).fit(Xtr.astype(np.float64), alpha_all_tr)
    cp_ridge = cp_from_alpha(ridge.predict(Xte.astype(np.float64)),
                             Phi.astype(np.float64), mu.astype(np.float64))
    ridge_relL2 = per_sample_cp_relL2(cp_ridge, cp_true)

    # 8) 口径无关排序指标（min(Cp) 前缘压强谷值，和升力排序同向）
    from scipy.stats import spearmanr
    def _cp_metric(cp_arr):
        return np.minimum.reduce(cp_arr, axis=1)
    cp_spear = float(spearmanr(_cp_metric(cp_pred), _cp_metric(cp_true)).correlation)

    # 9) 官方口径力系数（只报 C_L）
    #    ⚠ 判定实验输出的 C_L_pressure_relerr_test 是"用**真值** Cp 走压力项公式"
    #      与官方 C_L 的残差（= 公式本身的可信度），**不是模型误差**；
    #      直接当模型指标报会看起来像"B 的 C_L 误差 1.38%，碾压官方 0.517"。
    #      这里必须用**预测的** cp_pred 经同一公式算 B 自己的 C_L。
    import json as _json
    judge_path = os.path.join(ROOT, "04_results", "prediction",
                              "judge_lifting_full.json")
    judge = {}
    if os.path.exists(judge_path):
        judge = _json.load(open(judge_path, encoding="utf-8"))
    force_fields = {}
    official_cl = None
    if official_coefs and os.path.exists(official_coefs):
        _oc = np.load(official_coefs, allow_pickle=True)
        _onames = _oc["names"].tolist()
        official_cl = np.array([_oc["CL"][_onames.index(n)] for n in test_names],
                               dtype=np.float64)
    if judge.get("branch") == "a" and official_cl is not None:
        from official_metrics import cp_to_pressure_CL
        pred_cl = np.array([cp_to_pressure_CL(cp_pred[i], test_names[i], tr_root)
                            for i in range(len(test_names))], dtype=np.float64)
        _den = np.maximum(np.abs(official_cl), 1e-9)
        force_fields = {
            "C_L_relerr_pred_test": float((np.abs(pred_cl - official_cl) / _den).mean()),
            "C_L_pred_mean": float(pred_cl.mean()),
            "C_L_official_mean": float(official_cl.mean()),
            "C_L_spearman_pred": float(spearmanr(pred_cl, official_cl).correlation),
            "C_L_formula_fidelity_trueCp": judge.get("mean_pressure_vs_official_relerr"),
            "note": ("C_L_relerr_pred_test = 由**预测的** Cp 曲线做压力项线积分"
                     "（真实法向+真实弧长，缺壁面剪应力）与官方 Compute_coefficients "
                     "真值的逐样本相对误差。缺的摩擦项实测只占 C_L 的 0.02%，"
                     "但占 C_D 的 58%，故 C_D 不报。"
                     "C_L_formula_fidelity_trueCp 是把**真值** Cp 走同一公式的残差，"
                     "用来标定公式可信度，不是模型误差，两者不可混用。"),
        }
    elif judge.get("branch") == "b":
        force_fields = {"note": "branch b: C_L/C_D fields dropped (pressure-only formula not "
                                 "validated as lift-dominant); report calibration-free metrics above"}
    else:
        force_fields = {"note": "judgment not yet run or official coefs cache missing; "
                                 "C_L fields pending (judge_lifting_full.json + official_coefs_1000.npz)"}

    metrics = {
        "task": task, "chain": "B_official", "pod_r": r, "n_epochs": n_epochs,
        "hidden": hidden, "dim_in": dim_in,
        "train_n_eff": int(len(Xtr)), "val_n": int(len(val_names)),
        "test_n": int(len(Xte)),
        # 主指标：逐样本 relL2（统一口径，修复 3）
        "cp_relL2_test": cp_relL2,
        "cp_relL2_meanfield_test": meanfield_relL2,
        "cp_relL2_rank8_trunc_test": trunc_relL2,
        "cp_relL2_ridge_test": ridge_relL2,
        "ratio_agent_vs_trunc": (cp_relL2 / trunc_relL2 if trunc_relL2 > 0 else None),
        "ratio_agent_vs_meanfield": (cp_relL2 / meanfield_relL2 if meanfield_relL2 > 0 else None),
        "cp_spearman_min_cp_test": cp_spear,  # 口径无关排序指标
        **force_fields,
        "final_val_alpha_loss": best_vl,
        "loss_csv": "loss_history.csv", "best_model": "best_model.pt",
    }
    rd.save_eval(metrics)
    print(f"[{task}] B_official done: cp_relL2={cp_relL2:.4f} trunc={trunc_relL2:.4f} "
          f"meanfield={meanfield_relL2:.4f} ridge={ridge_relL2:.4f} "
          f"spear={cp_spear:.4f} branch={judge.get('branch','pending')}")
    return model, metrics, rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--wallcp-npz", default=None,
                    help="壁面 Cp 全量 npz（工作站 /home/user/airfrans/pod/wallcp_1000.npz）")
    ap.add_argument("--official-coefs", default=None,
                    help="官方真值 C_L/C_D 缓存 npz（official_coefs_1000.npz）；"
                         "缺省时与 --wallcp-npz 同目录")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--task-root", default=None,
                    help="Dataset 根目录（工作站 /home/user/airfrans/data/Dataset）")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    man_path = args.manifest or os.path.normpath(
        os.path.join(_HERE, "..", "..", "02_dataset", "Dataset", "manifest.json"))
    man = json.load(open(man_path))
    tr_names = man[args.task + "_train"]
    te_names = man["full_test"] if args.task == "scarce" else man[args.task + "_test"]
    print(f"[{args.task}] train={len(tr_names)} test={len(te_names)}")

    if args.smoke_only:
        # 冒烟：只验证几何读取 + 5 标量 + 一次 forward 不 NaN/OOM
        sub_tr = tr_names[:16]; sub_te = te_names[:16]
        scal_tr = np.stack([input_vector_5scalar(n) for n in sub_tr])
        scal_te = np.stack([input_vector_5scalar(n) for n in sub_te])
        # standardize 接收名字列表，直接传 sub_tr/sub_te
        sm = standardize(sub_tr, sub_te)
        geo = load_all_geometric(sub_tr, args.task_root or TASK_ROOT_DEFAULT, 64)
        gmean, gstd = geo.mean(0, keepdims=True), geo.std(0, keepdims=True)
        gstd[gstd == 0] = 1.0
        X = np.concatenate([sm["Xtr"], ((geo - gmean) / gstd).astype(np.float32)], 1)
        m = BBackbone(dim_in=X.shape[1], dim_out=POD_R).to("cpu")
        out = m(torch.from_numpy(X))
        ok = bool(torch.isfinite(out).all())
        print(f"[smoke] B_official forward OK, out={tuple(out.shape)}, finite={ok}")
        sys.exit(0 if ok else 1)

    train_official(args.task, tr_names, te_names, n_epochs=args.epochs,
                   device=args.device, lr=args.lr, patience=args.patience,
                   init_from=args.init_from, wallcp_npz=args.wallcp_npz,
                   hidden=args.hidden, task_root=args.task_root,
                   official_coefs=args.official_coefs or (
                       os.path.join(os.path.dirname(args.wallcp_npz), "official_coefs_1000.npz")
                       if args.wallcp_npz else None),
                   seed=args.seed)


if __name__ == "__main__":
    main()

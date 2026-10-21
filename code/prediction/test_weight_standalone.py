"""验证：只用 best_model.pt 一个文件，能否独立完成预测（不需要重训、不需要训练集）。

这是判断"权重是否可用"的决定性测试：从权重文件读出 model_state + norm_stats，
对新样本做归一化 -> 推理 -> 还原 Cp，然后与 eval_metrics.json 里记录的精度对比。
"""
import os, sys, json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from train_B_transformer import BBackbone, cp_from_alpha, POD_R  # noqa: E402
from geometry_input import load_all_geometric, input_vector_5scalar  # noqa: E402
from runkit import RunDir  # noqa: E402

TASK_ROOT = "/home/user/airfrans/data/Dataset"


def load_B_from_weight(task, device="cpu"):
    """只依赖 best_model.pt，重建可预测的模型（含归一化 + POD 基）。"""
    rd = RunDir(task, "B_official")
    st = torch.load(os.path.join(rd.dir, "best_model.pt"), map_location=device)
    ns = st["norm_stats"]
    m = BBackbone(dim_in=133, dim_out=ns["pod_r"], hidden=256, n_layers=4).to(device)
    m.load_state_dict(st["model_state"])
    m.eval()
    return {
        "model": m,
        "smean": np.array(ns["smean"]), "sstd": np.array(ns["sstd"]),
        "gmean": np.array(ns["gmean"]), "gstd": np.array(ns["gstd"]),
        "Phi": np.array(ns["pod_Phi"]), "mu": np.array(ns["pod_mu"]),
        "device": device,
    }


def predict_B(bundle, names, task_root=TASK_ROOT):
    """对样本名列表预测 Cp 曲线（256 点），只用权重里存的统计量。"""
    scal = np.stack([input_vector_5scalar(n) for n in names])
    geo = load_all_geometric(names, task_root)
    X = np.concatenate([(scal - bundle["smean"]) / bundle["sstd"],
                        (geo - bundle["gmean"]) / bundle["gstd"]],
                       axis=1).astype(np.float32)
    with torch.no_grad():
        alpha = bundle["model"](torch.from_numpy(X).to(bundle["device"])).cpu().numpy()
    return cp_from_alpha(alpha, bundle["Phi"], bundle["mu"])


if __name__ == "__main__":
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    for task in ["full", "scarce", "reynolds", "aoa"]:
        te = man["full_test"] if task == "scarce" else man[task + "_test"]
        b = load_B_from_weight(task)
        cp_pred = predict_B(b, te)
        # 真值：从 wallcp npz 取
        pod = np.load(b.get("npz", "/home/user/airfrans/pod/wallcp_1000.npz"),
                      allow_pickle=True) if False else None
        cp_all = np.load("/home/user/airfrans/pod/wallcp_1000.npz",
                         allow_pickle=True)
        names_all = cp_all["names"].tolist()
        CP = cp_all["CP"]
        idx = [names_all.index(n) for n in te]
        cp_true = CP[idx].astype(np.float64)
        d = np.linalg.norm(cp_pred - cp_true, axis=1)
        base = np.linalg.norm(cp_true, axis=1)
        relL2 = float((d / np.maximum(base, 1e-12)).mean())
        meta = json.load(open(os.path.join(
            RunDir(task, "B_official").dir, "eval_metrics.json"), encoding="utf-8"))
        rec = meta["cp_relL2_test"]
        print("%-9s 权重独立复现 relL2=%.6f | 训练记录=%.6f | 差=%.2e"
              % (task, relL2, rec, abs(relL2 - rec)))
    print("STANDALONE_WEIGHT_TEST_DONE")

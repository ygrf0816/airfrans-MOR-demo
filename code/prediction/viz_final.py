"""可视化：每个任务随机抽 6 个测试样本，只读这 6 个文件，用已训好的权重推理，
把 A 链路、B 链路、真值的全部输出拉出来对比。

A 链路（官方 GraphSAGE，逐点 4 场量 u/v/p/nu_t）：
  - 输入 = 单样本重建的 7 维 [x,y,Ux,Uy,sdf,nx,ny]，用训练时同一套 coef_norm 归一化
  - 推理 = 官方 Infer_test 口径（反复采 32k 直到全覆盖，重叠取平均）
  - 输出 = 去归一化后的 4 场量，和真值逐点对比 + 云图

B 链路（POD 降维，壁面 Cp 曲线）：
  - 输入 = 文件名 5 标量 + 128 维几何曲线（用训练时的 smean/sstd/gmean/gstd 归一化）
  - 推理 = MLP -> 8 维 POD 坐标 -> 还原 256 点 Cp 曲线（前缘锚定）-> 换算 x/c
  - 输出 = Cp 曲线，和真值表面 Cp 对比

归一化统计量：训练时没落盘，这里按训练代码的同一口径重算（几何 800 个 .vtp 仅 3 秒）。

输出（本地 04_results/prediction/figs/）：
  {task}_cp_6samples.png      Cp 曲线：真值/A/B
  {task}_field_6samples.png   4 场量云图：真值 vs A 预测（每样本一行）
  {task}_field_err.png        4 场量误差云图 |pred-true|
"""
import os, sys, json, random, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from runkit import RunDir  # noqa: E402
from train_A_gnn import OfficialGraphSAGE, build_radius_graph, SUBSAMP  # noqa: E402
from train_B_transformer import BBackbone, load_pod_basis, cp_from_alpha, POD_R  # noqa: E402
from geometry_input import load_all_geometric, input_vector_5scalar  # noqa: E402
from single_sample import load_sample, decode_sim, TASK_ROOT  # noqa: E402

FIGDIR = os.path.join(_HERE, "04_results", "prediction", "figs")
os.makedirs(FIGDIR, exist_ok=True)

# 中文字体：工作站无中文字体时会画成方框，这里只用英文标签
plt_zh = False
WALLCP_NPZ = "/home/user/airfrans/pod/wallcp_1000.npz"
TASKS = ["full", "scarce", "reynolds", "aoa"]
N_SAMP = 6
SEED = 20261004

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_names(task, n=N_SAMP):
    """从该任务的测试集随机抽 n 个样本名。
    官方 split：scarce 无独立测试集，沿用 full_test（见 manifest.json）。
    """
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    test_key = "full_test" if task == "scarce" else task + "_test"
    rng = random.Random(SEED + len(task) * 131)
    return rng.sample(man[test_key], n)


# ---------- 归一化（训练口径，重算并缓存） ----------

def a_coef_norm(task):
    """A 链路 coef_norm：直接从 eval_metrics.json 读（训练时已落盘，可靠）。"""
    meta = json.load(open(os.path.join(RunDir(task, "A").dir, "eval_metrics.json"),
                          encoding="utf-8"))
    c = meta["coef_norm"]
    return (np.asarray(c["mean_in"], np.float32), np.asarray(c["std_in"], np.float32) + 1e-8,
            np.asarray(c["mean_out"], np.float32), np.asarray(c["std_out"], np.float32) + 1e-8)


def b_stats(task):
    """B 链路归一化统计量：训练时没落盘，按训练代码同一口径重算。
    train_eff = task_train 去掉末尾 10%（val）；标量用 train_eff、几何用 train_eff。
    """
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


# ---------- A 链路推理 ----------

_A_CACHE = {}


def a_predict(sample, task, device="cuda:1"):
    """A 链路单样本推理 -> 4 场量（去归一化，物理量）。
    返回 (pred (N,4), true (N,4))，列序 u,v,p,nu_t。
    """
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
    pred = pred * so + mo          # 去归一化 -> 物理量
    return pred, sample["y"].astype(np.float64)


# ---------- B 链路推理 ----------

_B_CACHE = {}


def b_predict(sample, task, device="cuda:1"):
    """B 链路单样本推理 -> 256 点 Cp 曲线（POD 还原，前缘锚定）。
    返回 (cp256, 真值表面 Cp 同口径) 供曲线对比。
    """
    if task not in _B_CACHE:
        man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
        pod = load_pod_basis(man[task + "_train"], wallcp_npz=WALLCP_NPZ)
        rd = RunDir(task, "B_official")
        m = BBackbone(dim_in=133, dim_out=POD_R, hidden=256, n_layers=4).to(device)
        st = torch.load(os.path.join(rd.dir, "best_model.pt"), map_location=device)
        m.load_state_dict(st["model_state"])
        m.eval()
        _B_CACHE[task] = (pod, m, b_stats(task))
    pod, m, (smean, sstd, gmean, gstd) = _B_CACHE[task]

    nm = sample["name"]
    x1_ = ((np.asarray(input_vector_5scalar(nm)) - smean.ravel()) / sstd.ravel()
           ).ravel().astype(np.float32)
    x2_ = ((load_all_geometric([nm], TASK_ROOT)[0] - gmean.ravel()) / gstd.ravel()
           ).astype(np.float32)
    X = np.concatenate([x1_, x2_])
    with torch.no_grad():
        alpha = m(torch.from_numpy(X[None]).to(device)).cpu().numpy()
    cp = cp_from_alpha(alpha, pod["Phi"].astype(np.float64),
                       pod["mu"].astype(np.float64))[0]
    # 真值同口径：wallcp_1000 里同一样本的曲线
    names = pod["names_all"]
    cp_true = pod["CP_all"][names.index(nm)].astype(np.float64)
    return cp, cp_true


# ---------- 画图 ----------

def fig_cp(task, samples, device):
    """Cp 曲线：真值 Cp(x) vs A 预测（表面点 p）vs B 预测（POD 256 点）。"""
    import single_sample as ss
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, s in enumerate(samples):
        ax = axes[i // 3][i % 3]
        nm = s["name"]
        S = s["surf"]
        P = s["pos"][S]
        # 数据集的 p 列是 p/ρ（量纲速度²），Cp = p / (0.5 U∞²)。
        # 核对：该样本 p_max=3058.8，q=0.5*78.16²=3054.8，p/q 落在 [-1,1]。
        q = 0.5 * s["par"]["U_inf"] ** 2
        cp_true = s["y"][S, 2] / q
        pred, _ = a_predict(s, task, device)
        cp_a = pred[S, 2] / q
        cp_b, cp_b_true = b_predict(s, task, device)   # B：256 点弧长序（Cp 口径）

        # 真值/A 按弦向 x 的上下表面；B 按弧长序（前半上表面）
        order_t = np.argsort(P[:, 0])
        n = len(cp_b); half = n // 2
        ax.plot(P[order_t, 0], cp_true[order_t], "k.", ms=2, label="true (CFD)")
        ax.plot(P[order_t, 0], cp_a[order_t], "r.", ms=2, alpha=0.7, label="A-chain pred")
        s_b = np.linspace(0, 1, n)
        ax.plot(s_b, cp_b, "b-", lw=1.2, alpha=0.9, label="B-chain pred (POD 256)")
        ax.plot(s_b, cp_b_true, "k--", lw=1, alpha=0.6, label="true (POD-aligned)")
        ax.set_title(nm, fontsize=8)
        ax.set_xlabel("x/c (A/true)  |  s (B)"); ax.set_ylabel("Cp")
        ax.grid(alpha=0.3)
        ax.set_ylim(min(-2.0, np.nanmin(cp_true) - 0.5), max(1.5, np.nanmax(cp_true) + 0.5))
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"task={task}  Cp curve: true / A-chain / B-chain(POD)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"{task}_cp_6samples.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[fig] {p}", flush=True)


def fig_field(task, samples, device, field_idx=(0, 1, 2, 3), names=("u", "v", "p", "nu_t")):
    """4 场量云图：每个样本一行，列为 真值 / A 预测 / 误差。"""
    ns = len(samples)
    nf = len(field_idx)
    fig, axes = plt.subplots(ns, nf * 3, figsize=(4.2 * nf * 3, 3.0 * ns))
    if ns == 1:
        axes = axes[None, :]
    for r, s in enumerate(samples):
        pred, true = a_predict(s, task, device)
        P = s["pos"]
        for c, (fi, fn) in enumerate(zip(field_idx, names)):
            t = true[:, fi]; pr = pred[:, fi]
            err = np.abs(pr - t)
            for k, (vals, lab, cmap) in enumerate([
                    (t, "true", "viridis"), (pr, "A pred", "viridis"),
                    (err, "|err|", "hot")]):
                ax = axes[r, c * 3 + k]
                sc = ax.scatter(P[:, 0], P[:, 1], c=vals, s=0.4,
                                cmap=cmap, rasterized=True)
                plt.colorbar(sc, ax=ax, fraction=0.046)
                if r == 0:
                    ax.set_title(f"{fn} . {lab}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(s["name"][:22], fontsize=6)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_aspect("equal")
    fig.suptitle(f"task={task}  4-field contour: true / A-pred / |err|", fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"{task}_field_6samples.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[fig] {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--nsamp", type=int, default=N_SAMP)
    ap.add_argument("--load", type=str, default=None,
                    help="只跑指定 task（逗号分隔）")
    args = ap.parse_args()

    for task in args.tasks:
        names = pick_names(task, args.nsamp)
        print(f"[viz] {task}: {names}", flush=True)
        samples = [load_sample(nm) for nm in names]     # 只读这 6 个
        print(f"[viz] {task}: loaded {len(samples)} samples", flush=True)
        fig_cp(task, samples, args.device)
        fig_field(task, samples, args.device)
    print("[viz] ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

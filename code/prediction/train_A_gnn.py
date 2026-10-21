"""A 链路（官方口径对齐 AirfRANS）：GraphSAGE -> 逐点流场。

官方口径（可回溯 01_initial/AirfRANS-main/ + params.yaml + airfrans pip 库）：
  数据：pip airfrans.dataset.load(root, task, train) -> (list[ndarray(N,12)], list[name])
        12 列 = [pos(2), inlet_vel(2), sdf(1), normals(2), velocity(2), pressure(1),
                 nu_t(1), surface(1)]
        -> 输入 7 维 = 前 7 列 [x,y,Ux,Uy,sdf,nx,ny]（官方文档逐点 7 特征）
        -> 目标 4 维 = 中间 4 列 [u,v,p,nu_t]；surface bool = 第 12 列
  归一化：官方 dataset.py norm=True + main.py coef_norm，train 部分算 mean/std，
        val/test 复用；不归一化时 p 会单独主导 MSE
  图：radius graph（r=0.05 m, max_num_neighbors=64, loop=True）官方
      nng.radius_graph(pos, r, loop=True, max_num_neighbors)
  骨干：官方 GraphSAGE.py = encoder MLP[7,64,64,8] + SAGEConv×3(64)+BN
        + decoder MLP[8,64,64,4]（官方 params.yaml，参数量官方 ~29204）
  采样：每 epoch 随机采 SUBSAMP=32000 节点 + 重建 radius graph（官方 subsampling）
  loss：volume + λ·surface（官方 MSE_weighted，λ=1）
  官方 nb_epochs=398, lr=1e-3
  4 任务：full/scarce/reynolds/aoa（manifest.json 官方 split，val=train 末尾 10%）
止损点：单 epoch > 2h 仍不降 / 反复 OOM -> 降采样或 SGC 轻量 GCN。
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as nng

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from runkit import RunDir, is_nan  # noqa: E402

# ---- 官方 params.yaml GraphSAGE 口径 ----
ENC = [7, 64, 64, 8]
DEC = [8, 64, 64, 4]
HIDDEN = 64
NB_HIDDEN = 3
RADIUS = 0.05
MAX_NEIGHBORS = 64
SUBSAMP = 32000
LAMBDA_SURF = 1.0
EPOCHS = 398
SEED = 0                  # 默认 seed；多 seed 对照用 --seed 覆盖（官方跑 5 次报 mean±std）
VAL_ITER = 10

# pip airfrans.load 的 12 列布局（官方文档核实）
COL_XY = slice(0, 2)
COL_INLET = slice(2, 4)
COL_SDF = 4
COL_NORMAL = slice(5, 7)
COL_TARGET = slice(7, 11)   # u, v, p, nu_t
COL_SURF = 11


def _radius_graph(pos2d, r, loop, max_num_neighbors):
    """官方半径图（r=0.05, max_neighbors=64, loop=True）的等价入口。

    PyG 2.8 把 nng.radius_graph 收窄为必须 pyg-lib>=0.6.0，而本机 torch 2.6.0+cu124
    在 PyG 官方 wheel 索引上最高只有 pyg-lib 0.5.0（装了也被版本门禁拒绝）。
    旧版 PyG 在该场景本来就是回落到 torch_cluster，这里直接用同一实现，语义一致；
    torch_cluster 缺失时才退回 nng.radius_graph。
    """
    try:
        from torch_cluster import radius_graph
        return radius_graph(pos2d, r=r, loop=loop,
                            max_num_neighbors=max_num_neighbors)
    except ImportError:
        return nng.radius_graph(pos2d, r=r, loop=loop,
                                max_num_neighbors=max_num_neighbors)


def build_radius_graph(pos2d, device="cpu"):
    """pos2d (N,2) numpy/tensor -> 官方 radius graph edge_index (2,E)。

    官方 train.py:167 是在 GPU 上建图（pos.to(device)）再 .cpu() 存回；
    实测真实 CFD 网格（近壁面网格极密，平均度 ~52）：
      CPU 1291 ms/次  vs  GPU 6.9 ms/次（~190 倍）。
    必须建在 device 上，否则 398 epoch x 720 样本按 CPU 建图要 100+ 小时/任务。
    """
    if isinstance(pos2d, np.ndarray):
        pos2d = torch.from_numpy(np.ascontiguousarray(pos2d)).to(torch.float)
    else:
        pos2d = pos2d.to(torch.float)
    return _radius_graph(pos2d.to(device), RADIUS, True, MAX_NEIGHBORS)


def load_task(task_root, task, train=True, limit=None):
    """pip airfrans.load -> (list[Data], list[name])。
    每个 Data: x (N,7) 输入, y (N,4) 目标, pos (N,2), surf (N,) bool。
    limit：截断样本数（冒烟用；注意 pip load 仍会读完整个 split，再截断）。
    """
    import airfrans
    out = airfrans.dataset.load(task_root, task, train=train)
    arr, names = out
    if limit is not None:
        arr, names = arr[:limit], names[:limit]
    data_list = []
    for a in arr:
        X = a[:, 0:7].astype(np.float32)         # 7 维输入
        Y = a[:, COL_TARGET].astype(np.float32)  # 4 维目标（列切片，勿写成 a[COL_TARGET]）
        P = a[:, 0:2].astype(np.float32)         # 2D 位置
        S = a[:, COL_SURF].astype(bool)          # surface bool
        d = type('D', (), {'x': None, 'y': None, 'pos': None, 'surf': None})()
        d.x = torch.from_numpy(X); d.y = torch.from_numpy(Y)
        d.pos = torch.from_numpy(P); d.surf = torch.from_numpy(S)
        data_list.append(d)
    return data_list, list(names)


def compute_coef_norm(data_list):
    """官方 dataset.py norm=True 口径：在 train 部分的全部点上统计 x / y 的 mean、std。

    官方 main.py：train_dataset, coef_norm = Dataset(train_dataset, norm = True)，
    val/test 复用同一套 coef_norm；归一化式 (x - mean) / (std + 1e-8)。
    不归一化时 p（p/ρ）比 nu_t 大 5~6 个数量级，MSE 会被 p 单独主导。
    """
    n = 0
    sx = np.zeros(7, dtype=np.float64); sxx = np.zeros(7, dtype=np.float64)
    sy = np.zeros(4, dtype=np.float64); syy = np.zeros(4, dtype=np.float64)
    for d in data_list:
        x = d.x.numpy().astype(np.float64)
        y = d.y.numpy().astype(np.float64)
        sx += x.sum(axis=0); sxx += (x * x).sum(axis=0)
        sy += y.sum(axis=0); syy += (y * y).sum(axis=0)
        n += x.shape[0]
    mean_in = (sx / n).astype(np.float32)
    mean_out = (sy / n).astype(np.float32)
    std_in = np.sqrt(np.maximum(sxx / n - (sx / n) ** 2, 0.0)).astype(np.float32)
    std_out = np.sqrt(np.maximum(syy / n - (sy / n) ** 2, 0.0)).astype(np.float32)
    return {"mean_in": mean_in, "std_in": std_in,
            "mean_out": mean_out, "std_out": std_out}


def apply_coef_norm(data_list, coef):
    """按官方 coef_norm 就地归一化 x / y（val、test 用 train 的统计量）。"""
    mi = torch.from_numpy(coef["mean_in"]); si = torch.from_numpy(coef["std_in"] + 1e-8)
    mo = torch.from_numpy(coef["mean_out"]); so = torch.from_numpy(coef["std_out"] + 1e-8)
    for d in data_list:
        d.x = (d.x - mi) / si
        d.y = (d.y - mo) / so


class OfficialMLP(nn.Module):
    """官方 models/MLP.py（encoder/decoder，channel_list + ReLU）。

    官方 main.py:51-52 构造时传 `batch_norm = False`，即 encoder/decoder **不带 BN**
    （norms 退化成 Identity）。之前默认带 BatchNorm1d 会让参数量变成 29716，
    而官方 GraphSAGE 是 29204（差 512 = 4 个 BN 层的 weight+bias）；
    且 BN 在 eval 下用当前 batch 统计量（track_running_stats=False），
    会让按子采样分批推理的预测随批次抖动。
    """
    def __init__(self, channels, batch_norm=False):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(a, b) for a, b in
                                   zip(channels[:-1], channels[1:])])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(d, track_running_stats=False) if batch_norm else nn.Identity()
            for d in channels[1:-1]])

    def forward(self, x):
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            x = norm(x); x = x.relu_(); x = lin(x)
        return x


class OfficialGraphSAGE(nn.Module):
    """官方 models/GraphSAGE.py：encoder -> SAGE×3 + BN -> decoder。"""
    def __init__(self):
        super().__init__()
        self.encoder = OfficialMLP(ENC)
        self.decoder = OfficialMLP(DEC)
        self.in_layer = nng.SAGEConv(ENC[-1], HIDDEN)
        self.hidden_layers = nn.ModuleList([
            nng.SAGEConv(HIDDEN, HIDDEN) for _ in range(NB_HIDDEN - 1)])
        self.out_layer = nng.SAGEConv(HIDDEN, DEC[0])
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(HIDDEN, track_running_stats=False)
            for _ in range(NB_HIDDEN)])
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        z = self.encoder(x)
        z = self.act(self.bn[0](self.in_layer(z, edge_index)))
        for n in range(NB_HIDDEN - 1):
            z = self.act(self.bn[n + 1](self.hidden_layers[n](z, edge_index)))
        z = self.out_layer(z, edge_index)
        return self.decoder(z)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def _subsample(d, sub=SUBSAMP, seed=None, device="cpu"):
    """官方每 epoch subsampling：随机采 sub 节点 + 重建 radius graph。
    返回 (x, y, surf, edge_index) 张量。"""
    N = d.x.size(0)
    rng = random.Random(seed)
    idx = rng.sample(range(N), min(sub, N))
    idx = torch.tensor(idx, dtype=torch.long)
    x = d.x[idx]; y = d.y[idx]; pos = d.pos[idx]; surf = d.surf[idx]
    e = build_radius_graph(pos, device)
    return x, y, surf, e


def train_gnn(task, task_root, device="cuda:0", n_epochs=EPOCHS, lr=1e-3,
              limit=None, init_from=None, sub=SUBSAMP, patience=15, shuffle=True,
              seed=SEED):
    rd = RunDir(task, "A_s%d" % seed if seed != SEED else "A")
    # 可复现：同 seed 两次跑结果一致；不同 seed 之间 cp/loss 可差 ~10%（官方口径）
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"[{task}] A: load train (limit={limit})...", flush=True)
    data_tr, names_tr = load_task(task_root, task, train=True, limit=limit)
    data_te, names_te = load_task(task_root, task, train=False, limit=limit)
    # 官方 main.py：val = train 末尾 10%（limit 时至少保 1 个 val）
    n_val = max(1, int(0.1 * len(data_tr)))
    train_set = data_tr[:max(1, len(data_tr) - n_val)]
    val_set = data_tr[max(1, len(data_tr) - n_val):]
    # 官方归一化：coef_norm 在 train 部分上算，val/test 复用同一套（dataset.py norm=True）
    coef = compute_coef_norm(train_set)
    for _s in (train_set, val_set, data_te):
        apply_coef_norm(_s, coef)
    print(f"[{task}] A: train={len(train_set)} val={len(val_set)} "
          f"test={len(data_te)} (official coef_norm applied)", flush=True)

    model = OfficialGraphSAGE().to(device)
    print(f"[{task}] A: GraphSAGE params={model.count_params()}", flush=True)
    if init_from:
        model.load_state_dict(
            torch.load(init_from, map_location=device)["model_state"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # ⚠ 官方 train.py:135 是 torch.optim.Adam(..., lr=...) —— **没有 weight_decay**。
    # 之前用 AdamW 的默认 weight_decay=0.01 是解耦衰减 w *= (1 - lr*wd)：
    # full 任务 720 样本 x 398 epoch = 286560 步 -> exp(-2.87) ≈ 0.057，
    # 整个训练把权重压掉 94%，模型只能欠拟合（train/val 双双卡在 ~0.31 不降）。
    # 实测 A 测试集归一化 MSE 比官方 Table 2 差 25~37 倍，根因就在这里。
    # 官方 train.py:136-140：OneCycleLR(max_lr=lr, total_steps=(N/batch+1)*nb_epochs)，
    # 每个 batch 调一次 scheduler.step()；官方是**训满 398 epoch、无早停**。
    # 实测原配置（常数 lr + patience=15）在 reynolds 上 35/398 就被单次子采样噪声
    # 触发的早停砍掉，而当时 train/val 两条曲线都还在下降
    #（full: train 0.91->0.53，val 0.61->0.34），属于欠训而非收敛。
    steps_per_epoch = max(1, len(train_set))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps_per_epoch * n_epochs)
    mseloss = nn.MSELoss(reduction="none")

    best_vl, best_state, bad = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        # 官方 train.py:181 用 DataLoader(..., shuffle=True)：每个 epoch 打乱样本顺序。
        # 不打乱等于固定循环的增量梯度，长期训练会带顺序偏置。
        if shuffle:
            random.shuffle(train_set)
        ep_sum, ep_n = 0.0, 0
        for di, d in enumerate(train_set):
            x, y, surf, e = _subsample(d, sub, seed=1000 * ep + di, device=device)
            x, y, surf, e = x.to(device), y.to(device), surf.to(device), e.to(device)
            opt.zero_grad()
            out = model(x, e)
            loss_c = mseloss(out, y)
            # 官方 train.py:55-56：(loss_vol + reg*loss_surf).backward()，无 /2 因子
            vol = loss_c[~surf].mean() if (~surf).any() else out.new_zeros(())
            sr = loss_c[surf].mean() if surf.any() else out.new_zeros(())
            loss = vol + LAMBDA_SURF * sr
            loss.backward()
            opt.step()
            sched.step()
            ep_sum += loss.item(); ep_n += 1
        # val：子采样一次
        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for di, d in enumerate(val_set):
                x, y, surf, e = _subsample(d, sub, seed=9999 + di, device=device)
                x, y, surf, e = x.to(device), y.to(device), surf.to(device), e.to(device)
                out = model(x, e)
                loss_c = mseloss(out, y)
                vol = loss_c[~surf].mean() if (~surf).any() else out.new_zeros(())
                sr = loss_c[surf].mean() if surf.any() else out.new_zeros(())
                vl_sum += float(vol + LAMBDA_SURF * sr); vl_n += 1
        vl = vl_sum / max(vl_n, 1)
        rd.append_loss(ep, ep_sum / max(ep_n, 1), vl,
                       extra={"kind": "sage_official", "r": RADIUS,
                              "max_neigh": MAX_NEIGHBORS, "sub": sub})
        if vl < best_vl - 1e-4:
            best_vl, bad = vl, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[{task}] A early stop @ep{ep} (best={best_vl:.4f})",
                      flush=True); break
        if is_nan(vl):
            print(f"[{task}] A NaN @ep{ep}"); break

    # 超参落盘（修复：之前没调，hyperparams.json 一直是空的）
    rd.save_hyperparams({
        "task": task, "chain": "A", "kind": "sage_official",
        "enc": ENC, "dec": DEC, "hidden": HIDDEN, "nb_hidden": NB_HIDDEN,
        "radius": RADIUS, "max_neighbors": MAX_NEIGHBORS,
        "subsample": sub, "lambda_surf": LAMBDA_SURF,
        "n_epochs": n_epochs, "lr": lr, "patience": patience,
        "shuffle": bool(shuffle), "seed": int(seed),
        "optimizer": "Adam (no weight_decay), OneCycleLR",
        "input": "7-dim point (x,y,Ux,Uy,sdf,nx,ny)",
        "output": "4-dim point (u,v,p,nu_t)",
        "train_samples": int(len(train_set)), "val_samples": int(len(val_set)),
        "test_samples": int(len(data_te)),
    })
    # 权重 + 归一化统计量一起存（修复：之前只存 model_state，加载后无法归一化）
    rd.save_best({"model_state": best_state, "task": task,
                  "coef_norm": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in coef.items()}},
                 best_vl, ep)
    model.load_state_dict(best_state, strict=False)
    print(f"[{task}] A: test-set eval (official Infer_test protocol)...", flush=True)
    test_m = evaluate_test(model, data_te, device=device, sub=sub, coef=coef)
    metrics = {
        "task": task, "chain": "A", "kind": "sage_official",
        "params": model.count_params(), "radius": RADIUS,
        "max_neighbors": MAX_NEIGHBORS, "hidden": HIDDEN,
        "n_epochs": n_epochs, "subsample": sub,
        "final_val_loss": best_vl,
        "normalization": "official coef_norm (train-part mean/std, dataset.py norm=True)",
        "coef_norm": {k: v.tolist() for k, v in coef.items()},
        "input": "7-dim point (x,y,Ux,Uy,sdf,nx,ny) official airfrans.load()",
        "output": "4-dim point (u,v,p,nu_t)",
        "loss": "volume + lambda*surface (lambda=1) official MSE_weighted, no /2 factor",
        "train_samples": int(len(train_set)), "val_samples": int(len(val_set)),
        "test_samples": int(len(data_te)),
        "loss_csv": "loss_history.csv", "best_model": "best_model.pt",
        **test_m,
    }
    rd.save_eval(metrics)
    print(f"[{task}] A official SAGE done: val={best_vl:.4f} | "
          f"test vol MSE(mean,x1e-2)={test_m['test_volume_MSE_mean_x1e-2']:.4f} "
          f"surf p MSE(x1e-1)={test_m['test_surface_p_MSE_x1e-1']:.4f}", flush=True)
    return model, metrics, rd


def evaluate_test(model, data_te, device="cuda:0", sub=SUBSAMP, max_rounds=200,
                  seed=20261004, coef=None):
    """测试集评估 —— 官方 metrics.Infer_test 口径（全覆盖 + 重叠取平均）。

    官方 Infer_test 是**反复随机采 32k 点直到覆盖全部点**，重叠处预测取平均，
    再对**归一化**场算 MSE（论文 Table 2 报的就是归一化 MSE）。
    这里逐字复刻该协议，产出可与官方 Table 2/3 直接并列的数字：
      - test_volume_MSE_x1e-2     : u / v / p / nu_t 四个体积场（×10⁻²，官方同量纲）
      - test_surface_p_MSE_x1e-1  : 壁面 p（×10⁻¹，官方同量纲）
      - test_relL2_per_field      : 去归一化后逐场相对 L2（EVAL_PROTOCOL 要的绝对量对照）
    注意用 d.pos（原始米制坐标，未被归一化）建 radius graph，半径 0.05 m 才有意义。
    """
    model.eval()
    VNAMES = ["u", "v", "p", "nu_t"]
    vol_acc = np.zeros(4); vol_n = 0
    srf_acc = 0.0; srf_n = 0
    rel_num = np.zeros(4); rel_den = np.zeros(4)
    mo = so = None
    if coef is not None:
        mo = torch.from_numpy(np.asarray(coef["mean_out"], dtype=np.float32))
        so = torch.from_numpy(np.asarray(coef["std_out"], dtype=np.float32))
    with torch.no_grad():
        for si, d in enumerate(data_te):
            N = d.x.size(0)
            pred = torch.zeros(N, 4); cnt = torch.zeros(N, 1)
            seen = np.zeros(N, dtype=bool)
            rng = random.Random(seed + si)
            for _ in range(max_rounds):
                if seen.all():
                    break
                idx = rng.sample(range(N), min(sub, N))
                it = torch.tensor(idx, dtype=torch.long)
                e = build_radius_graph(d.pos[it], device)
                out = model(d.x[it].to(device), e.to(device)).cpu()
                pred[it] += out
                cnt[it] += 1
                seen[idx] = True
            pred = pred / cnt.clamp(min=1)
            err = (pred - d.y) ** 2
            surf = d.surf.bool()
            if (~surf).any():
                vol_acc += err[~surf].sum(0).numpy()
                vol_n += int((~surf).sum())
            if surf.any():
                srf_acc += float(err[surf, 2].sum()); srf_n += int(surf.sum())
            if mo is not None:
                p_dn = pred * so + mo
                t_dn = d.y * so + mo
                rel_num += torch.linalg.norm(p_dn - t_dn, dim=0).numpy()
                rel_den += torch.linalg.norm(t_dn, dim=0).numpy()
    vn = max(vol_n, 1)
    out = {
        "test_volume_MSE_x1e-2": {VNAMES[i]: float(vol_acc[i] / vn * 100) for i in range(4)},
        "test_volume_MSE_mean_x1e-2": float(vol_acc.mean() / vn * 100),
        "test_surface_p_MSE_x1e-1": float(srf_acc / max(srf_n, 1) * 10),
        "test_n_points_volume": int(vol_n),
        "test_n_points_surface": int(srf_n),
        "test_eval_protocol": "official Infer_test: repeat random 32k subsample until full coverage, average overlaps",
    }
    if mo is not None:
        out["test_relL2_per_field"] = {
            VNAMES[i]: float(rel_num[i] / max(rel_den[i], 1e-12)) for i in range(4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                   choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sub", type=int, default=SUBSAMP)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--task-root", default="/home/zch/airfrans/data/Dataset")
    ap.add_argument("--limit", type=int, default=None, help="冒烟 100")
    ap.add_argument("--seed", type=int, default=SEED,
                   help="多 seed 对照（官方跑 5 次报 mean±std，默认 0）")
    ap.add_argument("--patience", type=int, default=15,
                    help="val 连续多少轮不降就早停；官方无早停，跑满 398 epoch 时设大值")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="官方 DataLoader(shuffle=True)，默认打乱；此项用于做不打乱的对照")
    args = ap.parse_args()
    train_gnn(args.task, args.task_root, device=args.device,
              n_epochs=args.epochs, lr=args.lr, limit=args.limit,
              init_from=args.init_from, sub=args.sub, seed=args.seed,
              patience=args.patience, shuffle=not args.no_shuffle)


if __name__ == "__main__":
    main()

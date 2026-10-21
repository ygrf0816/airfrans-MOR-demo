"""B 链路场预测（第 2/3 步）：来流+几何 -> 体积场 POD 坐标 -> 还原 4 个场。

背景（用户要求）：原 B 链路只还原壁面 Cp 曲线，**没有场预测**，无法与 A 链路
（全量原始数据逐点预测 u,v,p,nu_t）做全链路对标。本脚本补上这块：
每个官方任务都额外预测 u / v / p / nu_t 四个体积场，并给出与 A 同定义的对标数字。

做法：
  1. 从 vol_fields_1000.npz 取 256x128 公共网格上的真值场（交集掩膜，定长）
  2. 用**该任务官方 coef_norm 的 mean_out/std_out**（取自 A 的 eval_metrics.json）
     把场标准化 —— 于是重构误差就是"归一化场 MSE"，与 A 报的
     test_volume_MSE_x1e-2 是**同一个定义**
  3. 逐场做 POD（基只用 train_eff，val/test 复用同一基），目标 = 4 场 alpha 拼接
  4. MLP(133 -> 4r) 预测 alpha，Adam(lr=3e-4)，独立 val 早停，固定 seed
  5. 评估：逐场归一化 MSE、相对 L2、Spearman，并给出均值场 / rank-r 截断 / ridge 三个对照
     同时把 A 的对应数字并列写进 eval_metrics，便于直接对标

口径差异（必须如实写进报告）：A 的 MSE 在 ~18 万网格节点上算，B 在 256x128
公共网格的 ~1.4 万个有效格上算；两者定义相同、域不同（网格是域内均匀子采样）。

用法：
  python train_B_field.py --task full  --device cuda:0 \
     --cache /home/zch/airfrans/pod/vol_fields_1000.npz \
     --manifest /home/zch/airfrans/data/manifest.json \
     --task-root /home/zch/airfrans/data/Dataset --epochs 3000 --patience 300
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from geometry_input import load_all_geometric, input_vector_5scalar, N_STATION  # noqa: E402
from runkit import RunDir, is_nan  # noqa: E402

ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
FIELDS = ["u", "v", "p", "nu_t"]
POD_R = 32
SEED = 0


class FieldMLP(nn.Module):
    """(5 来流标量 + 128 几何 = 133) -> 4r 个 POD 坐标。与 B 的 Cp 链路同族骨干。"""

    def __init__(self, dim_in, dim_out, hidden=256, n_layers=4, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(dim_in, hidden), nn.LayerNorm(hidden), nn.GELU()]
        for i in range(n_layers):
            layers += [nn.Linear(hidden, hidden)]
            if i < n_layers - 1:
                layers += [nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        self.tower = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim_out))

    def forward(self, x):
        return self.head(self.tower(x))


def build_inputs(tr, val, te, task_root, geom_station=64):
    """5 标量 + 128 几何，统计量只用 tr 算，val/test 复用。

    同时返回统计量本身（smean/sstd/gmean/gstd），供落盘持久化 —— 没有它，
    权重文件拿到别处就无法对新样本做与训练一致的归一化。
    """
    def scal(ns):
        return np.stack([input_vector_5scalar(n) for n in ns]).astype(np.float64)
    s_tr, s_val, s_te = scal(tr), scal(val), scal(te)
    m, s = s_tr.mean(0, keepdims=True), s_tr.std(0, keepdims=True)
    s[s == 0] = 1.0
    g_tr = load_all_geometric(tr, task_root, geom_station)
    g_val = load_all_geometric(val, task_root, geom_station)
    g_te = load_all_geometric(te, task_root, geom_station)
    gm, gs = g_tr.mean(0, keepdims=True), g_tr.std(0, keepdims=True)
    gs[gs == 0] = 1.0
    cat = lambda a, b: np.concatenate([a, b], axis=1).astype(np.float32)
    stats = {"smean": m, "sstd": s, "gmean": gm, "gstd": gs,
             "geom_station": int(geom_station)}
    return (cat((s_tr - m) / s, (g_tr - gm) / gs),
            cat((s_val - m) / s, (g_val - gm) / gs),
            cat((s_te - m) / s, (g_te - gm) / gs),
            stats)


def mesh_domain_eval(rec, Xn_te, te_order, cp, cpb, args, am):
    """把 B 的重构插值回每个测试样本的 CFD 网格节点，用与 A **完全相同**的口径
    （官方 coef_norm 归一化场、只取体积点、per-field MSE）算误差。

    cpb is None（grid 表示）：整个域都用 256x128 笛卡尔网格插值。
    cpb 非 None（bodyfit 表示）：每个节点按它到翼型的距离 d 选区 ——
      d < d1 用 (s,d) 双线性回读贴体带；否则用远场笛卡尔区。

    同时用**真值目标**走同一条回读路径算出本表示的"下限"（floor）：B 的模型误差
    不可能低于它。没有这个数就无法判断误差是表示装不下还是模型没学好。

    口径提示：`airfrans.load` 第 4 列是**到壁的非负距离**（0 在壁面、向外增大），
    不是有符号 sdf；bodyfit.node_map 就是按这个口径写的，这里不要再取负号。

    返回 (pred_mse, 覆盖率, 参与节点数, 近壁带 pred_mse, 近壁带节点数,
          floor_mse, 近壁带 floor_mse)；grid 表示下两个近壁带量为 None。
    """
    import train_A_gnn as A
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    if cpb is not None:
        import bodyfit as BF
        ns = int(cpb["ns"]); nd = int(cpb["nd"])
        d0 = float(cpb["d0"]); d1 = float(cpb["d1"])
        idxb = {n: i for i, n in enumerate(cpb["names"].tolist())}
    else:
        ns = nd = 0; d0 = d1 = 0.0
        idxb = {}
    n_near = ns * nd
    pos = {n: i for i, n in enumerate(te_order)}

    # 远场用散点三角插值。**不要**图省事换成"把翼型内部的洞填上再用规则网格插值"：
    # 填洞会把壁面值平移到翼型内部，而近壁节点的插值模板正好会碰到洞里，
    # 实测让表示下限从 7.07 涨到 40.7（网格版 27.8 -> 106.4），并多出 4.3% 的 NaN。
    ny = int(cp["ny"]); nx = int(cp["nx"])
    xs = np.linspace(float(cp["x0"]), float(cp["x1"]), nx)
    ys = np.linspace(float(cp["y0"]), float(cp["y1"]), ny)
    GX, GY = np.meshgrid(xs, ys, indexing="xy")
    far_xy = np.column_stack([GX.ravel(), GY.ravel()])[cp["mask"]]
    far_tri = Delaunay(far_xy)

    def far_query(vals, q):
        return LinearNDInterpolator(far_tri, vals, fill_value=np.nan)(q)

    data_te, ld_names = A.load_task(args.task_root, args.task, train=False)
    if getattr(args, "max_eval_samples", 0):
        data_te, ld_names = data_te[:args.max_eval_samples], ld_names[:args.max_eval_samples]
    bad_order = [n for n in ld_names if n not in pos]
    if bad_order:
        raise SystemExit("load_task 返回了 manifest 之外的样本：%s" % bad_order[:3])
    coef = {k: np.asarray(v, dtype=np.float32) for k, v in am["coef_norm"].items()}
    A.apply_coef_norm(data_te, coef)
    mi = coef["mean_in"]; si = coef["std_in"]

    acc = np.zeros(4); acc_near = np.zeros(4); acc_floor = np.zeros(4)
    acc_floor_near = np.zeros(4)
    n_used = 0; n_tot = 0; n_near_used = 0
    n_sel_tot = 0
    for i, d in enumerate(data_te):
        r = pos[ld_names[i]]
        xy = d.pos.numpy()
        y = d.y.numpy()
        surf = d.surf.numpy().astype(bool)
        n_tot += int((~surf).sum())
        if cpb is not None:
            # 第 4 列 = 到壁非负距离（未归一化）
            sdf_raw = d.x.numpy()[:, 4] * si[4] + mi[4]
            i0, i1, wi, j0, j1, wj, sel, d_node = BF.node_map(
                ld_names[i], args.task_root, xy, sdf_raw, ns, nd, d0, d1)
            bad = cpb["near_bad"][idxb[ld_names[i]]].reshape(ns, nd)
            badnode = bad[i0, j0] | bad[i0, j1] | bad[i1, j0] | bad[i1, j1]
            sel = sel & (~badnode)
        else:
            sel = np.zeros(xy.shape[0], dtype=bool)
        n_sel_tot += int(sel.sum())

        # 8 个场（4 预测 + 4 真值）一次插值：点在网格里定位是这里最贵的一步，
        # 分成 8 次调用会把同样的定位重复 8 遍；而且只有**不在贴体带内**的节点
        # 才需要远场插值（约 40%），近壁那 60% 直接走 (s,d) 回读。
        vals8 = np.concatenate([rec[r, n_near:, :], Xn_te[r, n_near:, :]], axis=1)
        far8 = np.full((xy.shape[0], 8), np.nan)
        far_nodes = ~sel
        if far_nodes.any():
            far8[far_nodes] = far_query(vals8, xy[far_nodes])
        if cpb is not None:
            near8 = np.empty((xy.shape[0], 8))
            for k in range(4):
                near8[:, k] = BF.read_near(rec[r, :n_near, k], i0, i1, wi, j0, j1, wj, ns, nd)
                near8[:, 4 + k] = BF.read_near(Xn_te[r, :n_near, k],
                                               i0, i1, wi, j0, j1, wj, ns, nd)
            pv = np.where(sel[:, None], near8[:, :4], far8[:, :4])
            fv = np.where(sel[:, None], near8[:, 4:], far8[:, 4:])
        else:
            pv, fv = far8[:, :4], far8[:, 4:]
        m = np.isfinite(pv).all(axis=1) & (~surf)
        acc += ((pv[m] - y[m]) ** 2).sum(axis=0)
        mf = np.isfinite(fv).all(axis=1) & (~surf)
        acc_floor += ((fv[mf] - y[mf]) ** 2).sum(axis=0)
        mn = m & sel
        acc_near += ((pv[mn] - y[mn]) ** 2).sum(axis=0)
        acc_floor_near += ((fv[mn] - y[mn]) ** 2).sum(axis=0)
        n_near_used += int(mn.sum())
        n_used += int(m.sum())

    if cpb is not None and n_near_used == 0:
        raise SystemExit(
            "贴体近壁带一个节点都没选上（sel 全空）—— 表示静默退化了，必须修，"
            "不能拿这个结果当 wall-fitted 版。检查 airfrans.load 第 4 列的距离口径。")
    cov = n_sel_tot / max(n_tot, 1)
    if cpb is not None:
        print("   贴体带覆盖：%d/%d = %.3f 的节点走 (s,d) 回读" % (n_sel_tot, n_tot, cov),
              flush=True)
    near_out = (None, 0) if cpb is None else (acc_near / max(n_near_used, 1), n_near_used)
    floor_near_out = None if cpb is None else acc_floor_near / max(n_near_used, 1)
    return (acc / max(n_used, 1), (n_used / max(n_tot, 1)), n_used,
            near_out[0], near_out[1],
            acc_floor / max(n_used, 1), floor_near_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default="/home/zch/airfrans/pod/vol_fields_1000.npz")
    ap.add_argument("--manifest", default="/home/zch/airfrans/data/manifest.json")
    ap.add_argument("--task-root", default="/home/zch/airfrans/data/Dataset")
    ap.add_argument("--runs-a", default=os.path.join(
        ROOT, "04_results", "prediction", "runs"),
        help="A 链路的 runs 目录（用于取官方 coef_norm 与 A 的指标做对标）")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--r", type=int, default=POD_R)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-mesh-eval", action="store_true",
                    help="跳过【插值回 CFD 网格】的严格同域评估（默认做，较慢）")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--representation", choices=["grid", "bodyfit"], default="grid",
                    help="grid=原 256x128 均匀网格；bodyfit=笛卡尔网格 + 贴体近壁 (s,d) 带")
    ap.add_argument("--bodyfit-cache",
                    default="/home/zch/airfrans/pod/vol_fields_bodyfit_1000.npz")
    ap.add_argument("--max-eval-samples", type=int, default=0,
                    help="网格回读评估只用前 N 个测试样本（0=全部）；快速自检用")
    args = ap.parse_args()

    rd = RunDir(args.task, "B_field" if args.representation == "grid" else "B_field_bf")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    man = json.load(open(args.manifest, encoding="utf-8"))
    tr_all = man[args.task + "_train"]
    te_names = man["full_test"] if args.task == "scarce" else man[args.task + "_test"]
    n_val = max(1, int(args.val_frac * len(tr_all)))
    n_tr = max(1, len(tr_all) - n_val)
    tr_names, val_names = tr_all[:n_tr], tr_all[n_tr:]
    print(f"[{args.task}] train={n_tr} val={n_val} test={len(te_names)}", flush=True)

    # ---- 真值场（网格 / 双区），并按任务官方 coef_norm 标准化 ----
    cp = np.load(args.cache, allow_pickle=True)
    names_all = cp["names"].tolist()
    idx = {n: i for i, n in enumerate(names_all)}

    # wall-fitted 版：把贴体近壁带拼在笛卡尔网格**前面**，组成"双区"表示
    #   列 0 .. ns*nd-1   近壁贴体 (s,d) 带（弦向弧长站位 x 壁法向对数层）
    #   列 ns*nd ..       远场 256x128 笛卡尔网格（沿用原缓存，一格未动）
    # 评估时按每个节点到壁的距离 d 选区：d<d1 走贴体、否则走笛卡尔（见
    # mesh_domain_eval）。这样既不牺牲远场，又把均匀网格根本分辨不了的
    # 边界层交给贴体坐标系 —— 这是 B 场链路误差的唯一主要来源。
    cpb, idxb = None, None
    if args.representation == "bodyfit":
        cpb = np.load(args.bodyfit_cache, allow_pickle=True)
        nbad = int(cpb["near_bad"].sum())
        print(" 贴体带 NS=%s ND=%s d=[%g,%g]  无效格 %d (%.6f)"
              % (cpb["ns"], cpb["nd"], cpb["d0"], cpb["d1"], nbad, cpb["near_bad"].mean()),
              flush=True)
        if cpb["near_bad"].mean() > 1e-3:
            raise SystemExit("贴体缓存无效格比例 %.4f 过高（法向/射线有问题），先查缓存"
                             % cpb["near_bad"].mean())
        idxb = {n: i for i, n in enumerate(cpb["names"].tolist())}
        missb = [n for n in (tr_all + te_names) if n not in idxb]
        if missb:
            raise SystemExit("贴体缓存缺 %d 个样本，例如 %s" % (len(missb), missb[:3]))

    miss = [n for n in (tr_all + te_names) if n not in idx]
    if miss:
        print(f"!! 缓存缺 {len(miss)} 个样本，例如 {miss[:3]}", flush=True)
    a_metrics = os.path.join(args.runs_a, f"{args.task}_A", "eval_metrics.json")
    if not os.path.exists(a_metrics):
        raise SystemExit(f"找不到 A 的 {a_metrics}（需要它的官方 coef_norm 做标准化）")
    am = json.load(open(a_metrics))
    mean_o = np.asarray(am["coef_norm"]["mean_out"], dtype=np.float64)
    std_o = np.asarray(am["coef_norm"]["std_out"], dtype=np.float64)
    print(" 官方 coef_norm(mean_out) =", np.round(mean_o, 4),
          " std_out =", np.round(std_o, 4), flush=True)

    # 每个 npz 成员**只读一次**再切片：NpzFile 的花式索引会反复读取整个成员，
    # 800 样本 x 4 场这样搞会多花十几分钟。
    # 贴体缓存里 4 个场的键名就是 u/v/p/nu_t（与笛卡尔缓存分开两个文件，不冲突），
    # 用 ns/nd/d0/d1 标明它们是 (s,d) 带上的值。
    far_all = np.stack([np.asarray(cp[k], dtype=np.float64) for k in FIELDS], axis=-1)
    if cpb is not None:
        near_all = np.stack([np.asarray(cpb[k], dtype=np.float64) for k in FIELDS], axis=-1)
        assert near_all.shape[1] == int(cpb["ns"]) * int(cpb["nd"]), \
            "贴体缓存形状与 ns*nd 不符：%s vs %d" % (near_all.shape, int(cpb["ns"]) * int(cpb["nd"]))
        raw_all = np.concatenate([near_all, far_all], axis=1)   # (1000, ns*nd+nfar, 4)
        del near_all
    else:
        raw_all = far_all
    Xn_all = (raw_all - mean_o) / std_o
    del raw_all
    row = {n: i for i, n in enumerate(cpb["names"].tolist() if cpb is not None
                                      else names_all)}

    def pick(namelist):
        return Xn_all[[row[n] for n in namelist]]

    Xn_tr_all = pick(tr_all)
    Xn_te = pick(te_names)
    i_tr = [tr_all.index(n) for n in tr_names]
    i_val = [tr_all.index(n) for n in val_names]
    Xn_tr = Xn_tr_all[i_tr]; Xn_val = Xn_tr_all[i_val]
    del Xn_tr_all

    # ---- 逐场 POD（基只用 train_eff）----
    Phi, Mu, S_full = [], [], []
    for k in range(4):
        Xk = Xn_tr[:, :, k].astype(np.float64)
        mu = Xk.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(Xk - mu, full_matrices=False)
        Phi.append(Vt[:args.r].T); Mu.append(mu); S_full.append(S)
    Phis = np.concatenate(Phi, axis=1)                   # (ncell, 4r)
    Mus = np.concatenate(Mu, axis=0)                     # (1, 4r)

    def alpha_of(Xn):                                    # (N,ncell,4) -> (N,4r)
        return np.concatenate([(Xn[:, :, k] - Mu[k]) @ Phi[k] for k in range(4)], axis=1)

    al_tr = alpha_of(Xn_tr).astype(np.float32)
    al_val = alpha_of(Xn_val).astype(np.float32)
    al_te = alpha_of(Xn_te).astype(np.float32)
    al_mu = al_tr.mean(axis=0, keepdims=True)
    al_sd = np.ones_like(al_mu)
    # 逐分量标准化 POD 坐标**不要**做：网格表示实测 0.384 -> 1.299（差 3.4 倍），
    # 贴体表示只从 10.60 好到 8.65（18%）。标准化让 val loss 变好看，但把权重从
    # 主导场 MSE 的前几个模态挪走，端到端反而更差。这里保留 mu/sd 只为落盘记录。
    print(" POD 坐标尺度：|alpha| 最大 %.3g  末位分量中位 |alpha| %.3g  比值 %.1f"
          % (np.abs(al_tr).max(), np.median(np.abs(al_tr[:, -1])),
             np.abs(al_tr).max() / max(np.median(np.abs(al_tr[:, -1])), 1e-12)), flush=True)

    # ---- 输入 ----
    Xtr, Xval, Xte, in_stats = build_inputs(tr_names, val_names, te_names, args.task_root)
    print(f" 输入维度 {Xtr.shape[1]}，目标 {al_tr.shape[1]}（4 场 x r={args.r}）", flush=True)
    rd.save_hyperparams({
        "chain": "B_field", "task": args.task, "representation": args.representation,
        "pod_r": args.r, "hidden": args.hidden, "n_layers": 4,
        "dim_in": int(Xtr.shape[1]), "target_dim": int(al_tr.shape[1]),
        "val_frac": args.val_frac, "lr": args.lr, "wd": args.wd, "batch": args.batch,
        "optimizer": "AdamW", "alpha_standardized": False,
        "n_epochs": args.epochs, "patience": args.patience, "seed": args.seed,
        "cache": os.path.abspath(args.cache),
        "bodyfit_cache": (os.path.abspath(args.bodyfit_cache) if cpb is not None else None),
        "coef_norm_source": a_metrics,
        "train_n_eff": int(len(tr_names)), "val_n": int(len(val_names)),
        "test_n": int(len(te_names)), "n_grid_cells": int(Phis.shape[0]),
        "input": "5 来流标量 + 128 维几何（弦向 y/c 曲线）",
        "output": ("4 场 x r POD 坐标 -> 贴体近壁带 + 笛卡尔远场 双区还原"
                   if cpb is not None else "4 场 x r POD 坐标 -> 256x128 网格还原"),
        "norm_stats_in": "见 best_model.pt 的 norm_stats 字段（含 POD 基路径与键名）",
    })

    # ---- 训练 ----
    dev = args.device
    model = FieldMLP(Xtr.shape[1], al_tr.shape[1], hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.MSELoss()
    tr_ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(al_tr)),
        batch_size=args.batch, shuffle=True)
    Xval_t = torch.from_numpy(Xval).to(dev); al_val_t = torch.from_numpy(al_val).to(dev)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tl += loss.item()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xval_t), al_val_t).item()
        rd.append_loss(ep, tl / max(len(tr_ld), 1), vl,
                       extra={"r": args.r, "fields": 4, "lr": args.lr})
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{args.task}] early stop @ep{ep} (best val={best:.4f})", flush=True)
                break
        if is_nan(vl):
            print(f"[{args.task}] NaN @ep{ep} -> abort", flush=True); break
    # ---- 落盘：权重 + 归一化统计量 + 4 场 POD 基 ----
    # 没有这两样，权重文件拿到别处就无法预测（算不出输入归一化、还原不出场）。
    # 统计量很小，直接内联进 best_model.pt；POD 基太大，存到与缓存同目录的
    # bfield_basis.npz，按 task+表示 分键，可反复增量写入。
    tag = args.task if args.representation == "grid" else args.task + "_bf"
    basis_npz = os.path.join(os.path.dirname(os.path.abspath(args.cache)),
                             "bfield_basis.npz")
    try:
        old = dict(np.load(basis_npz, allow_pickle=True)) if os.path.exists(basis_npz) else {}
    except Exception:
        old = {}
    old["Phi_" + tag] = np.stack(Phi)                # (4, ncell, r)
    old["mu_" + tag] = np.stack(Mu)                  # (4, 1, ncell)
    old["S_" + tag] = np.stack(S_full)               # (4, ncell)，全谱，供截断分析
    old["ncell_" + tag] = np.array([Phis.shape[0]])
    np.savez(basis_npz, **old)
    norm_stats = {
        "chain": "B_field", "task": args.task, "pod_r": int(args.r),
        "representation": args.representation,
        "n_near": (int(cpb["ns"]) * int(cpb["nd"]) if cpb is not None else 0),
        "smean": in_stats["smean"].tolist(), "sstd": in_stats["sstd"].tolist(),
        "gmean": in_stats["gmean"].tolist(), "gstd": in_stats["gstd"].tolist(),
        "geom_station": in_stats["geom_station"],
        "fields": FIELDS, "n_grid_cells": int(Phis.shape[0]),
        "alpha_norm": {"mu": al_mu.ravel().tolist(), "sd": al_sd.ravel().tolist()},
        "cache": os.path.abspath(args.cache),
        "bodyfit_cache": (os.path.abspath(args.bodyfit_cache) if cpb is not None else None),
        "basis_npz": basis_npz,
        "basis_keys": ["Phi_%s" % tag, "mu_%s" % tag],
        "coef_norm": {"mean_out": mean_o.tolist(), "std_out": std_o.tolist()},
        "tr_eff_names": list(tr_names), "val_names": list(val_names),
    }
    rd.save_best({"model_state": best_state, "task": args.task,
                  "norm_stats": norm_stats, "_norm_stats": norm_stats}, best, ep)
    model.load_state_dict(best_state, strict=False); model.eval()

    # ---- 评估（网格域）----
    def recon(al):                                       # (N,4r) -> (N,ncell,4)
        out = np.empty((al.shape[0], Phis.shape[0], 4))
        for k in range(4):
            out[:, :, k] = al[:, k * args.r:(k + 1) * args.r] @ Phi[k].T + Mu[k]
        return out

    def mse_per_field(pred_n, true_n):
        return ((pred_n - true_n) ** 2).mean(axis=(0, 1))

    def relL2_per_field(pred_n, true_n):
        p = pred_n * std_o + mean_o
        t = true_n * std_o + mean_o
        num = np.linalg.norm((p - t).reshape(-1, 4), axis=0)
        den = np.linalg.norm(t.reshape(-1, 4), axis=0)
        return num / np.maximum(den, 1e-12)

    with torch.no_grad():
        pred_te = model(torch.from_numpy(Xte).to(dev)).cpu().numpy()
    rec = recon(pred_te)
    mse_pred = mse_per_field(rec, Xn_te)
    rel_pred = relL2_per_field(rec, Xn_te)
    # 对照
    rec_trunc = recon(al_te)                                    # rank-r oracle
    rec_mean = recon(np.tile(al_mu, (len(te_names), 1)))        # 均值场
    from sklearn.linear_model import Ridge
    rg = Ridge(alpha=1.0).fit(Xtr.astype(np.float64), al_tr.astype(np.float64))
    rec_ridge = recon(rg.predict(Xte.astype(np.float64)))
    grid_floor = float(np.mean([np.sqrt(np.sum(S_full[k][args.r:] ** 2) /
                                         np.sum(S_full[k] ** 2)) for k in range(4)]))

    # ---- 严格同域对标：插值回 CFD 网格节点，用与 A 完全相同的口径重算 ----
    mesh_mse = mesh_cov = None
    mesh_n = 0
    near_mse = None
    near_n = 0
    floor_mse = None
    floor_near_mse = None
    if not args.no_mesh_eval:
        print(f"[{args.task}] mesh-domain eval (插值回 CFD 网格节点, "
              f"{args.representation})...", flush=True)
        (mesh_mse, mesh_cov, mesh_n, near_mse, near_n,
         floor_mse, floor_near_mse) = mesh_domain_eval(
            rec, Xn_te, te_names, cp, cpb, args, am)
        if near_mse is not None:
            print("   近壁带内逐场 MSE(x1e-2) = %s  节点=%d"
                  % (np.round(near_mse * 100, 3), near_n), flush=True)
        print("   表示下限 全网格 %.3f%s（模型不可能低于这条线）"
              % (floor_mse.mean() * 100,
                 (" | 近壁带 %.3f" % (floor_near_mse.mean() * 100))
                 if floor_near_mse is not None else ""), flush=True)
        print("   网格域逐场 MSE(x1e-2) = %s  覆盖率=%.3f  参与节点=%d"
              % (np.round(mesh_mse * 100, 3), mesh_cov, mesh_n), flush=True)

    spear = {}
    for k, name in enumerate(FIELDS):
        t = Xn_te[:, :, k].ravel(); p = rec[:, :, k].ravel()
        spear[name] = float(spearmanr(t, p).correlation) if np.std(t) > 1e-12 else None

    a_mse = am.get("test_volume_MSE_x1e-2", {})
    metrics = {
        "task": args.task, "chain": "B_field", "pod_r": args.r,
        "representation": args.representation,
        "target": ("贴体近壁 (s,d) 带 + 远场 256x128 笛卡尔网格 上的 4 个体积场 (u,v,p,nu_t)"
                   if cpb is not None else
                   "256x128 公共网格上的 4 个体积场 (u,v,p,nu_t)"),
        "normalization": "用 A 链路该任务的官方 coef_norm(mean_out/std_out) 标准化",
        "alpha_standardized": False,
        "alpha_norm": {"mu": al_mu.ravel().tolist(), "sd": al_sd.ravel().tolist()},
        "train_n_eff": int(len(tr_names)), "val_n": int(len(val_names)),
        "test_n": int(len(te_names)), "n_grid_cells": int(Phis.shape[0]),
        "n_epochs_run": ep + 1, "final_val_alpha_loss": float(best),
        # —— B 的场预测误差（归一化 MSE，与 A 同定义、域为网格子采样）——
        "field_MSE_grid_x1e-2": {FIELDS[k]: float(mse_pred[k] * 100) for k in range(4)},
        "field_MSE_grid_mean_x1e-2": float(mse_pred.mean() * 100),
        # —— 严格同域对标：与 A 完全相同的 CFD 网格口径 ——
        "field_MSE_mesh_x1e-2": ({FIELDS[k]: float(mesh_mse[k] * 100) for k in range(4)}
                                 if mesh_mse is not None else None),
        "field_MSE_mesh_mean_x1e-2": (float(mesh_mse.mean() * 100)
                                      if mesh_mse is not None else None),
        "field_MSE_near_band_x1e-2": ({FIELDS[k]: float(near_mse[k] * 100) for k in range(4)}
                                      if near_mse is not None else None),
        "near_band_nodes": near_n,
        # —— 本表示方式的下限（真值目标走同一条回读路径），与模型误差对比才知瓶颈在哪 ——
        "floor_MSE_mesh_x1e-2": ({FIELDS[k]: float(floor_mse[k] * 100) for k in range(4)}
                                 if floor_mse is not None else None),
        "floor_MSE_mesh_mean_x1e-2": (float(floor_mse.mean() * 100)
                                      if floor_mse is not None else None),
        "floor_MSE_near_band_mean_x1e-2": (float(floor_near_mse.mean() * 100)
                                           if floor_near_mse is not None else None),
        "mesh_eval_coverage": mesh_cov,
        "field_relL2_grid": {FIELDS[k]: float(rel_pred[k]) for k in range(4)},
        "field_spearman_grid": spear,
        # —— 三个对照（同网格同口径）——
        "field_MSE_grid_meanfield_x1e-2": float(mse_per_field(rec_mean, Xn_te).mean() * 100),
        "field_MSE_grid_rankr_trunc_x1e-2": float(mse_per_field(rec_trunc, Xn_te).mean() * 100),
        "field_MSE_grid_ridge_x1e-2": float(mse_per_field(rec_ridge, Xn_te).mean() * 100),
        "pod_trunc_energy_relL2_mean": grid_floor,
        # —— A 链路同任务数字（域不同：A 在 ~18 万网格节点）——
        "A_field_MSE_x1e-2": a_mse,
        "A_field_MSE_mean_x1e-2": am.get("test_volume_MSE_mean_x1e-2"),
        "A_surface_p_MSE_x1e-1": am.get("test_surface_p_MSE_x1e-1"),
        "A_params": am.get("params"), "A_n_epochs": am.get("n_epochs"),
        "domain_caveat": ("A 的 MSE 在 ~18 万 CFD 网格节点上算；B 在 256x128 公共网格的 "
                          "%d 个有效格上算。两者都是【官方 coef_norm 归一化场 MSE】，"
                          "定义相同、域不同（网格是域内均匀子采样），并列时需注明。"
                          % Phis.shape[0]),
        "loss_csv": "loss_history.csv", "best_model": "best_model.pt",
    }
    rd.save_eval(metrics)
    print(f"[{args.task}] B_field done: grid MSE(mean,x1e-2)={metrics['field_MSE_grid_mean_x1e-2']:.3f} "
          f"| 对照 meanfield={metrics['field_MSE_grid_meanfield_x1e-2']:.3f} "
          f"rankr={metrics['field_MSE_grid_rankr_trunc_x1e-2']:.3f} "
          f"ridge={metrics['field_MSE_grid_ridge_x1e-2']:.3f} "
          f"| A(网格外,均值)={metrics['A_field_MSE_mean_x1e-2']}", flush=True)
    print("   逐场 B:", {k: round(v, 3) for k, v in metrics["field_MSE_grid_x1e-2"].items()}, flush=True)
    print("   逐场 A:", {k: round(v, 3) for k, v in (a_mse or {}).items()}, flush=True)


if __name__ == "__main__":
    main()

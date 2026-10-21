"""速度基准：量化 A 链路（官方 GNN 大模型）vs B 链路（POD 降维小模型）的
训练成本 与 单样本预测响应时间。

A 链路：逐点图神经网络，输入 18 万个点，需要建 radius graph。
B 链路：133 维向量 -> 8 维 POD 坐标 -> 256 点曲线，纯 MLP 前向。

测量项：
  1. 参数量
  2. 单样本预测耗时（含/不含数据读取）
  3. 单样本预测的显存/内存占用
  4. 批量预测吞吐（100 个样本）
  5. 训练总时长（从历史的 eval_metrics.json 里读 _total_train_sec）
"""
import os, sys, json, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from runkit import RunDir  # noqa: E402
from train_A_gnn import (OfficialGraphSAGE, build_radius_graph, SUBSAMP)  # noqa: E402
from train_B_transformer import BBackbone, cp_from_alpha  # noqa: E402
from geometry_input import load_all_geometric, input_vector_5scalar  # noqa: E402
from single_sample import load_sample  # noqa: E402

TASK_ROOT = "/home/user/airfrans/data/Dataset"
WALLCP = "/home/user/airfrans/pod/wallcp_1000.npz"


def timeit(fn, n=1, warmup=1):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    task = "full"
    te = man["full_test"]

    # ---------- 参数量 ----------
    a_model = OfficialGraphSAGE().to(device)
    b_model = BBackbone(dim_in=133, dim_out=8, hidden=256, n_layers=4).to(device)
    a_params = a_model.count_params()
    b_params = sum(p.numel() for p in b_model.parameters())
    print("=" * 66)
    print("参数量：A=%d  B=%d" % (a_params, b_params))

    # ---------- 数据读取耗时 ----------
    nm = te[0]
    t_read = timeit(lambda: load_sample(nm), n=3)
    print("单样本数据读取（.vtu 15MB）：%.3f s" % t_read)

    # ---------- B 链路预测 ----------
    b_model.eval()
    scal = np.stack([input_vector_5scalar(n) for n in te[:100]])
    geo = load_all_geometric(te[:100], TASK_ROOT)
    smean, sstd = scal.mean(0, keepdims=True), scal.std(0, keepdims=True)
    gmean, gstd = geo.mean(0, keepdims=True), geo.std(0, keepdims=True)
    X100 = np.concatenate([(scal - smean) / sstd, (geo - gmean) / gstd],
                          axis=1).astype(np.float32)
    xb = torch.from_numpy(X100).to(device)

    with torch.no_grad():
        t_b_single = timeit(lambda: b_model(xb), n=20)
    X1 = torch.from_numpy(X100[:1]).to(device)
    with torch.no_grad():
        t_b_one = timeit(lambda: b_model(X1), n=50)
    print("B 单样本前向：%.4f ms   100 样本批量：%.2f ms（合 %.4f ms/样本）"
          % (t_b_one * 1000, t_b_single * 1000, t_b_single * 1000 / 100))

    # ---------- A 链路预测（单样本，全覆盖子采样口径） ----------
    s = load_sample(nm)
    P = s["pos"]
    N = P.shape[0]
    xn = (torch.from_numpy(s["x"]).to(device))
    a_model.eval()

    def a_infer_once():
        rng = np.random.RandomState(0)
        seen = np.zeros(N, dtype=bool)
        pred = np.zeros((N, 4)); cnt = np.zeros(N, dtype=int)
        with torch.no_grad():
            for _ in range(200):
                if seen.all():
                    break
                sel = rng.choice(N, size=min(SUBSAMP, N), replace=False)
                it = torch.from_numpy(sel).long()
                e = build_radius_graph(P[sel], device)
                o = a_model(xn[it], e).cpu().numpy()
                pred[sel] += o; cnt[sel] += 1; seen[sel] = True
        return pred

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_a_once = timeit(a_infer_once, n=1, warmup=0)
    peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    n_rounds = int(np.ceil(N / SUBSAMP))
    print("A 单样本全覆盖推理：%.2f s（%d 轮 x 32k 点）峰值显存 %.0f MB"
          % (t_a_once, n_rounds, peak))

    # ---------- 加速比 ----------
    total_b = t_read + t_b_one
    total_a = t_read + t_a_once
    print("-" * 66)
    print("单样本端到端（含读数据）：A=%.2f s   B=%.4f s   加速比=%.0fx"
          % (total_a, total_b, total_a / total_b))
    print("纯推理（不含读数据）：   A=%.2f s   B=%.4f s   加速比=%.0fx"
          % (t_a_once, t_b_one, t_a_once / t_b_one))

    # ---------- 训练时长（历史） ----------
    print("-" * 66)
    print("训练时长（历史实测，取自 eval_metrics.json）：")
    tot_a, tot_b = 0.0, 0.0
    for t in ["full", "scarce", "reynolds", "aoa"]:
        ma = json.load(open(os.path.join(RunDir(t, "A").dir, "eval_metrics.json"),
                            encoding="utf-8"))
        mb = json.load(open(os.path.join(RunDir(t, "B_official").dir, "eval_metrics.json"),
                            encoding="utf-8"))
        sa, sb = ma.get("_total_train_sec", 0), mb.get("_total_train_sec", 0)
        tot_a += sa; tot_b += sb
        print("  %-9s A=%8.1f s (%4.1f h)   B=%7.1f s   A/B=%6.1fx"
              % (t, sa, sa / 3600, sb, sa / max(sb, 1e-9)))
    print("  合计      A=%8.1f s (%4.1f h)   B=%7.1f s   A/B=%6.1fx"
          % (tot_a, tot_a / 3600, tot_b, tot_a / max(tot_b, 1e-9)))

    out = {
        "params_A": a_params, "params_B": b_params,
        "read_one_sample_sec": t_read,
        "B_forward_one_sec": t_b_one, "B_forward_100_sec": t_b_single,
        "A_infer_one_sample_sec": t_a_once, "A_peak_mem_MB": peak,
        "speedup_end_to_end": total_a / total_b,
        "speedup_infer_only": t_a_once / t_b_one,
        "train_sec": {t: {"A": json.load(open(os.path.join(RunDir(t, "A").dir, "eval_metrics.json"),
                                              encoding="utf-8")).get("_total_train_sec"),
                          "B": json.load(open(os.path.join(RunDir(t, "B_official").dir, "eval_metrics.json"),
                                              encoding="utf-8")).get("_total_train_sec")}
                      for t in ["full", "scarce", "reynolds", "aoa"]},
        "train_total_sec": {"A": tot_a, "B": tot_b},
        "device": device,
    }
    json.dump(out, open("/home/user/airfrans/prediction/04_results/prediction/speed_benchmark.json",
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved speed_benchmark.json")
    print("BENCH_DONE")


if __name__ == "__main__":
    main()

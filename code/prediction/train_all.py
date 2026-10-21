"""前向预测阶段：8 个训练任务（4 官方任务 x 2 链路）双卡并发调度 + 4 件套记录。

官方口径对齐版（P3 重做）：
  A 链路：官方 GraphSAGE（radius graph r=0.05, max_neighbors=64, SAGE×3×64 +
          encoder/decoder MLP, 逐点 7 维输入/4 维目标, 每 epoch 子采样 32000 节点,
          volume+λ·surface loss λ=1）-> 逐点流场
  B 链路：官方目标口径（5 来流标量 + 128 几何向量 -> POD r=8 坐标 -> 还原 Cp ->
          C_L/C_D 相对误差 + Spearman 对标官方 GraphSAGE full C_L=0.517）

调度（双卡）：
  GPU0(cuda:0): A-full -> A-reynolds -> A-aoa -> A-scarce（重，串行）
  GPU1(cuda:1): B-full -> B-reynolds -> B-aoa -> B-scarce（轻，串行）

4 件套：每任务 04_results/prediction/runs/<task>_<chain>/
  {hyperparams.json, loss_history.csv, best_model.pt, eval_metrics.json}

冒烟闸门：先 --smoke-only 100 样本验证不 NaN/OOM 再放行全量。
止损：B full 误差 > 2 x POD 截断误差 -> 停；A 单 epoch > 2h 不降 -> 停。
"""
import os, sys, json, time, argparse, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
LOG = os.path.join(ROOT, "04_results", "prediction", "logs")
RUNS = os.path.join(ROOT, "04_results", "prediction", "runs")
os.makedirs(LOG, exist_ok=True); os.makedirs(RUNS, exist_ok=True)


def log(msg):
    print("[train_all %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def run_chain_B(task, device, workdir, init_from=None, smoke=False,
                wallcp_npz=None, manifest=None, task_root=None):
    """B 链路官方口径：几何向量 + 5 标量 -> POD 坐标 -> C_L/C_D 评估。"""
    cmd = ["python", os.path.join(workdir, "train_B_transformer.py"),
           "--task", task, "--device", device, "--epochs", "80",
           "--lr", "1e-3", "--patience", "12", "--hidden", "256"]
    if init_from:
        cmd += ["--init-from", init_from]
    if wallcp_npz:
        cmd += ["--wallcp-npz", wallcp_npz]
    if manifest:
        cmd += ["--manifest", manifest]
    if task_root:
        cmd += ["--task-root", task_root]
    if smoke:
        cmd.append("--smoke-only")
    log("B %s %s -> %s" % (task, "SMOKE" if smoke else "FULL", device))
    return subprocess.run(cmd, cwd=workdir).returncode


def run_chain_A(task, device, workdir, init_from=None, smoke=False,
                task_root=None, limit=None, epochs=398):
    """A 链路官方 GraphSAGE 口径。"""
    cmd = ["python", os.path.join(workdir, "train_A_gnn.py"),
           "--task", task, "--device", device, "--epochs", str(epochs),
           "--lr", "1e-3"]
    if init_from:
        cmd += ["--init-from", init_from]
    if task_root:
        cmd += ["--task-root", task_root]
    if limit:
        cmd += ["--limit", str(limit)]
    log("A %s %s -> %s" % (task, "SMOKE" if smoke else "FULL", device))
    return subprocess.run(cmd, cwd=workdir).returncode


def read_best(runs_dir, task, chain):
    p = os.path.join(runs_dir, f"{task}_{chain}", "best_model.pt")
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="full,scarce,reynolds,aoa")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--wallcp-npz", default=None,
                    help="B 链路壁面 Cp 全量 npz（工作站 /home/zch/airfrans/pod/wallcp_1000.npz）")
    ap.add_argument("--manifest", default=None,
                    help="官方 4 任务 manifest.json（工作站 /home/zch/airfrans/data/manifest.json）")
    ap.add_argument("--task-root", default=None,
                    help="airfrans load 数据根（工作站 /home/zch/airfrans/data/Dataset）；"
                         "B 链路几何生成器也读它")
    ap.add_argument("--smoke-limit", type=int, default=100)
    ap.add_argument("--smoke-epochs", type=int, default=3,
                    help="冒烟阶段 A 链路 epoch 数（3 看趋势，非 398 全量）")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    order = [t for t in ["full", "reynolds", "aoa", "scarce"] if t in tasks]
    gpu_b = "cuda:1"; gpu_a = "cuda:0"
    workdir = _HERE
    rc_total = 0

    if args.smoke_only:
        log("=== SMOKE GATE: B + A (100 samples, 3 epochs, radius graph) ===")
        ok_b = run_chain_B("full", gpu_b, workdir, smoke=True,
                           wallcp_npz=args.wallcp_npz, manifest=args.manifest,
                           task_root=args.task_root) == 0
        ok_a = run_chain_A("full", gpu_a, workdir, smoke=True,
                           task_root=args.task_root, limit=args.smoke_limit,
                           epochs=args.smoke_epochs) == 0
        rc_total = 0 if (ok_b and ok_a) else 1
        log("SMOKE B=%s A=%s -> %s" % (ok_b, ok_a,
                                        "PASS, 放行全量" if ok_b and ok_a
                                        else "FAIL, 不硬跑"))
        sys.exit(rc_total)

    # 全量：B 链路 GPU1 串行
    log("=== FULL: B 链路官方口径 GPU1 串行 ===")
    for t in order:
        rc = run_chain_B(t, gpu_b, workdir, wallcp_npz=args.wallcp_npz,
                         manifest=args.manifest, task_root=args.task_root)
        rc_total = rc_total or rc
        if rc != 0:
            log("B %s rc=%d -> STOP (止损)" % (t, rc)); break

    log("=== FULL: A 链路官方 GraphSAGE GPU0 串行 ===")
    for t in order:
        rc = run_chain_A(t, gpu_a, workdir, task_root=args.task_root,
                         epochs=398)
        rc_total = rc_total or rc
        if rc != 0:
            log("A %s rc=%d -> STOP (止损)" % (t, rc)); break

    log("=== ALL DONE rc_total=%d ===" % rc_total)
    summary = {}
    for t in order:
        for c in ["B", "A"]:
            p = os.path.join(RUNS, f"{t}_{c}", "eval_metrics.json")
            if os.path.exists(p):
                summary[f"{t}_{c}"] = json.load(open(p, encoding="utf-8"))
    sp = os.path.join(RUNS, "summary_all.json")
    json.dump(summary, open(sp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    log("summary -> %s" % sp)
    sys.exit(rc_total)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        log("FATAL\n" + traceback.format_exc())
        sys.exit(2)

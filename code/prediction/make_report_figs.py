"""生成报告用的汇总图表（柱状图/箱线图），数据全部来自历史结果文件，不重训。

输出到 04_results/prediction/figs/report_*.png：
  report_A_fields_bar.png     A 链路 4 任务 × 4 场量的相对误差（分组柱状图）
  report_A_vs_official.png    A 链路 vs 官方 Table5（full 任务）对比柱状图
  report_B_baselines_bar.png  B 链路 vs 3 个基线（4 任务分组柱状图）
  report_B_seed_box.png       B 链路 5 次重复的误差分布（箱线图）
  report_B_cl.png             B 链路升力误差 + 排序相关性（柱状图）
  report_speed.png            训练时长 + 推理时间对比（对数柱状图）
  report_train_loss.png       训练损失曲线（A 与 B，各自 normalize）
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
# 指向项目根下的正式位置：03_code/prediction/ -> ../../04_results/prediction/
PRED = os.path.normpath(os.path.join(_HERE, "..", "..", "04_results", "prediction"))
FIGDIR = os.path.join(PRED, "figs")
os.makedirs(FIGDIR, exist_ok=True)
TASKS = ["full", "scarce", "reynolds", "aoa"]
TFIELD = ["u", "v", "p", "nu_t"]


def load_json(p):
    return json.load(open(p, encoding="utf-8"))


def _metrics(task, chain):
    """读某任务某链路的 eval_metrics.json（工作站上是 runs/<task>_<chain>/ 目录）。"""
    p = os.path.join(PRED, "runs", "%s_%s" % (task, chain), "eval_metrics.json")
    return json.load(open(p, encoding="utf-8"))


def _loss_csv(task, chain):
    return os.path.join(PRED, "runs", "%s_%s" % (task, chain), "loss_history.csv")


# ---------- 1. A 链路 4 场量误差 ----------
def fig_A_fields():
    data = {}
    for t in TASKS:
        m = _metrics(t, "A")
        data[t] = m["test_relL2_per_field"]
    x = np.arange(len(TASKS)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, f in enumerate(TFIELD):
        vals = [data[t][f] * 100 for t in TASKS]
        b = ax.bar(x + (i - 1.5) * w, vals, w, label=f)
        ax.bar_label(b, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(TASKS)
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title("Chain A (official GraphSAGE): per-field relative error by task")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(FIGDIR, "report_A_fields_bar.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("[fig]", p)


# ---------- 2. A vs 官方 ----------
def fig_A_vs_official():
    ours = _metrics("full", "A")
    p3 = load_json(os.path.join(PRED, "P3_comparison.json"))
    ref = p3["official_reference"]["GraphSAGE_full"]
    labels = ["u", "v", "p", "nu_t"]
    ours_v = [ours["test_volume_MSE_x1e-2"][k] for k in labels]
    ref_v = [ref["volume_MSE_x1e-2"][k] for k in labels]
    x = np.arange(len(labels)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    b1 = ax.bar(x - w/2, ours_v, w, label="ours (chain A, 1 run)")
    b2 = ax.bar(x + w/2, ref_v, w, label="official paper (5-run mean)")
    ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("volume MSE (x1e-2, official caliber)")
    ax.set_title("full task: volume-field MSE")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    our_sp = ours["test_surface_p_MSE_x1e-1"]
    ref_sp = ref["surface_p_MSE_x1e-1"]
    b1 = ax.bar([0], [our_sp], 0.35, label="ours")
    b2 = ax.bar([0.4], [ref_sp], 0.35, label="official")
    ax.bar_label(b1, fmt="%.3f"); ax.bar_label(b2, fmt="%.3f")
    ax.set_xticks([0, 0.4]); ax.set_xticklabels(["ours", "official"])
    ax.set_ylabel("surface p MSE (x1e-1)")
    ax.set_title("full task: wall pressure MSE (ours better)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(FIGDIR, "report_A_vs_official.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("[fig]", p)


# ---------- 3. B 链路 vs 基线 ----------
def fig_B_baselines():
    ms = load_json(os.path.join(PRED, "B_multiseed.json"))
    keys = ["cp_relL2_test", "cp_relL2_meanfield_test",
            "cp_relL2_rank8_trunc_test", "cp_relL2_ridge_test"]
    lab = ["B-chain (ours)", "baseline: mean field",
           "baseline: rank-8 POD trunc", "baseline: ridge"]
    x = np.arange(len(TASKS)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (k, l) in enumerate(zip(keys, lab)):
        vals = [ms[t][k]["mean"] for t in TASKS]
        errs = [ms[t][k]["std"] for t in TASKS]
        b = ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=3, label=l)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(TASKS)
    ax.set_ylabel("Cp per-sample relative L2 (lower is better)")
    ax.set_title("Chain B (POD reduced-dim) vs 3 baselines, 5-seed mean")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(FIGDIR, "report_B_baselines_bar.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("[fig]", p)


# ---------- 4. B 链路 5 seed 箱线图 ----------
def fig_B_seed_box():
    ms = load_json(os.path.join(PRED, "B_multiseed.json"))
    data = [ms[t]["per_seed_cp_relL2"] for t in TASKS]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(data, labels=TASKS, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
    for i, d in enumerate(data):
        ax.scatter([i + 1] * len(d), d, color="k", s=18, zorder=3)
    ax.set_ylabel("Cp per-sample relative L2")
    ax.set_title("Chain B: seed-to-seed spread (5 runs per task, dots = individual runs)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(FIGDIR, "report_B_seed_box.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("[fig]", p)


# ---------- 5. B 链路 C_L ----------
def fig_B_cl():
    ms = load_json(os.path.join(PRED, "B_multiseed.json"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(TASKS)); w = 0.35
    cl = [ms[t]["C_L_relerr_pred_test"]["mean"] for t in TASKS]
    cle = [ms[t]["C_L_relerr_pred_test"]["std"] for t in TASKS]
    b = ax[0].bar(x, cl, w, yerr=cle, capsize=4, color="#e57373")
    ax[0].bar_label(b, fmt="%.2f", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(TASKS)
    ax[0].set_ylabel("C_L relative error vs official truth")
    ax[0].set_title("Chain B: lift coefficient error (5-seed mean±std)")
    ax[0].grid(axis="y", alpha=0.3)

    sp = [ms[t]["C_L_spearman_pred"]["mean"] for t in TASKS]
    spe = [ms[t]["C_L_spearman_pred"]["std"] for t in TASKS]
    b = ax[1].bar(x, sp, w, yerr=spe, capsize=4, color="#81c784")
    ax[1].bar_label(b, fmt="%.3f", fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(TASKS)
    ax[1].set_ylim(0, 1.0)
    ax[1].set_ylabel("Spearman correlation")
    ax[1].set_title("Chain B: lift ranking correlation (1.0 = perfect)")
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(FIGDIR, "report_B_cl.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("[fig]", p)


# ---------- 6. 速度 ----------
def fig_speed():
    sp = load_json(os.path.join(PRED, "speed_benchmark.json"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    tr = sp["train_sec"]
    A = [tr[t]["A"] for t in TASKS]
    B = [tr[t]["B"] for t in TASKS]
    x = np.arange(len(TASKS)); w = 0.35
    b1 = ax.bar(x - w/2, A, w, label="chain A (GNN)")
    b2 = ax.bar(x + w/2, B, w, label="chain B (POD)")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f", fontsize=7, padding=1)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(TASKS)
    ax.set_ylabel("training time (s, log scale)")
    ax.set_title("training time per task")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    tA = sp["train_total_sec"]["A"]; tB = sp["train_total_sec"]["B"]
    b = ax.bar(["chain A", "chain B"], [tA, tB], color=["#e57373", "#81c784"])
    ax.bar_label(b, fmt="%.0f s", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("total training time (s, log scale)")
    ax.set_title("4-task total: %.1f h vs %.1f s (%.0fx)"
                 % (tA/3600, tB, tA/tB))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    inferA = sp["A_infer_one_sample_sec"] * 1000
    inferB = sp["B_forward_one_sec"] * 1000
    b = ax.bar(["chain A", "chain B"], [inferA, inferB], color=["#e57373", "#81c784"])
    ax.bar_label(b, fmt="%.3f ms", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("per-sample inference (ms, log scale)")
    ax.set_title("inference speedup = %.0fx" % (inferA / inferB))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(FIGDIR, "report_speed.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("[fig]", p)


# ---------- 7. 训练损失曲线 ----------
def fig_loss():
    import csv
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for t in TASKS:
        p = _loss_csv(t, "A")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        ep = [int(r["epoch"]) for r in rows]
        tr = [float(r["train_loss"]) for r in rows]
        vl = [float(r["val_loss"]) for r in rows]
        axes[0].plot(ep, tr, lw=1, label="%s train" % t)
        axes[0].plot(ep, vl, lw=1, ls="--", label="%s val" % t)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss (log)")
    axes[0].set_title("Chain A loss history (398 epochs, official MSE)")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)

    for t in TASKS:
        p = _loss_csv(t, "B_official")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        ep = [int(r["epoch"]) for r in rows]
        tr = [float(r["train_loss"]) for r in rows]
        vl = [float(r["val_loss"]) for r in rows]
        axes[1].plot(ep, tr, lw=1, label="%s train" % t)
        axes[1].plot(ep, vl, lw=1, ls="--", label="%s val" % t)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("alpha loss (log)")
    axes[1].set_title("Chain B loss history (POD-coordinate MSE)")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(FIGDIR, "report_train_loss.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("[fig]", p)


if __name__ == "__main__":
    fig_A_fields()
    fig_A_vs_official()
    fig_B_baselines()
    fig_B_seed_box()
    fig_B_cl()
    fig_speed()
    fig_loss()
    print("REPORT_FIGS_DONE")

"""统一评估（3.4）：汇总 4 任务 x 2 链路的 eval_metrics.json，
按 4 位族 / 5 位族分层报误差，B 链路额外两段拆解（POD 截断 vs 代理误差）。

输出：04_results/prediction/summary_all.json + summary_all.md（人读）

用法：
  python eval_unified.py
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
RUNS = os.path.join(ROOT, "04_results", "prediction", "runs")
MANIFEST = os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json")


def classify_family(name):
    """airFoil2D_SST_<U>_<AoA>_<...>：后 3 段 = 4 位族，后 4 段 = 5 位族。"""
    p = name.replace("airFoil2D_SST_", "").split("_")
    return "5digit" if len(p) - 2 == 4 else "4digit"


def main():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    # 按族分层：每个任务的 test 集合里 4 位 / 5 位各多少
    fam_counts = {}
    for task in ["full", "scarce", "reynolds", "aoa"]:
        test_names = man[task + "_test"]
        fam_counts[task] = {
            "4digit": sum(1 for n in test_names if classify_family(n) == "4digit"),
            "5digit": sum(1 for n in test_names if classify_family(n) == "5digit"),
        }
    # 读所有 eval
    summary = {}
    for task in ["full", "scarce", "reynolds", "aoa"]:
        for chain in ["B", "A"]:
            p = os.path.join(RUNS, f"{task}_{chain}", "eval_metrics.json")
            if os.path.exists(p):
                summary[f"{task}_{chain}"] = json.load(open(p, encoding="utf-8"))
            else:
                summary[f"{task}_{chain}"] = {"_missing": True}
    # 写 md
    out = []
    out.append("# 前向预测统一评估汇总（3.4）\n")
    out.append("## 按族分层 test 计数\n")
    out.append("| 任务 | 4位族 | 5位族 |")
    out.append("|---|---|---|")
    for t, c in fam_counts.items():
        out.append("| %s | %d | %d |" % (t, c["4digit"], c["5digit"]))
    out.append("\n## 两链路 x 4 任务主指标\n")
    out.append("| 任务 | 链路 | 末次 val loss | alpha_relL2 (B) / C_L RMSE (A) | 训练时长(s) |")
    out.append("|---|---|---|---|---|")
    for key, m in summary.items():
        if m.get("_missing"):
            out.append("| %s | - | - | - |" % key)
            continue
        metric = m.get("alpha_relL2_test") if "B" in key else m.get("final_val_loss")
        dur = m.get("_total_train_sec", "-")
        out.append("| %s | %s | %.4f | %s | %s |" % (
            key.split("_")[0], key.split("_")[1],
            m.get("final_val_alpha_loss", m.get("final_val_loss", 0)),
            ("%.4f" % metric) if metric is not None else "-", dur))
    out.append("\n## B 链路两段误差拆解（POD 截断 vs 代理误差）\n")
    out.append("| 任务 | POD r=8 截断误差 | 代理 Cp 误差 | 比值(代理/截断) | 是否放大 |")
    out.append("|---|---|---|---|---|")
    for key, m in summary.items():
        if "B" not in key or m.get("_missing"):
            continue
        trunc = m.get("pod_trunc_relerr_r8", 0)
        cp = m.get("cp_relL2_test", 0)
        ratio = m.get("ratio_agent_vs_pod_trunc", 0)
        amplified = "⚠ 放大" if ratio > 1.0 else "OK"
        out.append("| %s | %.4f | %.4f | %.2f | %s |" % (
            key.split("_")[0], trunc, cp, ratio, amplified))
    # 写文件
    sp = os.path.join(RUNS, "summary_all.json")
    json.dump({"family_counts": fam_counts, "metrics": summary},
              open(sp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    mp = os.path.join(RUNS, "summary_all.md")
    open(mp, "w", encoding="utf-8").write("\n".join(out))
    print("OK_EVAL_UNIFIED -> %s + %s" % (sp, mp))


if __name__ == "__main__":
    main()

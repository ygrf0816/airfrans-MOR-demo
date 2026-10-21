"""S2 目标可达性预检函数。
口径来源 04_results/inversion/01_reachability.md §1.1（已核实）：
5 位族、U∞=62±4 m/s、AoA 4–10° 带内官方 CL 全范围 [0.5604, 1.4225]；
设计点 8±1° 子集 [0.8787, 1.1464]。
"""
import csv
import numpy as np

# 边界常数（来源 01_reachability.md §1.1，已核实）
CL_MIN = 0.5604   # 带内最小
CL_MAX = 1.4225   # 带内最大
DSUB_MIN = 0.8787  # 设计点 8±1° 子集最小
DSUB_MAX = 1.1464  # 设计点 8±1° 子集最大

def target_precheck(target_cl: float):
    """返回 (verdict, in_design_subset, note)。"""
    in_band = CL_MIN < target_cl < CL_MAX
    in_ds   = DSUB_MIN <= target_cl <= DSUB_MAX
    if target_cl <= CL_MIN:
        verdict = "不可达"
        note    = f"CL={target_cl:.4f} ≤ 带下界 {CL_MIN}，超出数据库覆盖"
    elif target_cl >= CL_MAX:
        verdict = "不可达"
        note    = f"CL={target_cl:.4f} ≥ 带上界 {CL_MAX}，超出数据库覆盖"
    else:
        verdict = "可达"
        note    = f"CL={target_cl:.4f} 落在带内 ({CL_MIN},{CL_MAX})" + \
                  ("；在设计点子集内" if in_ds else "；在设计点子集外")
    return verdict, in_ds, note

# 代表性目标（_next_task.md 指定）
targets = [0.4, 0.5, 0.5604, 0.7, 0.9, 1.1464, 1.3, 1.4225, 1.5]

if __name__ == "__main__":
    import os
    OUT = "04_results/inversion/target_precheck.csv"
    rows = []
    for t in targets:
        v, in_ds, note = target_precheck(t)
        rows.append([f"{t:.4f}", v, "是" if in_ds else "否", note])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_cl", "verdict", "in_design_subset", "note"])
        w.writerows(rows)
    print(f"written {OUT}, {len(rows)} rows")
    for r in rows:
        print(r)

"""汇总 B 链路场预测的三种口径对照，产出给报告直接引用的一个 json + 一张控制台表。

三种口径必须分清（这是之前最大的坑）：
  A_official    A 链路按官方 Infer_test 协议（多轮 32k 子采样 + 重叠节点预测平均）
                的误差。这是论文/官方仓库记录的那个数，A 因此"便宜"了 13~27 倍。
  A_single      A 链路在同一协议下**只跑一轮**（不做多轮平均）。与 B 同口径可比。
  B_grid        B 场链路（256x128 均匀笛卡尔网格表示）全网格单遍。
  B_bodyfit     B 场链路（贴体近壁 (s,d) 带 + 笛卡尔远场 双区表示）全网格单遍。
  B_bodyfit_floor  该双区表示的下限：真值目标走同一条回读路径。模型不可能低于它。
                用它才能判断误差是"表示装不下"还是"模型没学好"。

用法：python b_field_bodyfit_compare.py [--runs <runs 目录>] [--out <json 路径>]
"""
import os, json, argparse
import numpy as np

POD = "/home/user/airfrans/pod"
TASKS = ["full", "scarce", "reynolds", "aoa"]
FIELDS = ["u", "v", "p", "nu_t"]


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/user/airfrans/prediction/04_results/prediction/runs")
    ap.add_argument("--out", default="/home/user/airfrans/prediction/04_results/prediction/"
                                     "B_field_bodyfit_comparison.json")
    ap.add_argument("--protocol", default=os.path.join(POD, "protocol_aligned_comparison.json"))
    args = ap.parse_args()

    proto = load(args.protocol) or {}
    out = {
        "note": (
            "A_official 是官方 Infer_test 协议（多轮 32k 子采样 + 重叠预测平均）下的误差；"
            "A_single 是同一模型只跑一轮的误差，与 B 的口径一致。B 的场预测是确定性的，"
            "单遍即最终结果，没有可平均的多轮。因此把 B 与 A_official 直接并列会系统性低估 A。"),
        "protocol_caveat": (
            "B_bodyfit_floor 是双区表示自身的下限（真值目标走同一条插值回读），"
            "B_bodyfit 与它的差就是模型误差；两者接近说明瓶颈在表示，差距大说明在模型。"),
        "per_task": {},
    }

    print("%-9s %11s %11s %11s %11s %11s %11s %9s" %
          ("task", "A_官方协议", "A_单轮32k", "B_网格", "网格下限", "B_贴体", "贴体下限",
           "B/A单轮"))
    for t in TASKS:
        row = {}
        p = proto.get(t, {})
        row["A_official_protocol_x1e-2"] = p.get("A_official_protocol_x1e-2")
        row["A_single_round_x1e-2"] = p.get("A_single_round_32k_x1e-2")
        g = load(os.path.join(args.runs, "%s_B_field" % t, "eval_metrics.json"))
        b = load(os.path.join(args.runs, "%s_B_field_bf" % t, "eval_metrics.json"))
        if g:
            row["B_grid_mesh_x1e-2"] = g.get("field_MSE_mesh_mean_x1e-2")
            row["B_grid_per_field_x1e-2"] = g.get("field_MSE_mesh_x1e-2")
            row["B_grid_floor_x1e-2"] = g.get("floor_MSE_mesh_mean_x1e-2")
            row["B_grid_floor_per_field_x1e-2"] = g.get("floor_MSE_mesh_x1e-2")
            row["B_grid_grid_x1e-2"] = g.get("field_MSE_grid_mean_x1e-2")
        if b:
            row["B_bodyfit_mesh_x1e-2"] = b.get("field_MSE_mesh_mean_x1e-2")
            row["B_bodyfit_per_field_x1e-2"] = b.get("field_MSE_mesh_x1e-2")
            row["B_bodyfit_floor_x1e-2"] = b.get("floor_MSE_mesh_mean_x1e-2")
            row["B_bodyfit_floor_per_field_x1e-2"] = b.get("floor_MSE_mesh_x1e-2")
            row["B_bodyfit_near_band_x1e-2"] = b.get("field_MSE_near_band_x1e-2")
            row["B_bodyfit_floor_near_band_x1e-2"] = b.get("floor_MSE_near_band_mean_x1e-2")
            row["B_bodyfit_grid_x1e-2"] = b.get("field_MSE_grid_mean_x1e-2")
            row["B_bodyfit_coverage"] = b.get("mesh_eval_coverage")
            row["n_epochs"] = b.get("n_epochs_run")
        a1 = row.get("A_single_round_x1e-2")
        bb = row.get("B_bodyfit_mesh_x1e-2")
        row["B_bodyfit_over_A_single"] = (bb / a1) if (a1 and bb) else None
        bg = row.get("B_grid_mesh_x1e-2")
        row["B_bodyfit_over_B_grid"] = (bb / bg) if (bg and bb) else None
        out["per_task"][t] = row

        def f(v):
            return ("%.2f" % v) if isinstance(v, (int, float)) else "-"
        print("%-9s %11s %11s %11s %11s %11s %11s %9s" %
              (t, f(row.get("A_official_protocol_x1e-2")), f(a1), f(bg),
               f(row.get("B_grid_floor_x1e-2")), f(bb),
               f(row.get("B_bodyfit_floor_x1e-2")), f(row.get("B_bodyfit_over_A_single"))))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print("\n写出", args.out)


if __name__ == "__main__":
    main()

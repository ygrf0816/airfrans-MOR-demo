# -*- coding: utf-8 -*-
"""把 04_results/explain 的落盘结果压成网页用的 explain.json。

页面只做展示，所有数字都在这里算好，不在浏览器里重算，
避免同一份结论在两处各算一遍而悄悄分叉。
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(DEMO))
EXPLAIN = os.path.join(ROOT, "04_results", "explain")
PUBLIC = os.path.join(DEMO, "app", "public", "data")

COMPONENTS = ["yc_mid", "th_peak_half", "th_max_xc", "re_radius_c"]
FAMILIES = ["4digit", "5digit"]

LABELS = {
    "yc_mid": "弯度中点",
    "th_peak_half": "最大半厚",
    "th_max_xc": "最大厚度位置",
    "re_radius_c": "前缘半径",
}
SEG_LABELS = {"front": "前缘段", "mid": "中弦段", "rear": "后缘段"}


def read_csv(name):
    path = os.path.join(EXPLAIN, name)
    if not os.path.exists(path):
        raise SystemExit("缺少输入文件 " + path)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def r6(v):
    """六位有效数字，去掉浮点尾巴，JSON 体积可控。"""
    if v is None:
        return None
    f = float(v)
    if not np.isfinite(f):
        return None
    if f == 0:
        return 0.0
    return float(f"{f:.6g}")


def ols(x, y):
    """一元最小二乘，返回斜率、截距、R²。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    A = np.column_stack([x, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coef[0]), float(coef[1]), r2


def partial_coef(target, names, controls):
    """target 对 names 的偏回归系数（先对其余控制量回归取残差，再回归）。"""
    y = np.asarray(target, float)
    X = np.column_stack([np.asarray(names, float), np.asarray(controls, float).T])
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid_y = y - pred + coef[0] * X[:, 0]
    # 控制变量与常数上的残差
    C = np.column_stack([X[:, 1:], np.ones(len(y))])
    ccoef, *_ = np.linalg.lstsq(C, y, rcond=None)
    resid = y - C @ ccoef
    x0 = X[:, 0]
    x0c, *_ = np.linalg.lstsq(C, x0, rcond=None)
    x0r = x0 - C @ x0c
    slope = float((x0r @ resid) / (x0r @ x0r))
    # 偏相关
    pr = float(np.corrcoef(x0r, resid)[0, 1])
    return slope, pr, resid_y


def build_global():
    rows = read_csv("sensitivity_data_side.csv")
    data = {fam: {c: [] for c in COMPONENTS} for fam in FAMILIES}
    counts = {}
    for r in rows:
        fam = r["fam"]
        comp = r["comp"]
        if fam in data and comp in data[fam]:
            data[fam][comp].append(float(r["coef_std"]))
            counts[fam] = int(r["n"])
    model_rows = read_csv("sensitivity_model_side.csv")
    model = {fam: {c: [] for c in COMPONENTS} for fam in FAMILIES}
    for r in model_rows:
        fam = r["fam"]
        comp = r["comp"]
        if fam in model and comp in model[fam]:
            model[fam][comp].append(float(r["pod_disp"]))
    out = {"components": COMPONENTS, "labels": LABELS, "families": {}}
    for fam in FAMILIES:
        out["families"][fam] = {
            "n": counts.get(fam),
            "data": [r6(np.mean(data[fam][c])) for c in COMPONENTS],
            "model": [r6(np.mean(model[fam][c])) for c in COMPONENTS],
            "modelAmp": [r6(np.mean([abs(x) for x in model[fam][c]])) for c in COMPONENTS],
        }
    return out


def build_local():
    rows = read_csv("local_sensitivity_256_fixed.csv")
    xs = {}
    for fam in FAMILIES:
        sel = [r for r in rows if r["family"] == fam]
        sel.sort(key=lambda r: int(r["station_idx"]))
        xs[fam] = [r6(r["x_half"]) for r in sel]
        out = {}
        for c in COMPONENTS:
            out[c] = {
                "data": [r6(r["stdcoef_" + c]) for r in sel],
                "model": [r6(r["model_dcp_" + c]) for r in sel],
            }
        xs[fam] = {"x": xs[fam], "curves": out, "n": len(sel)}
    verdict_rows = read_csv("local_sensitivity_256_fixed_verdict.csv")
    verdict = []
    for r in verdict_rows:
        verdict.append({
            "family": r["family"],
            "comp": r["comp"],
            "dataDom": r["data_dom"],
            "modelDom": r["model_dom"],
            "dataDomLabel": SEG_LABELS.get(r["data_dom"], r["data_dom"]),
            "modelDomLabel": SEG_LABELS.get(r["model_dom"], r["model_dom"]),
            "agree": r["agree"].strip().lower() == "true",
        })
    return {"labels": LABELS, "families": xs, "verdict": verdict,
            "segLabels": SEG_LABELS}


def build_interaction():
    rows = read_csv("interaction_data_side.csv")
    out = []
    for r in rows:
        ratio = float(r["abs_over_main_max"])
        t = float(r["t"])
        out.append({
            "family": r["family"],
            "response": r["response"],
            "term": r["interaction"].replace("x", " × ").replace("  ×  ", " × "),
            "coef": r6(r["std_coef"]),
            "t": r6(t),
            "ratio": r6(ratio),
            # 判据与报告一致：系数不到主效应最大者的 10% 且 |t|<2 才算可忽略
            "negligible": abs(ratio) < 0.10 and abs(t) < 2.0,
            "n": int(r["n"]),
        })
    out.sort(key=lambda d: -abs(d["coef"] or 0))
    return out


def build_theory():
    slopes = {r["family"]: r for r in read_csv("theory_check_slopes.csv")}
    cl_rows = read_csv("model_side_cl.csv")
    model_slope = {}
    for fam in FAMILIES:
        sel = [r for r in cl_rows if r["family"] == fam]
        s, b, r2 = ols([float(r["AoA"]) for r in sel], [float(r["CL_model"]) for r in sel])
        model_slope[fam] = {
            "n": len(sel), "slope": r6(s * 180.0 / np.pi), "slopePerRad": r6(s),
            "r2": r6(r2),
        }

    sp_data = {r["name"]: r for r in read_csv("theory_check_suction_peak.csv")}
    sp_model = {r["name"]: r for r in read_csv("model_side_suction_peak.csv")}
    suction = {}
    for fam in FAMILIES:
        dsel = [r for r in sp_data.values() if r["family"] == fam]
        s, b, r2 = ols([float(r["AoA"]) for r in dsel],
                       [float(r["peak_x_chord"]) for r in dsel])
        msel = [r for r in sp_model.values() if r["family"] == fam]
        ms, mb, mr2 = ols([float(r["AoA"]) for r in msel],
                          [float(r["peak_x_chord_model"]) for r in msel])
        suction[fam] = {
            "dataSlope": r6(s), "dataR2": r6(r2), "nData": len(dsel),
            "modelSlope": r6(ms), "modelR2": r6(mr2), "nModel": len(msel),
        }

    te_rows = read_csv("theory_check_te_recovery.csv")
    te = {}
    for fam in FAMILIES:
        sel = [r for r in te_rows if r["family"] == fam]
        y = [float(r["cp_te"]) for r in sel]
        x = [float(r["th_peak_half"]) for r in sel]
        ctrl = [[float(r["yc_mid"]) for r in sel],
                [float(r["th_max_xc"]) for r in sel],
                [float(r["re_radius_c"]) for r in sel]]
        slope, pr, _ = partial_coef(y, x, ctrl)
        simple = float(np.corrcoef(x, y)[0, 1])
        te[fam] = {"n": len(sel), "coef": r6(slope), "r": r6(simple),
                   "rPartial": r6(pr), "cpTeMean": r6(np.mean(y))}

    te_model_rows = read_csv("model_side_te_recovery.csv")
    for fam in FAMILIES:
        sel = [r for r in te_model_rows if r["family"] == fam]
        half = [(float(r["d_cp_te_plus"]) - float(r["d_cp_te_minus"])) / 2.0 for r in sel]
        te[fam]["modelHalf"] = r6(np.mean(half))
        te[fam]["nModel"] = len(sel)

    return {
        "slope": {
            "theoryThin": r6(slopes["4digit"]["theory_thin2pi"]),
            "theoryPgRef": r6(slopes["4digit"]["theory_pg_ref"]),
            "bandLo": r6(slopes["4digit"]["theory_band_lo"]),
            "bandHi": r6(slopes["4digit"]["theory_band_hi"]),
            "bandNote": "薄翼 2π 再乘 Prandtl–Glauert 并计入粘性压低后的期望区间",
            "families": {fam: {
                "n": int(slopes[fam]["n"]),
                "raw": r6(slopes[fam]["raw_slope_per_rad"]),
                "rawR2": r6(slopes[fam]["raw_r2"]),
                "ctrl": r6(slopes[fam]["ctrl_slope_per_rad"]),
                "ctrlSe": r6(slopes[fam]["ctrl_se"]),
                "model": model_slope[fam]["slope"],
                "modelR2": model_slope[fam]["r2"],
                "modelN": model_slope[fam]["n"],
            } for fam in FAMILIES},
        },
        "suction": suction,
        "te": te,
    }


def parse_name(name):
    """airFoil2D_SST_{uinf}_{aoa}_{...} 里取出 uinf 与 aoa。"""
    parts = name.split("_")
    try:
        return float(parts[2]), float(parts[3])
    except (IndexError, ValueError):
        return None, None


def build_scatter():
    with open(os.path.join(PUBLIC, "samples.json"), encoding="utf-8") as fh:
        samples = json.load(fh)["items"]
    comp_rows = {r["name"]: r for r in read_csv("geom_decomp_1000.csv")}
    fam, aoa, uinf, cl, cd = [], [], [], [], []
    vals = {c: [] for c in COMPONENTS}
    skipped = 0
    pool = []
    for s in samples:
        g = comp_rows.get(s["name"])
        if g is None:
            skipped += 1
            continue
        fam.append(0 if s["series"] == "4digit" else 1)
        aoa.append(r6(s["aoa"]))
        uinf.append(r6(s["uInf"]))
        cl.append(r6(s["clOfficial"]))
        cd.append(r6(s["cdOfficial"]))
        for c in COMPONENTS:
            vals[c].append(r6(g[c]))
        if (s["series"] == "5digit" and abs(s["uInf"] - 62.0) <= 4.0
                and 4.0 <= s["aoa"] <= 10.0):
            pool.append(s["clOfficial"])
    return {
        "n": len(fam), "skipped": skipped,
        "family": fam, "aoa": aoa, "uinf": uinf, "cl": cl, "cd": cd,
        "components": COMPONENTS, "labels": LABELS,
        "values": vals,
        "band": {"lo": r6(min(pool)), "hi": r6(max(pool)), "n": len(pool)}
        if pool else None,
    }


def build_top10():
    rows = read_csv("design_top10.csv")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "rank": i + 1,
            "name": r["name"],
            "family": r["family"],
            "uinf": r6(r["Uinf"]),
            "aoa": r6(r["AoA"]),
            "cl": r6(r["CL"]),
            "cd": r6(r["CD"]),
            "mLocal": r6(r["M_local"]),
            "ycMid": r6(r["yc_mid"]),
            "thPeakHalf": r6(r["th_peak_half"]),
            "thMaxXc": r6(r["th_max_xc"]),
            "reRadius": r6(r["re_radius_c"]),
        })
    dev = []
    for r in read_csv("design_comp_compare.csv"):
        dev.append({
            "comp": r["comp"],
            "label": LABELS.get(r["comp"], r["comp"]),
            "poolMedian": r6(r["pool_median"]),
            "top10Mean": r6(r["top10_mean"]),
            "devMean": r6(r["dev_mean"]),
        })
    return {"rows": out, "dev": dev}


def main():
    out = {
        "source": "04_results/explain",
        "global": build_global(),
        "local": build_local(),
        "interaction": build_interaction(),
        "theory": build_theory(),
        "scatter": build_scatter(),
        "design": build_top10(),
    }
    os.makedirs(PUBLIC, exist_ok=True)
    dest = os.path.join(PUBLIC, "explain.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(dest)
    print("写出 " + dest)
    print("体积 %.1f KB" % (size / 1024.0))
    print("散点样本 %d 条，未匹配 %d 条" % (out["scatter"]["n"], out["scatter"]["skipped"]))
    g = out["global"]["families"]
    for fam in FAMILIES:
        print("%s 数据侧 %s" % (fam, g[fam]["data"]))
        print("%s 模型侧 %s" % (fam, g[fam]["model"]))
    print("一致组合 %d / %d" % (
        sum(1 for v in out["local"]["verdict"] if v["agree"]),
        len(out["local"]["verdict"])))
    th = out["theory"]
    for fam in FAMILIES:
        s = th["slope"]["families"][fam]
        print("%s 斜率 实测 %s 模型 %s (R² %s)" % (fam, s["ctrl"], s["model"], s["modelR2"]))
        print("%s 吸力峰 数据 %s 模型 %s" % (fam, th["suction"][fam]["dataSlope"],
                                          th["suction"][fam]["modelSlope"]))
        print("%s 后缘恢复 数据 %s 模型 %s" % (fam, th["te"][fam]["coef"],
                                          th["te"][fam]["modelHalf"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

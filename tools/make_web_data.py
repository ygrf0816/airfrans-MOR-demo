# -*- coding: utf-8 -*-
"""把本地已验收的产物汇总成网页资产，写到 app/public/data/。

本地跑，不需要 torch，也不需要工作站。只有远端导出的
cp_basis_meta.json 与 samples_alpha_field.* 需要先从 data_remote 取回。

用法：
  python 04_results/demo/tools/make_web_data.py
  python 04_results/demo/tools/make_web_data.py --skip-models   # 远端产物还没回来时先跑

产物见 04_results/demo/01_DATA_CONTRACT.md 第 1 节。
输出口径的三条硬规矩：不改物理量、归一化统计量随模型走、索引对齐靠名字。
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEMO = os.path.join(ROOT, "04_results", "demo")
REMOTE = os.path.join(DEMO, "data_remote")
OUT = os.path.join(DEMO, "app", "public", "data")
RES = os.path.join(ROOT, "04_results", "inversion", "results")
PRED = os.path.join(ROOT, "04_results", "prediction")
PAPER = os.path.join(ROOT, "04_results", "paper")

FIELDS = ["u", "v", "p", "nu_t"]
FIELD_UNIT = {"u": "m/s", "v": "m/s", "p": "m^2/s^2", "nu_t": "m^2/s"}
CASE_LABEL = {
    "case1_success": ("成功案例", "三项约束全部满足，升力也高"),
    "case2_boundary": ("边界案例", "分离代理正好压在约束线上，没有余量"),
    "case3_fail_mach": ("失败案例", "局部马赫代理超过上限"),
    "case4_fail_robust": ("对照案例", "约束满足，但把参数推到边界附近"),
    "case5_interior": ("内部案例", "设计空间内部的一个普通可行解"),
    "db_designbest": ("数据库设计点最优", "官方数据库里八度升力最高的样本"),
    "db_bandbest": ("数据库带内最优", "官方数据库里带内损失最小的样本"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--skip-models", action="store_true",
                help="跳过需要远端产物的那几项，先出本地能出的部分")
ap.add_argument("--no-figs", action="store_true", help="跳过图片转换")
ap.add_argument("--out", default=None, help="输出目录，默认 app/public/data")
ap.add_argument("--remote", default=None, help="远端产物目录，默认 04_results/demo/data_remote")
args = ap.parse_args()

if args.out:
    OUT = os.path.abspath(args.out)
if args.remote:
    REMOTE = os.path.abspath(args.remote)
os.makedirs(OUT, exist_ok=True)
report = {"written": [], "skipped": [], "notes": []}


def w(name):
    p = os.path.join(OUT, name)
    report["written"].append({"file": name, "bytes": int(os.path.getsize(p))})
    print("  %-26s %10d B" % (name, os.path.getsize(p)))


def jdump(name, obj, compact=True):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as fh:
        if compact:
            json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
    w(name)


def read_csv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(v, nd=6):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def b(v):
    return str(v).strip().lower() in ("true", "1", "yes")


# ================= 1 样本与几何 =================
print("== 样本与几何")
splits = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"),
                        encoding="utf-8"))
split_of = {}
for key, tag in (("full_train", "train"), ("scarce_train", "train"),
                 ("reynolds_train", "train"), ("aoa_train", "train"),
                 ("full_test", "test"), ("reynolds_test", "test"), ("aoa_test", "test")):
    for n in splits.get(key, []):
        split_of.setdefault(n, tag)

wc = np.load(os.path.join(ROOT, "09_tmp", "data", "wallcp_1000.npz"), allow_pickle=True)
CP = np.asarray(wc["CP"], np.float64)
names = [str(x) for x in wc["names"]]
gv = np.load(os.path.join(ROOT, "09_tmp", "data", "geomvec_1000.npz"), allow_pickle=True)
G = np.asarray(gv["G"], np.float64)
gv_names = [str(x) for x in gv["names"]]
oc = np.load(os.path.join(PRED, "official_coefs_1000.npz"), allow_pickle=True)
oc_names = [str(x) for x in oc["names"]]
CL = np.asarray(oc["CL"], np.float64)
CD = np.asarray(oc["CD"], np.float64)

assert set(names) == set(gv_names) == set(oc_names), "三个 npz 的样本名集合不一致"
gi = {n: i for i, n in enumerate(gv_names)}
oi = {n: i for i, n in enumerate(oc_names)}
assert CP.shape == (len(names), 256), CP.shape
assert G.shape == (len(names), 128), G.shape


def parse_name(n):
    p = n.replace("airFoil2D_SST_", "").split("_")
    u_inf, aoa = float(p[0]), float(p[1])
    rest = [float(x) for x in p[2:]]
    if len(rest) == 4:
        L, P, Q, T = rest
        return dict(uInf=u_inf, aoa=aoa, series="5digit", L=L, P=P, Q=int(round(Q)), T=T, M=None)
    M, P, T = rest
    return dict(uInf=u_inf, aoa=aoa, series="4digit", L=None, P=P, Q=0, T=T, M=M)


parsed = [parse_name(n) for n in names]


def file_facts(name):
    """读一条样本的两个原始文件，取字节数与点数。

    降维演示要展示"原始样本多大、压成多少"，所以这两个数必须是真的。
    点数写在 VTK XML 的 Piece 标签里，靠前，读文件头几 KB 就够，
    不必把几百 MB 的 vtu 整个读进来。
    """
    d = os.path.join(ROOT, "02_dataset", "Dataset", name)
    out = {"bytesVol": None, "nVol": None, "bytesSurf": None, "nSurf": None}
    for key, suffix, bkey, nkey in (("vtu", "_internal.vtu", "bytesVol", "nVol"),
                                    ("vtp", "_aerofoil.vtp", "bytesSurf", "nSurf")):
        p = os.path.join(d, name + suffix)
        if not os.path.isfile(p):
            continue
        out[bkey] = int(os.path.getsize(p))
        try:
            with open(p, "rb") as fh:
                head = fh.read(8192).decode("utf-8", "replace")
            m = re.search(r'NumberOfPoints="(\d+)"', head)
            if m:
                out[nkey] = int(m.group(1))
        except OSError:
            pass
    return out


facts = {}
for n in names:
    facts[n] = file_facts(n)
have_vol = sum(1 for v in facts.values() if v["bytesVol"])
print("  读到原始文件大小的样本 %d 条（internal.vtu 存在 %d 条）" % (len(facts), have_vol))
if have_vol:
    avg_v = np.mean([v["bytesVol"] for v in facts.values() if v["bytesVol"]])
    avg_s = np.mean([v["bytesSurf"] for v in facts.values() if v["bytesSurf"]])
    print("  平均 内部体积文件 %.2f MB，表面文件 %.2f MB" % (avg_v / 1048576, avg_s / 1048576))

items = []
for i, n in enumerate(names):
    d = dict(parsed[i])
    d.update({"i": i, "name": n, "clOfficial": f(CL[oi[n]], 5), "cdOfficial": f(CD[oi[n]], 6),
              "split": split_of.get(n, "unknown")})
    d.update(facts[n])
    items.append(d)
n_four = sum(1 for d in parsed if d["series"] == "4digit")
print("  样本 %d，四位族 %d，五位族 %d" % (len(names), n_four, len(names) - n_four))
jdump("samples.json", {
    "count": len(items),
    "fields": ["name", "uInf", "aoa", "series", "L", "P", "Q", "T", "M",
               "clOfficial", "cdOfficial", "split"],
    "note": ("四位族没有 L 与 Q，用 M 表示弯度百分数。P 的单位是弦长的十分之一，"
             "T 是厚弦比的百分数。"),
    "items": items,
})
np.asarray(G, np.float32).tofile(os.path.join(OUT, "samples_geom.bin"))
w("samples_geom.bin")
np.asarray(CP, np.float32).tofile(os.path.join(OUT, "samples_cp.bin"))
w("samples_cp.bin")

# ================= 2 压力曲线基与归一化统计量 =================
print("\n== 压力曲线基")
meta_path = os.path.join(REMOTE, "cp_basis_meta.json")
alpha_cp = None
if os.path.isfile(meta_path):
    cmeta = json.load(open(meta_path, encoding="utf-8"))
    cp_phi = np.asarray(cmeta["cp"]["podPhi"], np.float64)
    cp_mu = np.asarray(cmeta["cp"]["podMu"], np.float64).ravel()
    jdump("cp_basis.json", {
        "podR": int(cmeta["cp"]["podR"]), "nPts": 256,
        "s": [float(v) for v in np.linspace(0, 1, 256)],
        "mu": [f(v, 8) for v in cp_mu],
        "phi": [[f(v, 8) for v in row] for row in cp_phi],
        "smean": [f(v, 8) for v in cmeta["cp"]["smean"]],
        "sstd": [f(v, 8) for v in cmeta["cp"]["sstd"]],
        "gmean": [f(v, 8) for v in cmeta["cp"]["gmean"]],
        "gstd": [f(v, 8) for v in cmeta["cp"]["gstd"]],
        "fieldSmean": [f(v, 8) for v in cmeta["field"]["smean"]],
        "fieldSstd": [f(v, 8) for v in cmeta["field"]["sstd"]],
        "fieldGmean": [f(v, 8) for v in cmeta["field"]["gmean"]],
        "fieldGstd": [f(v, 8) for v in cmeta["field"]["gstd"]],
        "note": ("前缘锚定弧长序，s 等于零在前缘，约零点五到后缘，首尾闭合。"
                 "z-score 是逐维减均值再除标准差。"),
    })
    alpha_cp = (CP - cp_mu) @ cp_phi
    print("  压力曲线基 %s，样本坐标 %s" % (cp_phi.shape, alpha_cp.shape))
else:
    if not args.skip_models:
        print("  缺少 %s，请先跑远端导出并取回" % meta_path)
        sys.exit(2)
    report["skipped"].append("cp_basis.json")
    print("  跳过（缺 cp_basis_meta.json）")

# ================= 3 流场基、网格、样本坐标 =================
print("\n== 流场基与网格")
need = ["field_basis.bin", "field_basis.json", "grid.json", "b_cp.onnx", "b_field.onnx"]
missing = [n for n in need if not os.path.isfile(os.path.join(REMOTE, n))]
alpha_field = None
field_floor = {}
if missing:
    if not args.skip_models:
        print("  远端产物还没回来，缺 %s" % missing)
        sys.exit(2)
    report["skipped"].extend(missing)
    print("  跳过（缺 %s）" % missing)
else:
    for n in need:
        shutil.copyfile(os.path.join(REMOTE, n), os.path.join(OUT, n))
        w(n)
    fmeta = json.load(open(os.path.join(REMOTE, "field_basis.json"), encoding="utf-8"))
    af_path = os.path.join(REMOTE, "samples_alpha_field.bin")
    af_json = os.path.join(REMOTE, "samples_alpha_field.json")
    if os.path.isfile(af_path) and os.path.isfile(af_json):
        aj = json.load(open(af_json, encoding="utf-8"))
        A = np.fromfile(af_path, dtype=np.float32).reshape(len(aj["names"]), -1)
        row = {n: i for i, n in enumerate(aj["names"])}
        alpha_field = A[[row[n] for n in names]]
        print("  流场样本坐标 %s（已按名字重排）" % (alpha_field.shape,))
        if aj.get("truncationFloor"):
            field_floor = {k: {kk: v[kk] for kk in ("relL2Mean", "relL2Median", "relL2P95",
                                                    "ampRatio")}
                           for k, v in aj["truncationFloor"].items()}
            print("  截断地板（32 个模态）：" + "  ".join(
                "%s %.4f" % (k, v["relL2Mean"]) for k, v in field_floor.items()))
    else:
        report["notes"].append("缺少 samples_alpha_field，样本坐标里流场那部分留零")
        print("  缺 samples_alpha_field，流场坐标留零")
    if alpha_cp is not None:
        af = (alpha_field if alpha_field is not None
              else np.zeros((len(names), 4 * int(fmeta["podR"])), np.float32))
        alpha = np.concatenate([alpha_cp.astype(np.float32), af.astype(np.float32)], axis=1)
        assert alpha.shape == (len(names), 8 + 4 * int(fmeta["podR"])), alpha.shape
        alpha.tofile(os.path.join(OUT, "samples_alpha.bin"))
        w("samples_alpha.bin")

# ================= 4 案例流场 =================
print("\n== 案例流场")
cf_path = os.path.join(RES, "cases_field.npz")
if os.path.isfile(cf_path):
    cf = np.load(cf_path, allow_pickle=True)
    cmeta2 = json.loads(str(cf["meta"]))
    mask = np.asarray(cf["mask"], bool).reshape(128, 256)
    n_valid = int(mask.sum())
    blocks, chunks, total = [], [], 0
    for code, cinfo in cmeta2["cases"].items():
        for aoa in cinfo["aoas"]:
            for fn in FIELDS:
                key = "%s_%s_%d" % (fn, code, int(aoa))
                if key not in cf.files:
                    raise KeyError("cases_field.npz 缺 %s" % key)
                vals = np.asarray(cf[key], np.float32).reshape(128, 256)[mask]
                assert vals.size == n_valid
                chunks.append(vals)
                blocks.append({"key": key, "field": fn, "case": code, "aoa": int(aoa),
                               "offset": total, "count": n_valid})
                total += n_valid
    blob = np.concatenate(chunks).astype(np.float32)
    blob.tofile(os.path.join(OUT, "cases_field.bin"))
    w("cases_field.bin")
    cases_meta = []
    for code, cinfo in cmeta2["cases"].items():
        label, reason = CASE_LABEL.get(code, (code, ""))
        cases_meta.append({"code": code, "label": label, "reason": reason,
                           "L": f(cinfo["L"], 6), "P": f(cinfo["P"], 6),
                           "Q": int(cinfo["Q"]), "T": f(cinfo["T"], 6),
                           "uInf": cinfo["u_inf"], "aoas": [int(a) for a in cinfo["aoas"]]})
    jdump("cases_field.json", {
        "cases": cases_meta, "blocks": blocks, "nGrid": n_valid,
        "totalFloats": int(blob.size), "channelOrder": FIELDS,
        "unit": {k: FIELD_UNIT[k] for k in FIELDS},
        "maskOrder": "行优先 ny 乘 nx，与 grid.json 的掩膜一致",
        "note": "每个块是 nGrid 个 float32，无效格前端不填，单位与论文图一致。",
    })
    print("  %d 个块，共 %d 个数" % (len(blocks), blob.size))
else:
    report["skipped"].append("cases_field.bin")
    print("  缺少 cases_field.npz")

# ================= 5 results.json =================
print("\n== results.json")
res = {}

aoas = np.array([d["aoa"] for d in parsed])
re_ = np.array([d["uInf"] for d in parsed]) / 1.56e-5
u_inf = np.array([d["uInf"] for d in parsed])
ah, ae = np.histogram(aoas, bins=28)
rh, re_edges = np.histogram(re_, bins=28)
res["dataset"] = {
    "nTotal": len(names),
    "nTrain": len(splits.get("full_train", [])), "nTest": len(splits.get("full_test", [])),
    "nFourDigit": int(n_four), "nFiveDigit": int(len(names) - n_four),
    "reRange": [float(re_.min()), float(re_.max())],
    "aoaRange": [float(aoas.min()), float(aoas.max())],
    "uInfRange": [float(u_inf.min()), float(u_inf.max())],
    "aoaHist": {"binEdges": [f(v, 4) for v in ae], "counts": [int(v) for v in ah]},
    "reHist": {"binEdges": [f(v, 1) for v in re_edges], "counts": [int(v) for v in rh]},
    "splitCards": [
        {"name": "full", "label": "大数据划分", "train": len(splits.get("full_train", [])),
         "test": len(splits.get("full_test", []))},
        {"name": "scarce", "label": "少数据划分", "train": len(splits.get("scarce_train", [])),
         "test": len(splits.get("full_test", []))},
        {"name": "reynolds", "label": "按雷诺数划分",
         "train": len(splits.get("reynolds_train", [])),
         "test": len(splits.get("reynolds_test", []))},
        {"name": "aoa", "label": "按攻角划分", "train": len(splits.get("aoa_train", [])),
         "test": len(splits.get("aoa_test", []))},
    ],
    "cpStats": {"s": [f(v, 5) for v in np.linspace(0, 1, 256)],
                "p05": [f(v, 4) for v in np.percentile(CP, 5, axis=0)],
                "p50": [f(v, 4) for v in np.percentile(CP, 50, axis=0)],
                "p95": [f(v, 4) for v in np.percentile(CP, 95, axis=0)]},
    "note": "雷诺数按弦长一米与运动黏度 1.56e-5 换算，攻角与来流速度取自样本文件名。",
}


def load_json(p):
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


mb = load_json(os.path.join(PRED, "runs_full_B_official_eval_metrics.json"))
mfd = load_json(os.path.join(PRED, "runs_full_B_field_eval_metrics.json"))
ma = load_json(os.path.join(PRED, "runs_full_A_eval_metrics.json"))
bench = load_json(os.path.join(PRED, "benchmark_official.json"))
res["modelMetrics"] = {
    "bCp": {
        "label": "压力曲线代理模型", "relL2Test": f(mb.get("cp_relL2_test"), 4),
        "relL2MeanField": f(mb.get("cp_relL2_meanfield_test"), 4),
        "relL2Rank8Trunc": f(mb.get("cp_relL2_rank8_trunc_test"), 4),
        "relL2Ridge": f(mb.get("cp_relL2_ridge_test"), 4),
        "spearmanMinCp": f(mb.get("cp_spearman_min_cp_test"), 4),
        "clRelErrPred": f(mb.get("C_L_relerr_pred_test"), 4),
        "clSpearman": f(mb.get("C_L_spearman_pred"), 4),
        "clFormulaFidelity": f(mb.get("C_L_formula_fidelity_trueCp"), 4),
        "podR": mb.get("pod_r"), "nParam": 133 * 256 + 4 * 256 * 256 + 8 * 256,
        "trainN": mb.get("train_n_eff"), "testN": mb.get("test_n"),
    },
    "bField": {
        "label": "流场代理模型",
        "mseGridMean": f(mfd.get("field_MSE_grid_mean_x1e-2"), 4),
        "mseMeshMean": f(mfd.get("field_MSE_mesh_mean_x1e-2"), 4),
        "floorMseMean": f(mfd.get("floor_MSE_mesh_mean_x1e-2"), 4),
        "truncMseGrid": f(mfd.get("field_MSE_grid_rankr_trunc_x1e-2"), 4),
        "meanFieldMseGrid": f(mfd.get("field_MSE_grid_meanfield_x1e-2"), 4),
        "ridgeMseGrid": f(mfd.get("field_MSE_grid_ridge_x1e-2"), 4),
        "podTruncRelL2Mean": f(mfd.get("pod_trunc_energy_relL2_mean"), 4),
        "truncNote": ("截断地板是把真值投影到 32 个模态再还原的误差，与模型无关。"
                      "模型误差除以它，才是比基本身差多少的倍数。"),
        "mseGridPerField": {k: f(v, 4) for k, v in (mfd.get("field_MSE_grid_x1e-2") or {}).items()},
        "relL2GridPerField": {k: f(v, 4) for k, v in (mfd.get("field_relL2_grid") or {}).items()},
        "spearmanGridPerField": {k: f(v, 4)
                                 for k, v in (mfd.get("field_spearman_grid") or {}).items()},
        "podR": mfd.get("pod_r"), "nGrid": mfd.get("n_grid_cells"),
        "trainN": mfd.get("train_n_eff"), "testN": mfd.get("test_n"),
        "truncationFloorPerField": field_floor or None,
        "truncationFloorNote": ("把真值投影到 32 个模态再还原的相对 L2，逐样本在 1000 条上统计。"
                               "只反映基本身的表示能力，与模型无关。"),
    },
    "aGnn": {
        "label": "官方逐点图网络对照",
        "testVolumeMseMean": f(ma.get("test_volume_MSE_mean_x1e-2"), 4),
        "testSurfacePMse": f(ma.get("test_surface_p_MSE_x1e-1"), 4),
        "relL2PerField": {k: f(v, 4) for k, v in (ma.get("test_relL2_per_field") or {}).items()},
        "nParam": ma.get("params"), "nTrain": ma.get("train_samples"),
        "caveat": ma.get("test_eval_protocol"),
    },
    "notes": [mb.get("note", ""), mfd.get("domain_caveat", "")],
}
res["officialBenchmark"] = {
    "baselineFull": bench.get("baseline_full"),
    "graphSage4Task": bench.get("graph_sage_4task"),
    "note": ("并列时注意两个口径的网格不是同一个。A 链路在约十八万个 CFD 网格节点上算，"
             "B 链路在 256 乘 128 公共网格的三万一千零六十六个有效格上算，"
             "只能比同口径的排序。"),
}
res["objective"] = {
    "clTargetAoa": 8.0, "bandAoas": [4.0, 8.0, 10.0], "uInf": 62.0,
    "machLimit": 0.6, "sepLimit": 0.007874, "dbDesignBest": 1.146,
    "bandUpperBound": 1.4225,
    "sepLimitNote": "取 511 个五位族样本分离代理的九十分位，正好等于一个离散热度步长",
    "note": ("带内损失取八度升力减四度升力与八度升力减十度升力中的大者。"
             "局部马赫代理取四度八度十度三个状态里的最大值。"),
}

fronts = []
for tag in ("s42", "s7", "s2024"):
    p = os.path.join(RES, "ga_front_%s.csv" % tag)
    if not os.path.isfile(p):
        continue
    rows = read_csv(p)
    npz = os.path.join(RES, "ga_front_%s.npz" % tag)
    mmax = sep = None
    if os.path.isfile(npz):
        z = np.load(npz, allow_pickle=True)
        mmax = np.asarray(z["m_max"], float).ravel()
        sep = np.asarray(z["sep"], float).ravel()
        assert len(mmax) == len(rows), "%s 行数 %d 与 npz %d 不一致" % (tag, len(rows), len(mmax))
    for i, r in enumerate(rows):
        cl4, cl8, cl10 = float(r["CL4"]), float(r["CL8"]), float(r["CL10"])
        fronts.append({
            "seed": int(tag[1:]), "idx": int(r["idx"]), "rank": int(r["rank"]),
            "feasible": b(r["feasible"]), "violation": f(r["violation"], 5),
            "L": f(r["L"]), "P": f(r["P"]), "Q": int(float(r["Q"])), "T": f(r["T"]),
            "cl4": f(cl4), "cl8": f(cl8), "cl10": f(cl10),
            "delta": f(max(cl8 - cl4, cl8 - cl10), 6),
            "machMax": f(mmax[i], 6) if mmax is not None else None,
            "sepMax": f(sep[i], 6) if sep is not None else None,
        })
res["fronts"] = fronts
print("  前沿成员 %d 个" % len(fronts))

hv = {}
hp = os.path.join(RES, "hv_corrected.csv")
if os.path.isfile(hp):
    for r in read_csv(hp):
        hv[(r["tag"], int(r["gen"]))] = f(r["hv_corrected"], 5)
history = []
for tag in ("s42", "s7", "s2024"):
    p = os.path.join(RES, "ga_history_%s.csv" % tag)
    if not os.path.isfile(p):
        continue
    for r in read_csv(p):
        history.append({"seed": int(tag[1:]), "gen": int(r["gen"]),
                        "hypervolume": f(r["hypervolume"], 5),
                        "hvCorrected": hv.get((tag, int(r["gen"]))),
                        "nFeasible": int(r["n_feasible"]), "bestCl8": f(r["best_CL8"]),
                        "bestFeasibleCl8": f(r["best_feasible_CL8"]),
                        "minDeltaFeasible": f(r["min_delta_feasible"]),
                        "frontSize": int(r["front0"])})
res["history"] = history

grid = []
gp = os.path.join(RES, "grid_all_full.csv")
if os.path.isfile(gp):
    for r in read_csv(gp):
        grid.append({"L": f(r["L"], 4), "P": f(r["P"], 4), "Q": int(float(r["Q"])),
                     "T": f(r["T"], 4), "cl4": f(r["CL4"], 4), "cl8": f(r["CL8"], 4),
                     "cl10": f(r["CL10"], 4), "delta": f(r["delta"], 4),
                     "machMax": f(r["m_max"], 4), "sepMax": f(r["sep"], 6),
                     "feasible": b(r["feasible"])})
res["grid"] = grid
print("  网格点 %d，其中可行 %d" % (len(grid), sum(1 for g in grid if g["feasible"])))

cases, case_cp = [], {}
for fname, tag in (("cases_eval_s8.csv", "ga"), ("cases_eval_db.csv", "db")):
    p = os.path.join(RES, fname)
    if not os.path.isfile(p):
        continue
    for r in read_csv(p):
        cases.append({"tag": tag, "case": int(r["case"]), "L": f(r["L"]), "P": f(r["P"]),
                      "Q": int(float(r["Q"])), "T": f(r["T"]), "cl4": f(r["CL4"]),
                      "cl8": f(r["CL8"]), "cl10": f(r["CL10"]), "delta": f(r["delta"]),
                      "machMax": f(r["m_max"]), "sepMax": f(r["sep"], 6),
                      "feasible": b(r["feasible"]), "binding": r["binding"]})
for fname, tag in (("cases_eval_s8.npz", "ga"), ("cases_eval_db.npz", "db")):
    p = os.path.join(RES, fname)
    if not os.path.isfile(p):
        continue
    z = np.load(p, allow_pickle=True)
    for i in range(len(z["params"])):
        case_cp["%s-%d" % (tag, i + 1)] = {
            "cp4": [f(v, 4) for v in np.asarray(z["cp4"][i], float)],
            "cp8": [f(v, 4) for v in np.asarray(z["cp8"][i], float)],
            "cp10": [f(v, 4) for v in np.asarray(z["cp10"][i], float)],
            "geom128": [f(v, 6) for v in np.asarray(z["geometry"][i], float)],
        }
res["cases"] = cases
res["caseCurves"] = case_cp

inv_rows = {}
tp = os.path.join(RES, "s6_targets_all.csv")
if os.path.isfile(tp):
    for r in read_csv(tp):
        inv_rows[r["name"]] = r
inversions = []
for jp in sorted(glob.glob(os.path.join(RES, "invert_*.json"))):
    j = json.load(open(jp, encoding="utf-8"))
    tname = j["target"]
    row = inv_rows.get(tname, {})
    npz = jp.replace(".json", ".npz")
    n_within = n_cand = None
    multi = None
    target_cp = rec_cp = true_cp = []
    if os.path.isfile(npz):
        z = np.load(npz, allow_pickle=True)
        cand = np.asarray(z["params"], float)
        true = np.asarray(z["true_params"], float).ravel()
        cp_rel = np.asarray(z["cp_relL2"], float).ravel()
        n_cand = int(cand.shape[0])
        # 多解性的口径与报告一致：数"压力距离落在最优值百分之若干以内"的候选个数，
        # 不是数"参数落在真值百分之若干以内"。后者在落盘数据上完全不是一个量级，
        # 早期口头转述曾把它说错，这里固定用前者。
        best = float(cp_rel.min())
        ladder = {t: int(np.sum(cp_rel <= best * (1.0 + t))) for t in (0.01, 0.02, 0.05, 0.10)}
        near = cand[cp_rel <= best * 1.01]
        multi = {
            "nCandidates": n_cand,
            "nUniqueCandidates": len({tuple(np.round(r, 3)) for r in cand}),
            "bestCpRelL2": f(best, 4),
            "nWithinCpTol": ladder,
            "tolDefinition": "候选的 cp_relL2 不超过最优 cp_relL2 的多少倍",
            "nWithinTol1pct": ladder[0.01],
            "nUniqueWithinTol1pct": len({tuple(np.round(r, 3)) for r in near}),
            "paramSpreadWithinTol1pct": {
                "L": f(near[:, 0].max() - near[:, 0].min(), 3),
                "P": f(near[:, 1].max() - near[:, 1].min(), 3),
                "Q": int(round(near[:, 2].max() - near[:, 2].min())),
                "T": f(near[:, 3].max() - near[:, 3].min(), 3)},
            "paramSpreadNote": ("百分之一容差下的近优解，四个参数各自的极差。"
                                "数值大说明形状差别很大却给出几乎一样的压力分布。"),
        }
        n_within = ladder[0.01]
        target_cp = [f(v, 4) for v in np.asarray(z["target_cp"], float)]
        rec_cp = [f(v, 4) for v in np.asarray(z["best_cp"], float)]
        true_cp = [f(v, 4) for v in np.asarray(z["cp_at_true"], float)]
    inversions.append({
        "name": tname, "group": row.get("group"),
        "isHeldOut": row.get("group") == "heldout",
        "uInf": f(row.get("u_inf")), "aoa": f(row.get("aoa")),
        "trueParams": {"L": f(row.get("l")), "P": f(row.get("p")),
                       "Q": int(float(row["q"])) if row.get("q") else None,
                       "T": f(row.get("t"))},
        "recoveredParams": {"L": f(j["recovered_params"][0]), "P": f(j["recovered_params"][1]),
                            "Q": int(round(j["recovered_params"][2])),
                            "T": f(j["recovered_params"][3])},
        "cpRelL2": f(j["cp_relL2"], 4),
        "modelFloorRelL2": f(j.get("cp_true_relL2"), 4),
        "clRecovered": f(j.get("CL_recovered"), 4),
        "clOfficial": f(row.get("cl_official"), 4),
        "nCandidates": n_cand, "nParamsWithin1pct": n_within, "multiSolution": multi,
        "targetCp": target_cp, "recoveredCp": rec_cp, "cpAtTrue": true_cp,
    })
res["inversions"] = inversions
print("  反演目标 %d 个，其中留出集 %d 个"
      % (len(inversions), sum(1 for x in inversions if x["isHeldOut"])))

anchor = {}
asp = os.path.join(RES, "anchor_summary.csv")
if os.path.isfile(asp):
    anchor["summary"] = [{"convention": r["convention"], "n": int(r["n"]),
                          "spearman": f(r["spearman"], 4), "r2": f(r["r2"], 4),
                          "biasMeanPct": f(r["bias_mean_pct"], 3),
                          "biasMedianPct": f(r["bias_median_pct"], 3),
                          "mae": f(r["mae"], 4)} for r in read_csv(asp)]
dp = os.path.join(RES, "anchor_deciles.csv")
if os.path.isfile(dp):
    anchor["deciles"] = [{"bin": int(r["bin"]), "clLo": f(r["cl_lo"], 4),
                          "clHi": f(r["cl_hi"], 4), "n": int(r["n"]),
                          "officialMean": f(r["cl_official_mean"], 4),
                          "modelMean": f(r["cl_model_mean"], 4),
                          "biasPct": f(r["bias_mean_pct"], 3)} for r in read_csv(dp)]
npth = os.path.join(RES, "anchor_neighbour.csv")
if os.path.isfile(npth):
    anchor["neighbour"] = [{"rank": int(r["rank"]), "name": r["name"], "uInf": f(r["u_inf"]),
                            "aoa": f(r["aoa"]), "L": f(r["L"]), "P": f(r["P"]),
                            "Q": int(float(r["Q"])), "T": f(r["T"]),
                            "dist": f(r["norm_dist"], 4),
                            "clOfficial": f(r["cl_official"], 4),
                            "clModel": f(r["cl_model"], 4),
                            "relPct": f(r["rel_pct"], 3)} for r in read_csv(npth)]
arp = os.path.join(RES, "anchor_real.npz")
if os.path.isfile(arp):
    z = np.load(arp, allow_pickle=True)
    anchor["scatter"] = [{"name": str(n), "uInf": f(z["u_inf"][i]), "aoa": f(z["aoas"][i]),
                          "clOfficial": f(z["cl_official"][i], 4),
                          "clModel": f(z["cl_model"][i], 4),
                          "clModelFixed": f(z["cl_model_u62"][i], 4)}
                         for i, n in enumerate(z["names"])]
res["anchor"] = anchor

res["llmTools"] = {
    "designVars": [
        {"key": "L", "min": 0.004, "max": 3.993, "step": 0.001, "label": "弯度系数",
         "effect": "越大弯度越高，升力越高，低头力矩与阻力也跟着变大"},
        {"key": "P", "min": 3.005, "max": 7.969, "step": 0.001, "label": "弯度位置",
         "effect": "乘零点零五得到最大弯度所在的弦向位置，越大越靠后"},
        {"key": "Q", "min": 0, "max": 1, "step": 1, "label": "弯型",
         "effect": "零是普通弯度，一是反弯，反弯能减小低头力矩"},
        {"key": "T", "min": 5.008, "max": 19.989, "step": 0.001, "label": "厚度系数",
         "effect": "除以一百得到厚弦比，越厚结构越好但阻力越大"},
        {"key": "aoa", "min": -4.94, "max": 14.93, "step": 0.1, "label": "攻角",
         "effect": "来流方向与弦线的夹角，单位是度"},
        {"key": "uInf", "min": 31.4, "max": 93.6, "step": 0.1, "label": "来流速度",
         "effect": "单位是米每秒，与弦长一起决定雷诺数"},
    ],
    "intents": [
        {"id": "max_cl_band", "label": "八度附近升力尽量高，十度也别掉",
         "keywords": ["升力", "高", "大", "8", "八", "10", "十", "不掉", "别掉", "保持"],
         "goal": {"maximize": "cl8", "constrain": ["bandLoss"], "bandAoas": [8, 10]}},
        {"id": "low_delta", "label": "更看重带内损失小",
         "keywords": ["损失", "稳", "平滑", "平稳", "带内"],
         "goal": {"maximize": "cl8", "constrain": ["bandLoss"], "weight": "delta"}},
        {"id": "thin_fast", "label": "薄一点快一点",
         "keywords": ["薄", "快", "高速", "阻力", "速度"],
         "goal": {"maximize": "cl8", "prefer": {"T": "low"}}},
        {"id": "reflex", "label": "用反弯减小低头力矩",
         "keywords": ["反弯", "力矩", "低头", "配平"],
         "goal": {"maximize": "cl8", "forceQ": 1}},
        {"id": "default", "label": "按默认目标给一版设计", "keywords": [],
         "goal": {"maximize": "cl8", "constrain": ["bandLoss"]}},
    ],
    "presets": [
        {"id": "seed_s7_best", "label": "三种子里八度升力最高",
         "L": 3.673469640579036, "P": 7.969, "Q": 0, "T": 15.08206399216206},
        {"id": "seed_s42_best", "label": "升力与带内损失的折中",
         "L": 3.9183696086964033, "P": 7.969, "Q": 0, "T": 15.542719989753579},
        {"id": "db_bandbest", "label": "数据库带内最优", "L": 3.311, "P": 7.957, "Q": 1, "T": 7.685},
        {"id": "interior", "label": "设计空间内部的稳妥解",
         "L": 2.3214083105048897, "P": 7.386408088512039, "Q": 0, "T": 11.557420377236344},
    ],
    "toolSchema": [
        {"name": "search_design", "desc": "在预计算的设计前沿里按目标排序取候选"},
        {"name": "evaluate_design", "desc": "用压力曲线模型算给定参数在三个攻角下的升力与约束"},
        {"name": "invert_pressure", "desc": "按目标压力曲线反演一版形状参数"},
        {"name": "explain_case", "desc": "给出该形状的几何与流场说明"},
    ],
    "note": "没有填接口信息时用关键词加预计算的方式出结果，这条路径不联网也能完整演示。",
}
res["caveats"] = [
    "绝对升力值没有经过风洞或官方力系数验证，只用相对排序与相对误差。",
    "压力曲线拟合得好不等于形状还原唯一，反演有明显多解。",
    "分离代理与局部马赫代理都是替代指标，不是直接求解出来的物理量。",
    "流场模型与官方图网络的评估网格不是同一个，并列时只能比同口径的排序。",
    "示例数据与真实数据必须在界面上区分显示。",
]
res["meta"] = {"generatedBy": "04_results/demo/tools/make_web_data.py",
               "note": "全部数值来自 04_results 下已验收的产物，只做格式转换，没有重算物理量。"}
jdump("results.json", res)

# ================= 6 图片 =================
if not args.no_figs:
    print("\n== 图片")
    from PIL import Image
    md = os.path.join(PAPER, "paper_zh.md")
    pairs = []
    if os.path.isfile(md):
        lines = open(md, encoding="utf-8").read().splitlines()
        for i, ln in enumerate(lines):
            m = re.match(r"!\[[^\]]*\]\(figs/([^)]+)\)", ln.strip())
            if not m:
                continue
            caption = ""
            for jj in range(i + 1, min(i + 5, len(lines))):
                if lines[jj].strip():
                    caption = lines[jj].strip()
                    break
            pairs.append((m.group(1), caption))
    figdir = os.path.join(OUT, "figs")
    os.makedirs(figdir, exist_ok=True)
    index = []
    for k, (fname, caption) in enumerate(pairs, 1):
        src = os.path.join(PAPER, "figs", fname)
        if not os.path.isfile(src):
            report["skipped"].append("figs/" + fname)
            continue
        im = Image.open(src).convert("RGB")
        if im.width > 1400:
            im = im.resize((1400, max(1, round(im.height * 1400 / im.width))), Image.LANCZOS)
        dst = "fig%02d.webp" % k
        im.save(os.path.join(figdir, dst), "WEBP", quality=82, method=5)
        clean = caption.replace("**", "")
        title = re.sub(r"^图\s*\d+[　\s]*", "", clean).split("。")[0].strip()
        index.append({"file": dst, "title": title, "caption": clean,
                      "source": "论文图 %d" % k, "original": fname})
    json.dump({"items": index, "note": "图片由论文插图转成 WebP，宽度上限 1400 像素"},
              open(os.path.join(figdir, "index.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    tot = sum(os.path.getsize(os.path.join(figdir, x)) for x in os.listdir(figdir))
    print("  图片 %d 张，合计 %.2f MB" % (len(index), tot / 1048576))
    report["written"].append({"file": "figs/", "count": len(index), "bytes": int(tot)})

# ================= 7 清单 =================
print("\n== manifest.json")
entries = []
for fname, kind, req in (
        ("b_cp.onnx", "model", True), ("b_field.onnx", "model", True),
        ("cp_basis.json", "basis", True), ("field_basis.bin", "basis", True),
        ("field_basis.json", "basis", True), ("grid.json", "grid", True),
        ("samples.json", "samples", True), ("samples_geom.bin", "samples", True),
        ("samples_cp.bin", "samples", True), ("samples_alpha.bin", "samples", True),
        ("cases_field.bin", "cases", True), ("cases_field.json", "cases", True),
        ("results.json", "results", True), ("geom_fixture.json", "basis", False)):
    p = os.path.join(OUT, fname)
    if os.path.isfile(p):
        entries.append({"name": fname, "kind": kind,
                        "sizeMb": round(os.path.getsize(p) / 1048576, 3),
                        "required": req, "bytes": int(os.path.getsize(p)),
                        "source": "04_results/demo/tools/"})
    else:
        report["skipped"].append(fname)
figdir = os.path.join(OUT, "figs")
if os.path.isdir(figdir):
    n_fig = len([x for x in os.listdir(figdir) if x.endswith(".webp")])
    fig_bytes = sum(os.path.getsize(os.path.join(figdir, x)) for x in os.listdir(figdir))
    entries.append({"name": "figs/", "kind": "image", "sizeMb": round(fig_bytes / 1048576, 3),
                    "required": True, "bytes": int(fig_bytes), "count": n_fig,
                    "source": "04_results/paper/figs"})
total = sum(e["bytes"] for e in entries)
jdump("manifest.json", {
    "generatedAt": "2026-10-06", "version": 1, "entries": entries,
    "totalMb": round(total / 1048576, 2),
    "note": "资产清单，供前端核对与容量检查",
}, compact=False)

top = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT)
          if os.path.isfile(os.path.join(OUT, x)))
print("\n== 汇总")
print("  public/data 顶层文件合计 %.2f MB" % (top / 1048576))
if report["skipped"]:
    print("  跳过 %d 项：%s" % (len(report["skipped"]), report["skipped"][:8]))
report["totalBytes"] = int(top)
os.makedirs(REMOTE, exist_ok=True)
json.dump(report, open(os.path.join(REMOTE, "make_web_data_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("MAKE_WEB_DATA_DONE")

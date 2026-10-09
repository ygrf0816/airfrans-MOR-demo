# -*- coding: utf-8 -*-
"""把演示资产的核对一次跑齐，并写出一份可复核的结论文件。

跑法（项目根目录下）：
  python 04_results/demo/tools/verify_demo_assets.py

做五件事：
  1. 重新读一遍 public/data 下的资产，核对文件齐不齐、字节数对不对、数组形状对不对；
  2. 跑前端的几何口径自检（node scripts/check-geom.mjs）；
  3. 跑前端的代理量与升力权重自检（node scripts/check-proxies.mjs）；
  4. 用本机 onnxruntime 独立打开两个模型，核对输入输出名称与形状；
  5. 核对外部导出的核对结果（export_check.json 与 trim_check.json）是否通过。

结论写到 04_results/demo/data_remote/verify_demo_assets.json。
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "04_results", "demo", "app", "public", "data")
APP = os.path.join(ROOT, "04_results", "demo", "app")
REMOTE = os.path.join(ROOT, "04_results", "demo", "data_remote")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
out = {"checks": [], "passed": True}


def add(name, ok, detail):
    out["checks"].append({"name": name, "passed": bool(ok), "detail": detail})
    if not ok:
        out["passed"] = False
    print("  [%s] %-34s %s" % ("通过" if ok else "不通过", name, detail))


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


print("=" * 78)
print("一、资产完整性与形状")
print("=" * 78)
meta_path = os.path.join(DATA, "manifest.json")
man = load_json(meta_path)
required = [e for e in man["entries"] if e.get("required") and e["name"] != "figs/"]
missing = [e["name"] for e in required if not os.path.isfile(os.path.join(DATA, e["name"]))]
add("必需资产齐全", not missing, "缺 %s" % missing if missing else "共 %d 项" % len(required))

wrong = []
for e in required:
    p = os.path.join(DATA, e["name"])
    if os.path.isfile(p) and os.path.getsize(p) != e["bytes"]:
        wrong.append((e["name"], e["bytes"], os.path.getsize(p)))
add("清单字节数与实际一致", not wrong, "不符 %s" % wrong if wrong else "全部一致")

cp_basis = load_json(os.path.join(DATA, "cp_basis.json"))
phi = np.asarray(cp_basis["phi"], float)
mu = np.asarray(cp_basis["mu"], float)
add("压力曲线基形状", phi.shape == (256, int(cp_basis["podR"])) and mu.size == 256,
    "phi %s mu %d podR %s" % (phi.shape, mu.size, cp_basis["podR"]))
gram = phi.T @ phi
# cp_basis.json 里的基保留到小数点后八位，所以正交性偏差在 1e-8 量级属正常，
# 不是基本身的问题（全精度元信息文件里量到的偏差是 1e-15）。
gram_dev = float(np.abs(gram - np.eye(phi.shape[1])).max())
add("压力曲线基列正交", gram_dev < 1e-6, "偏差 %.2e（八位小数取整所致）" % gram_dev)

fmeta = load_json(os.path.join(DATA, "field_basis.json"))
fbin = os.path.join(DATA, "field_basis.bin")
sz = os.path.getsize(fbin)
add("场基字节数", sz == fmeta["totalFloats"] * 4,
    "%d 字节，声明 %d" % (sz, fmeta["totalFloats"] * 4))
blob = np.fromfile(fbin, dtype=np.float32)
ok_off = True
cur = 0
for k in range(4):
    if fmeta["phiOffset"][k] != cur:
        ok_off = False
    cur += fmeta["nGrid"] * fmeta["podR"]
for k in range(4):
    if fmeta["muOffset"][k] != cur:
        ok_off = False
    cur += fmeta["nGrid"]
add("场基偏移量自洽", ok_off and cur == blob.size, "累加到 %d，总长 %d" % (cur, blob.size))

grid = load_json(os.path.join(DATA, "grid.json"))
flat = np.concatenate([np.full(n, bool(v)) for v, n in grid["maskRle"]])
add("网格掩膜游程", flat.size == grid["nx"] * grid["ny"] and int(flat.sum()) == grid["nValid"],
    "展开 %d，true %d，声明有效格 %d" % (flat.size, int(flat.sum()), grid["nValid"]))
add("网格有效格与场基一致", int(flat.sum()) == fmeta["nGrid"],
    "%d 对 %d" % (int(flat.sum()), fmeta["nGrid"]))

samples = load_json(os.path.join(DATA, "samples.json"))
n = samples["count"]
geom = np.fromfile(os.path.join(DATA, "samples_geom.bin"), dtype=np.float32)
cp_bin = np.fromfile(os.path.join(DATA, "samples_cp.bin"), dtype=np.float32)
alpha = np.fromfile(os.path.join(DATA, "samples_alpha.bin"), dtype=np.float32)
add("样本几何长度", geom.size == n * 128, "%d，期望 %d" % (geom.size, n * 128))
add("样本压力曲线长度", cp_bin.size == n * 256, "%d，期望 %d" % (cp_bin.size, n * 256))
add("样本模态坐标长度", alpha.size == n * 136, "%d，期望 %d" % (alpha.size, n * 136))

# 端到端：用前 8 列模态坐标还原压力曲线，与真值比
A = alpha.reshape(n, 136)
CP = cp_bin.reshape(n, 256)
rec = A[:, :8].astype(np.float64) @ phi.T + mu
rel = np.linalg.norm(rec - CP, axis=1) / np.maximum(np.linalg.norm(CP, axis=1), 1e-12)
add("压力曲线端到端还原", float(np.median(rel)) < 0.08,
    "逐条相对 L2 中位 %.4f，最大 %.4f" % (float(np.median(rel)), float(rel.max())))

cf = load_json(os.path.join(DATA, "cases_field.json"))
cbin = os.path.getsize(os.path.join(DATA, "cases_field.bin"))
add("案例流场长度", cbin == cf["totalFloats"] * 4,
    "%d 字节，声明 %d 个数，%d 个块" % (cbin, cf["totalFloats"], len(cf["blocks"])))

res = load_json(os.path.join(DATA, "results.json"))
need = ["fronts", "grid", "history", "cases", "inversions", "anchor", "dataset",
        "modelMetrics", "objective", "llmTools", "caveats"]
lack = [k for k in need if k not in res]
add("results.json 字段齐全", not lack, "缺 %s" % lack if lack else "共 %d 个顶层键" % len(res))
add("前沿成员数", len(res["fronts"]) == 240, "%d 个" % len(res["fronts"]))
add("反演目标数", len(res["inversions"]) == 11, "%d 个" % len(res["inversions"]))
add("案例数", len(res["cases"]) == 8, "%d 个" % len(res["cases"]))
figdir = os.path.join(DATA, "figs")
figs = load_json(os.path.join(figdir, "index.json"))
have = [x for x in os.listdir(figdir) if x.endswith(".webp")]
add("插图齐全", len(have) >= len(figs["items"]),
    "索引 %d 张，文件 %d 张" % (len(figs["items"]), len(have)))

print()
print("=" * 78)
print("二、前端口径自检")
print("=" * 78)
for script in ("check-geom.mjs", "check-proxies.mjs"):
    p = os.path.join(APP, "scripts", script)
    r = subprocess.run(["node", p], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=APP)
    text = (r.stdout or "") + (r.stderr or "")
    # 两个脚本成功时都会打印一个标记，标记不一定在最后一行，所以整段找
    ok = r.returncode == 0 and ("_OK" in text or "PASS" in text.upper())
    marker = [ln for ln in text.splitlines() if "_OK" in ln or "PASS" in ln.upper()]
    detail = (marker[-1] if marker else (text.strip().splitlines() or [""])[-1])[:150]
    add(script, ok, detail)

print()
print("=" * 78)
print("三、本机独立核对模型")
print("=" * 78)
try:
    import onnxruntime as ort
    for name, dim in (("b_cp.onnx", 8), ("b_field.onnx", 128)):
        p = os.path.join(DATA, name)
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        o = s.get_outputs()[0]
        ok = (i.name == "x" and list(i.shape) == [1, 133]
              and o.name == "alpha" and list(o.shape) == [1, dim])
        add("本机打开 " + name, ok,
            "输入 %s%s 输出 %s%s" % (i.name, list(i.shape), o.name, list(o.shape)))
except Exception as exc:
    add("本机 onnxruntime 核对", False, "异常 %s" % exc)

print()
print("=" * 78)
print("四、外部核对结论")
print("=" * 78)
for name, key in (("export_check.json", "allPassed"), ("trim_check.json", None)):
    p = os.path.join(REMOTE, name)
    if not os.path.isfile(p):
        add(name, False, "文件不在")
        continue
    d = load_json(p)
    if key:
        add(name, bool(d.get(key)), "allPassed=%s，判据：%s"
            % (d.get(key), d.get("criterion", "")))
    else:
        cons = d.get("consistency", d)
        ok = bool(cons.get("allPassed"))
        uc = cons.get("unitConversion") or {}
        add(name, ok, "对照通过=%s，物理换算结论：%s" % (ok, uc.get("conclusion", "")))
        fl = cons.get("truncationFloor")
        if fl:
            add(" 场截断地板", True, "  ".join("%s %.4f" % (k, v["relL2Mean"])
                                              for k, v in fl.items()))

print()
print("=" * 78)
print("结论：%s，共 %d 项，其中不通过 %d 项"
      % ("全部通过" if out["passed"] else "有不通过项",
         len(out["checks"]), sum(1 for c in out["checks"] if not c["passed"])))
print("=" * 78)
os.makedirs(REMOTE, exist_ok=True)
json.dump(out, open(os.path.join(REMOTE, "verify_demo_assets.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("-> %s" % os.path.join(REMOTE, "verify_demo_assets.json"))
sys.exit(0 if out["passed"] else 1)

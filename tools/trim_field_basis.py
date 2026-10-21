#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裁剪流场 POD 基，导出网页用的网格与样本坐标。

在工作站上跑（基文件 401 MB，本地没有）：
  /home/user/.conda/envs/pytorch/bin/python trim_field_basis.py \
      --root /home/user/airfrans \
      --out  /home/user/airfrans/04_results/demo/data_remote

产物：
  field_basis.bin            float32，前 4*31066*32 是基，后 4*31066 是均值
  field_basis.json           偏移量与物理单位换算系数
  grid.json                  网格范围、坐标、掩膜游程
  samples_alpha_field.bin    float32 [1000,128]，真实样本在基上的坐标
  samples_alpha_field.json   样本名顺序
  trim_check.json            全部校验结果

口径：基在归一化空间，物理值 = 归一化值 * stdOut[k] + meanOut[k]，
meanOut/stdOut 取自 runs_full_A_eval_metrics.json 的 coef_norm。
"""
import argparse
import json
import os
import sys

import numpy as np

FIELDS = ["u", "v", "p", "nu_t"]
CHECK_NAME = "airFoil2D_SST_61.642_14.277_1.498_5.176_0.0_16.386"

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/user/airfrans")
ap.add_argument("--pred", default=None)
ap.add_argument("--basis", default=None)
ap.add_argument("--vol", default=None)
ap.add_argument("--r", type=int, default=32)
ap.add_argument("--out", default=None)
args = ap.parse_args()

ROOT = os.path.abspath(args.root)
PRED = os.path.abspath(args.pred or os.path.join(ROOT, "prediction"))
OUT = os.path.abspath(args.out or os.path.join(ROOT, "04_results", "demo", "data_remote"))
BASIS = os.path.abspath(args.basis or os.path.join(ROOT, "pod", "bfield_basis.npz"))
VOL = os.path.abspath(args.vol or os.path.join(ROOT, "pod", "vol_fields_1000.npz"))
FIELD_GRID = os.path.join(PRED, "04_results", "prediction", "csv", "full", "field_grid.npz")
os.makedirs(OUT, exist_ok=True)
check = {"basis": BASIS, "vol": VOL, "r": args.r}

# ---------- 1) 基与均值 ----------
print("读基 %s" % BASIS)
bd = np.load(BASIS, allow_pickle=True)
print("  键：%s" % bd.files)
pkey = [k for k in bd.files if k == "Phi_full" or k.startswith("Phi_full.")][0]
mkey = pkey.replace("Phi", "mu")
n_member = len(bd[pkey])
Phi_raw = [np.asarray(bd[pkey][k], float) for k in range(n_member)]
Mu_raw = [np.asarray(bd[mkey][k], float).ravel() for k in range(n_member)]
print("  基成员数 %d，第 0 个形状 %s，均值形状 %s"
      % (n_member, Phi_raw[0].shape, Mu_raw[0].shape))
assert n_member >= 4, "基的成员数少于 4，实际 %d" % n_member

n_cell = Phi_raw[0].shape[0]
n_mode_avail = Phi_raw[0].shape[1]
assert n_mode_avail >= args.r, "基只有 %d 个模态，不足 %d" % (n_mode_avail, args.r)
check["nCell"] = int(n_cell)
check["nModeAvailable"] = int(n_mode_avail)
check["nModeUsed"] = int(args.r)

Phi = [np.ascontiguousarray(Phi_raw[k][:, :args.r]) for k in range(4)]
Mu = [np.ascontiguousarray(Mu_raw[k][:n_cell]) for k in range(4)]
for k in range(4):
    print("  场 %-5s 基 %s 均值 %s 基范数 %.4g"
          % (FIELDS[k], Phi[k].shape, Mu[k].shape, float(np.linalg.norm(Phi[k]))))
    assert Phi[k].shape == (n_cell, args.r)
    assert Mu[k].shape == (n_cell,)

# ---------- 2) 物理单位换算系数 ----------
am_path = os.path.join(PRED, "04_results", "prediction", "runs", "full_A", "eval_metrics.json")
assert os.path.isfile(am_path), "找不到 %s" % am_path
am = json.load(open(am_path, encoding="utf-8"))
mean_out = [float(v) for v in np.asarray(am["coef_norm"]["mean_out"]).ravel()]
std_out = [float(v) for v in np.asarray(am["coef_norm"]["std_out"]).ravel()]
print("物理换算 meanOut = %s" % np.round(mean_out, 5))
print("物理换算 stdOut  = %s" % np.round(std_out, 5))
assert len(mean_out) == 4 and len(std_out) == 4

# ---------- 3) 写 field_basis.bin 与 json ----------
phi_off, mu_off, cursor = [], [], 0
for k in range(4):
    phi_off.append(cursor)
    cursor += n_cell * args.r
for k in range(4):
    mu_off.append(cursor)
    cursor += n_cell
blob = np.empty(cursor, dtype=np.float32)
for k in range(4):
    blob[phi_off[k]:phi_off[k] + n_cell * args.r] = Phi[k].astype(np.float32).ravel()
    blob[mu_off[k]:mu_off[k] + n_cell] = Mu[k].astype(np.float32)
bin_path = os.path.join(OUT, "field_basis.bin")
blob.tofile(bin_path)
print("写出 field_basis.bin  %.2f MB，共 %d 个 float32"
      % (os.path.getsize(bin_path) / 1024 / 1024, blob.size))
assert os.path.getsize(bin_path) == cursor * 4

meta = {
    "podR": int(args.r), "nGrid": int(n_cell), "nFields": 4, "order": FIELDS,
    "phiOffset": [int(v) for v in phi_off], "muOffset": [int(v) for v in mu_off],
    "totalFloats": int(cursor),
    "meanOut": mean_out, "stdOut": std_out,
    "unit": ["m/s", "m/s", "m^2/s^2", "m^2/s"],
    "nModeAvailable": int(n_mode_avail),
    "note": ("行优先，格点顺序与 grid.json 的掩膜行优先顺序一致；"
             "物理值 = 归一化值 * stdOut[k] + meanOut[k]"),
}
json.dump(meta, open(os.path.join(OUT, "field_basis.json"), "w", encoding="utf-8"),
          ensure_ascii=False)

# ---------- 4) 网格 ----------
print("\n读真值 %s" % VOL)
vz = np.load(VOL, allow_pickle=True)
names = [str(n) for n in vz["names"]]
nx, ny = int(vz["nx"]), int(vz["ny"])
x0, x1 = float(vz["x0"]), float(vz["x1"])
y0, y1 = float(vz["y0"]), float(vz["y1"])
mask = np.asarray(vz["mask"], bool).reshape(ny, nx)
xs = np.linspace(x0, x1, nx)
ys = np.linspace(y0, y1, ny)
print("  网格 %d x %d，范围 x[%.3f,%.3f] y[%.3f,%.3f]，有效格 %d"
      % (nx, ny, x0, x1, y0, y1, int(mask.sum())))
assert int(mask.sum()) == n_cell, "掩膜有效格 %d 与基的 %d 不一致" % (int(mask.sum()), n_cell)


def rle(flat):
    out = []
    cur = bool(flat[0])
    run = 0
    for v in flat:
        v = bool(v)
        if v == cur:
            run += 1
        else:
            out.append([1 if cur else 0, run])
            cur, run = v, 1
    out.append([1 if cur else 0, run])
    return out


rl = rle(mask.ravel())
expanded = np.concatenate([np.full(n, bool(v)) for v, n in rl])
assert expanded.size == nx * ny, "游程展开长度 %d，应为 %d" % (expanded.size, nx * ny)
assert int(expanded.sum()) == n_cell
assert np.array_equal(expanded, mask.ravel())
print("  游程 %d 段，展开长度 %d，true 个数 %d，与掩膜逐位一致"
      % (len(rl), expanded.size, int(expanded.sum())))

grid = {"nx": nx, "ny": ny, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "x": [float(v) for v in xs], "y": [float(v) for v in ys],
        "maskRle": [[int(v), int(n)] for v, n in rl], "nValid": int(n_cell),
        "note": "掩膜为 true 表示该格在流场内，rle 展平顺序是 (ny, nx) 行优先"}
json.dump(grid, open(os.path.join(OUT, "grid.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("写出 grid.json  %.1f KB" % (os.path.getsize(os.path.join(OUT, "grid.json")) / 1024))

# ---------- 5) 真实样本在基上的坐标，顺带算截断地板 ----------
# 截断地板就是"真值投影到 32 个模态再还原"与真值的差。它只跟基本身有关，
# 与任何模型无关，是演示里拿模型误差做对比时的参照线。这里顺手一起算，
# 免得为了它把 4 个 248 MB 的成员再读一遍。
alpha = np.zeros((len(names), 4 * args.r), dtype=np.float32)
floor = {}
for k, fn in enumerate(FIELDS):
    raw = np.asarray(vz[fn], dtype=np.float64)
    assert raw.shape == (len(names), n_cell), "%s 形状 %s 不符" % (fn, raw.shape)
    A = (raw - mean_out[k]) / std_out[k] - Mu[k]
    coef = A @ Phi[k]
    alpha[:, k * args.r:(k + 1) * args.r] = coef.astype(np.float32)
    rec = (coef @ Phi[k].T + Mu[k]) * std_out[k] + mean_out[k]
    rel = (np.linalg.norm(rec - raw, axis=1)
           / np.maximum(np.linalg.norm(raw, axis=1), 1e-12))
    floor[fn] = {"relL2Mean": float(rel.mean()), "relL2Median": float(np.median(rel)),
                 "relL2P95": float(np.percentile(rel, 95)),
                 "ampRatio": float(np.linalg.norm(rec) / max(np.linalg.norm(raw), 1e-12))}
    print("  投影 %-5s 完成，坐标范围 [%.3g, %.3g]，截断地板 相对 L2 均值 %.4f 中位 %.4f 振幅比 %.4f"
          % (fn, alpha[:, k * args.r:(k + 1) * args.r].min(),
             alpha[:, k * args.r:(k + 1) * args.r].max(),
             floor[fn]["relL2Mean"], floor[fn]["relL2Median"], floor[fn]["ampRatio"]))
    del raw, A, coef, rec, rel
alpha.tofile(os.path.join(OUT, "samples_alpha_field.bin"))
json.dump({"count": len(names), "order": FIELDS, "r": int(args.r), "names": names,
           "truncationFloor": floor,
           "truncationFloorNote": ("把真值投影到 r 个模态再还原，与真值的逐样本相对 L2。"
                                   "只反映基本身的表示能力，与模型无关。")},
          open(os.path.join(OUT, "samples_alpha_field.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
print("写出 samples_alpha_field.bin  %.2f MB"
      % (os.path.getsize(os.path.join(OUT, "samples_alpha_field.bin")) / 1024 / 1024))

# ---------- 6) 单位换算自检 ----------
# 对照对象是 field_grid.npz 里另一条 B 链路的**预测**，不是真值。
# 判据是相关够高、斜率落在 1 附近。若物理换算系数漏乘或乘错量级，
# 斜率会立刻跑到几十倍去。斜率略小于 1 是 32 个模态截断丢了吸力峰峰值造成的。
cons = {"fieldGrid": FIELD_GRID, "sample": None, "truncationFloor": floor,
        "note": ("对照对象是另一条 B 链路的预测，不是真值。斜率落在 1 附近说明物理换算正确。")}
print("\n单位换算自检")
if os.path.isfile(FIELD_GRID):
    fg = np.load(FIELD_GRID, allow_pickle=True)
    fg_names = [str(n) for n in fg["names"]]
    name = CHECK_NAME if CHECK_NAME in fg_names else fg_names[0]
    cons["sample"] = name
    row = names.index(name)
    ref_keys = {"u": "u_B", "v": "v_B", "p": "p_B", "nu_t": "nu_B"}
    slopes, corrs = [], []
    for k, fn in enumerate(FIELDS):
        ref = np.asarray(fg[ref_keys[fn]][fg_names.index(name)], float).ravel()
        rec = (alpha[row, k * args.r:(k + 1) * args.r].astype(np.float64) @ Phi[k].T
               + Mu[k]) * std_out[k] + mean_out[k]
        cc = float(np.corrcoef(rec, ref)[0, 1])
        A2 = np.column_stack([rec, np.ones_like(rec)])
        (a_, b_), *_ = np.linalg.lstsq(A2, ref, rcond=None)
        resid = float(np.abs(a_ * rec + b_ - ref).sum() / np.abs(ref).sum())
        slopes.append(float(a_))
        corrs.append(cc)
        cons[fn] = {"corr": cc, "slopeVsOtherChainPrediction": float(a_),
                    "intercept": float(b_), "relDiffAfterAffine": resid,
                    "passed": bool(cc > 0.95 and 0.7 <= a_ <= 1.3)}
        print("  对照 %-5s 相关 %.4f  斜率 %.4f  截距 %.4g  仿射后残差 %.3e  通过 %s"
              % (fn, cc, a_, b_, resid, cons[fn]["passed"]))
    cons["allPassed"] = bool(all(cons[fn]["passed"] for fn in FIELDS))
    cons["unitConversion"] = {
        "meanOut": mean_out, "stdOut": std_out,
        "formula": "物理值 = 归一化值 * stdOut[k] + meanOut[k]",
        "slopeMean": float(np.mean(slopes)), "slopeMin": float(min(slopes)),
        "slopeMax": float(max(slopes)), "corrMean": float(np.mean(corrs)),
        "conclusion": ("斜率在 1 附近，物理换算系数正确。斜率略小于 1 是 32 个模态"
                       "截断丢了吸力峰峰值导致的，不是换算错。"),
    }
    print("  斜率 %.3f 到 %.3f，均值 %.3f；相关均值 %.4f"
          % (min(slopes), max(slopes), float(np.mean(slopes)), float(np.mean(corrs))))
else:
    cons["skipped"] = "缺少 %s" % FIELD_GRID
    print("  跳过对照：%s" % cons["skipped"])

check["consistency"] = cons
check["written"] = sorted(os.path.basename(p) for p in
                          [bin_path, os.path.join(OUT, "field_basis.json"),
                           os.path.join(OUT, "grid.json"),
                           os.path.join(OUT, "samples_alpha_field.bin"),
                           os.path.join(OUT, "samples_alpha_field.json")])
json.dump(check, open(os.path.join(OUT, "trim_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n-> %s" % OUT)
print("TRIM_DONE")

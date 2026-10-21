# -*- coding: utf-8 -*-
"""给**候选翼型**（不在数据集里的新形状）预测流场：用已有的 B 全场模型。

口径全部照抄已验证过的两条链路，不新造：
  * 输入：5 标量 `[U∞, 攻角, 0.15L, T/100, round(Q)]`（`input_side._camber_t` 口径）
          + 128 维几何（余弦站位、TE→LE，与 `geometry_input.geometric_vector` 同口径，
          本脚本用官方 `naca_generator` 现生成）；两者分别用训练集的 mean/std 做 z-score。
  * 输出：4 场 × 32 个 POD 坐标 → `Phi_full`/`mu_full` 还原到 256×128 网格（31066 个有效格）。
  * 归一化统计量直接取自 checkpoint 的 `norm_stats`（`patch_B_field_stats.py` 回填的训练统计量）。

产物（写到 04_results/inversion/results/）：
  cases_field.npz          x/y/mask + 每个「代号×攻角」的 p/u/v/nu（ny×nx，无效格 NaN）+ meta
  cases_field_summary.csv  case,aoa,p_min,p_max,u_max,nu_max
  cases_field_check.json   与已落盘 field_grid.npz 的自检相对差

用法（工作站上）：/home/zch/.conda/envs/pytorch/bin/python field_cases.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/zch/airfrans"
PRED = os.path.join(ROOT, "prediction")
RUN = os.path.join(PRED, "04_results", "prediction", "runs", "full_B_field")
OUT = os.path.join(ROOT, "04_results", "inversion", "results")
BASIS = os.path.join(ROOT, "pod", "bfield_basis.npz")
VOL = os.path.join(ROOT, "pod", "vol_fields_1000.npz")
FIELD_GRID = os.path.join(PRED, "04_results", "prediction", "csv", "full", "field_grid.npz")
TASK_ROOT = os.path.join(ROOT, "data", "Dataset")
sys.path.insert(0, PRED)
sys.path.insert(0, os.path.join(ROOT, "01_initial", "AirfRANS-main"))

import naca_generator as NG                      # noqa: E402
from train_B_field import FieldMLP               # noqa: E402
from input_side import input_vector              # noqa: E402

POD_R = 32
N_ST = 64
CASES = [
    ("case1_success", (3.673469640579036, 7.969, 0, 15.08206399216206), (4, 8, 10)),
    ("case2_boundary", (3.9183696086964033, 7.969, 0, 15.542719989753579), (8,)),
    ("case3_fail_mach", (2.905, 7.969, 0, 8.337), (8, 10)),
    ("case4_fail_robust", (3.268, 7.969, 0, 13.331), (8, 10)),
    ("case5_interior", (2.3214083105048897, 7.386408088512039, 0, 11.557420377236344), (8,)),
    ("db_designbest", (2.248, 5.618, 1, 8.267), (8,)),
    ("db_bandbest", (3.311, 7.957, 1, 7.685), (8, 10)),
]


def geom128(params, nb=400):
    """(L,P,Q,T) -> 128 维几何（与数据集 geometry_input 同口径）。"""
    L, P, Q, T = params
    pos = NG.naca_generator((L, P, int(round(Q)), T), nb_samples=nb,
                            cosine_spacing=True, verbose=False)
    X, Y = pos[:, 0], pos[:, 1]
    x0, x1 = X.min(), X.max()
    c = x1 - x0
    beta = np.pi * np.linspace(1, 0, N_ST, endpoint=False)
    xs = (1 - np.cos(beta)) / 2.0 * c + x0
    vec = np.zeros(2 * N_ST)
    for i, xx in enumerate(xs):
        band = np.abs(X - xx) < (c / N_ST) * 0.5
        ys = Y[band]
        if ys.size >= 2:
            vec[i], vec[N_ST + i] = ys.max() / c, ys.min() / c
        elif ys.size == 1:
            vec[i] = vec[N_ST + i] = ys[0] / c
    return vec


def main():
    st = torch.load(os.path.join(RUN, "best_model.pt"), map_location="cpu")
    ns = st["norm_stats"]
    smean, sstd = np.array(ns["smean"]).ravel(), np.array(ns["sstd"]).ravel()
    gmean, gstd = np.array(ns["gmean"]).ravel(), np.array(ns["gstd"]).ravel()
    model = FieldMLP(133, 4 * POD_R)
    model.load_state_dict(st["model_state"])
    model.eval()

    bd = np.load(BASIS, allow_pickle=True)
    pkey = [k for k in bd.files if k == "Phi_full" or k.startswith("Phi_full.")][0]
    Phi = [np.asarray(bd[pkey][k], float) for k in range(len(bd[pkey]))]
    Mu = [np.asarray(bd[pkey.replace("Phi", "mu")][k], float).ravel()
          for k in range(len(bd[pkey.replace("Phi", "mu")]))]
    print("POD 基：%d 场，每场 %d 个模态，%d 个网格点" % (len(Phi), Phi[0].shape[1],
                                                        Phi[0].shape[0]))

    vz = np.load(VOL, allow_pickle=True)
    nx, ny = int(vz["nx"]), int(vz["ny"])
    xs = np.linspace(float(vz["x0"]), float(vz["x1"]), nx)
    ys = np.linspace(float(vz["y0"]), float(vz["y1"]), ny)
    GX, GY = np.meshgrid(xs, ys, indexing="xy")
    mask = np.asarray(vz["mask"], bool).reshape(ny, nx)
    print("网格 %d×%d，有效格 %d（与基的 %d 一致：%s）"
          % (nx, ny, mask.sum(), Phi[0].shape[0], mask.sum() == Phi[0].shape[0]))

    def predict(params, aoa, geom=None, u_inf=62.0):
        g = geom128(params) if geom is None else np.asarray(geom, float).ravel()
        scal = np.array([[u_inf, aoa, 0.15 * params[0], params[3] / 100.0,
                          float(round(params[2]))]])
        X = np.concatenate([(scal - smean) / sstd,
                            ((g - gmean) / gstd)[None, :]], axis=1)
        with torch.no_grad():
            pred = model(torch.tensor(X, dtype=torch.float32)).numpy()[0]
        out = np.empty((Phi[0].shape[0], 4))
        for k in range(4):
            out[:, k] = pred[k * POD_R:(k + 1) * POD_R] @ Phi[k].T + Mu[k]
        return out

    # ---------- 自检：真实样本走同一条还原路径，与已落盘 field_grid.npz 比 ----------
    CHECK = "airFoil2D_SST_61.642_14.277_1.498_5.176_0.0_16.386"
    par = [float(x) for x in CHECK.replace("airFoil2D_SST_", "").split("_")[2:]]
    aoa_c = float(CHECK.replace("airFoil2D_SST_", "").split("_")[1])
    u_c = float(CHECK.replace("airFoil2D_SST_", "").split("_")[0])
    # 数据集里这个真实样本的 128 维几何（按**名字**取，不用下标跨文件对齐）
    gv = np.load(os.path.join(ROOT, "pod", "geomvec_1000.npz"), allow_pickle=True)
    gidx = {str(n): i for i, n in enumerate(gv["names"])}
    g_dat = np.asarray(gv["G"][gidx[CHECK]], float).ravel()
    mine_data = predict(par, aoa_c, geom=g_dat, u_inf=u_c)
    mine_gen = predict(par, aoa_c, u_inf=u_c)
    fg = np.load(FIELD_GRID, allow_pickle=True)
    names = [str(n) for n in fg["names"]]
    key = {"u": "u_B", "v": "v_B", "p": "p_B", "nu_t": "nu_B"}
    check = {"name": CHECK, "coef_norm": {k: (np.asarray(v).ravel().tolist()
                                              if not isinstance(v, (str, int, float, dict))
                                              else v)
                                          for k, v in dict(ns.get("coef_norm", {})).items()},
             "note": "field_grid.npz 的 *_B 与基的口径可能差一个常数缩放；"
                     "这里同时报「相关系数」（看空间形态）与「最优缩放因子」。"}
    ok = True
    affine = {}
    for k, fn in enumerate(("u", "v", "p", "nu_t")):
        ref = np.asarray(fg[key[fn]][names.index(CHECK)], float)
        a_raw = mine_data[:, k]
        cc = float(np.corrcoef(a_raw, ref)[0, 1])
        # 官方把场做成仿射归一化（减均值除标准差），所以拟合 ref ≈ a*mine + b
        A = np.column_stack([a_raw, np.ones_like(a_raw)])
        (a_, b_), *_ = np.linalg.lstsq(A, ref, rcond=None)
        resid = float(np.abs(a_ * a_raw + b_ - ref).sum() / np.abs(ref).sum())
        # 用现生成几何的版本再比一次（几何生成带来的额外误差）
        A2 = np.column_stack([mine_gen[:, k], np.ones_like(a_raw)])
        (a2_, b2_), *_ = np.linalg.lstsq(A2, ref, rcond=None)
        resid2 = float(np.abs(a2_ * mine_gen[:, k] + b2_ - ref).sum() / np.abs(ref).sum())
        check[fn + "_corr"] = cc
        check[fn + "_affine_a"] = float(a_)
        check[fn + "_affine_b"] = float(b_)
        check[fn + "_rel_diff_after_affine"] = resid
        check[fn + "_rel_diff_after_affine_generated_geom"] = resid2
        affine[fn] = (float(a_), float(b_))
        print("  自检 %-5s 相关 %.4f  仿射(a=%.4g,b=%.4g) 后残差 %.2e（现生成几何 %.2e）"
              % (fn, cc, a_, b_, resid, resid2))
        ok = ok and cc > 0.99 and resid < 0.05
    check["passed"] = bool(ok)
    check["conclusion"] = ("空间形态与已落盘产物完全一致（相关≈1），差异只是官方仿射归一化"
                           "（减均值除标准差）⇒ 还原口径可用；预测场已换算到该单位"
                           if ok else "形态或残差不达标，口径不一致，停止")
    json.dump(check, open(os.path.join(OUT, "cases_field_check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if not ok:
        raise SystemExit("自检没过，停止（见 cases_field_check.json）")

    # ---------- 正式预测 ----------
    data, meta, rows = {}, {}, []
    for code, params, aoas in CASES:
        meta[code] = dict(L=params[0], P=params[1], Q=params[2], T=params[3],
                          u_inf=62.0, aoas=list(aoas))
        for a in aoas:
            f = predict(params, float(a))
            # 换算到与 field_grid.npz 同一单位（官方仿射归一化），便于与已有图对照
            f = np.column_stack([f[:, k] * affine[nm][0] + affine[nm][1]
                                 for k, nm in enumerate(("u", "v", "p", "nu_t"))])
            for k, fn in enumerate(("u", "v", "p", "nu_t")):
                img = np.full((ny, nx), np.nan, float)
                img[mask] = f[:, k]
                data["%s_%s_%d" % (fn, code, a)] = img.astype(np.float32)
            rows.append(dict(case=code, aoa=a,
                             p_min=float(np.nanmin(f[:, 2])), p_max=float(np.nanmax(f[:, 2])),
                             u_max=float(np.nanmax(np.hypot(f[:, 0], f[:, 1]))),
                             nu_max=float(np.nanmax(f[:, 3]))))
            print("  %-16s AoA=%2d  p∈[%.1f, %.1f]  |V|max=%.1f  nu_max=%.4f"
                  % (code, a, rows[-1]["p_min"], rows[-1]["p_max"],
                     rows[-1]["u_max"], rows[-1]["nu_max"]))
    data["x"] = xs
    data["y"] = ys
    data["mask"] = mask
    data["meta"] = json.dumps(dict(cases=meta, unit="与 prediction/…/field_grid.npz 同一口径",
                                   note="p/u/v/nu_t 为模型在 256×128 网格上的重构场"),
                              ensure_ascii=False)
    np.savez_compressed(os.path.join(OUT, "cases_field.npz"), **data)
    import csv
    with open(os.path.join(OUT, "cases_field_summary.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "aoa", "p_min", "p_max", "u_max", "nu_max"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("-> cases_field.npz / cases_field_summary.csv / cases_field_check.json")


if __name__ == "__main__":
    main()

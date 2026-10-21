"""判定实验（修复 4，先于任何 C_L/C_D 实现）：
  步骤 1：用官方 metrics.Compute_coefficients 口径在真值场上算 200 个 full_test
          样本的官方真值 C_L/C_D（pyvista 读 .vtu/.vtp，一次性）。
  步骤 2：用"压力项版" C_L（真实法向 + 真实弧长，缺摩擦项）在同样真值上算 C_L。
  步骤 3：比较两者逐样本相对差，取均值：
            相对差 < 5%  -> branch=a（升力由压力项主导，采用压力项版，命名
                             C_L_pressure_relerr，同时报与官方真值的校验残差；
                             C_D 不报）
            相对差 >= 5% -> branch=b（删除 C_L/C_D 字段，B 只报口径无关指标）
  输出 04_results/prediction/judge_lifting_full.json（train_B_transformer.py
          读这个文件决定最终报哪套力系数字）。

  前置：wallcp_1000.npz 必须已是前缘锚定版（修复 2），否则线积分里 Cp 位置错。
  数据：02_dataset/Dataset/（本机）或 工作站 /home/zch/airfrans/data/Dataset/。
"""
import os, sys, json, time
import numpy as np
import pyvista as pv

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from official_metrics import (official_force_split, official_force_coefficients,
                              pressure_term_lift_cp)  # noqa: E402,F401
from geometry_input import decode_sim  # noqa: E402


def log(m):
    print("[judge_lifting %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def _reorganize(in_pts, out_pts, qty):
    """官方 reorganize.py 等价（逐字实现）。"""
    idx = np.zeros(out_pts.shape[0])
    for i in range(out_pts.shape[0]):
        cond = (out_pts[i] == in_pts)
        cond = cond[:, 0] * cond[:, 1]
        idx[i] = np.argwhere(cond)[0][0]
    idx = idx.astype('int')
    assert (in_pts[idx] == out_pts).all()
    return qty[idx]


def pressure_term_lift(name, task_root, cp_curve):
    """压力项版 C_L（真实法向 + 真实弧长，缺摩擦项），用 994 点原始轮廓。

    ⚠ 关键：cp_curve 来自 workstation_pod.wall_cp 的**前缘锚定**口径（s=0 在前缘、
    前半为上表面）。这里重建轮廓时必须用同一套锚定，否则压力分布相对几何整体
    旋转（两种起点的弧长位置差 ~14% 周长），对法向积分出来的力是错的。
    下面 anchored_surface() 与 workstation_pod.wall_cp 的锚定逻辑逐行一致。
    """
    n_in, d, s = anchored_surface(name, task_root)
    Uinf = decode_sim(name)["U_inf"]
    p994 = np.interp(s, np.linspace(0, 1, len(cp_curve)), cp_curve) * (0.5 * Uinf ** 2)
    F = ((p994[:, None] * n_in) * d[:, None]).sum(axis=0)
    AoA = decode_sim(name)["AoA_deg"] * np.pi / 180
    Uinf = decode_sim(name)["U_inf"]
    basis = np.array([[np.cos(AoA), np.sin(AoA)],
                      [-np.sin(AoA), np.cos(AoA)]])
    Frot = basis @ F
    return 2 * Frot[1] / Uinf ** 2


def anchored_surface(name, task_root):
    """翼型表面在**前缘锚定序**下的 (normals (N,2), dl (N,), s (N,))。

    锚定规则与 workstation_pod.wall_cp(anchor_le=True) 完全一致：
      1) 闭环起点取 x 最小点（前缘）；
      2) 若走到一半时 y 在下降，说明先走了下表面 -> 整条翻转，
         保证前半=上表面、s=0 在前缘。
    """
    import vtkxml2 as vx
    from wall_force import _cycle_from_edges, _flat_int_da
    import xml.etree.ElementTree as ET

    fp = os.path.join(task_root, name, name + "_aerofoil.vtp")
    pts, pdata, _, _, _ = vx.read_vtkxml(fp)
    root = ET.parse(fp).getroot()
    piece = next(e for e in root.iter() if e.tag == "Piece")
    conn = off = None
    for da in piece.find("Lines").findall("DataArray"):
        if da.get("Name") == "connectivity":
            conn = _flat_int_da(da)
        elif da.get("Name") == "offsets":
            off = _flat_int_da(da)
    loop = _cycle_from_edges(conn, off, pts[:, :2])
    N = len(loop)
    X = np.vstack([pts[loop, :2], pts[loop[0], :2]])      # N+1 点（闭环）
    n_in = pdata["Normals"][loop][:, :2]                  # N 个法向
    le = int(np.argmin(X[:, 0]))                          # 前缘 = x 最小
    Xr = np.roll(X, -le, axis=0)
    Nr = len(Xr)
    reverse = Xr[Nr // 2, 1] < Xr[Nr // 2 - 1, 1]
    if reverse:
        Xr = Xr[::-1]
    nr = np.roll(n_in, -le, axis=0)
    if reverse:
        nr = nr[::-1]
    Xr_N = Xr[:N]
    d = np.linalg.norm(np.diff(np.vstack([Xr_N, Xr_N[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])[:-1]
    s = s / s[-1]
    return nr, d, s


def main():
    ap_task_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "02_dataset", "Dataset")
    ap_wallcp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "04_results", "mor", "p2_wallcp_anchored.npz")
    man = json.load(open(os.path.join(ap_task_root, "manifest.json"),
                         encoding="utf-8"))
    test_names = man["full_test"]
    log("test n=%d, task_root=%s, wallcp=%s" % (
        len(test_names), ap_task_root, ap_wallcp))

    cp_all = np.load(ap_wallcp, allow_pickle=True)
    CP = cp_all["CP"]; names_all = cp_all["names"].tolist()
    idx = [names_all.index(n) for n in test_names]
    cp_true_curve = CP[idx]  # (200, 256) 真值 Cp 曲线（前缘锚定）

    official_CL, official_CD = [], []
    pressure_CL = []
    # 官方力系数的压力/摩擦分解（"C_D 不报"的存档依据）
    pres_CD, pres_CL, fric_CD, fric_CL = [], [], [], []
    t0 = time.time()
    for i, nm in enumerate(test_names):
        intern = pv.read(os.path.join(ap_task_root, nm, nm + "_internal.vtu"))
        airfoil = pv.read(os.path.join(ap_task_root, nm, nm + "_aerofoil.vtp"))
        d = decode_sim(nm)
        Uinf, AoA = d["U_inf"], d["AoA_deg"]
        # 官方口径（逐字复刻 Compute_coefficients，含 Wss + p 项，真实法向+弧长）
        full, pres, fric = official_force_split(intern, airfoil, Uinf, AoA)
        official_CD.append(full[0]); official_CL.append(full[1])
        pres_CD.append(pres[0]); pres_CL.append(pres[1])
        fric_CD.append(fric[0]); fric_CL.append(fric[1])
        # 压力项版（缺 Wss，仅 p 项，真实法向+弧长）
        pcl = pressure_term_lift(nm, ap_task_root, cp_true_curve[i])
        pressure_CL.append(pcl)
        if (i + 1) % 20 == 0:
            log("  %d/%d (%.1f%%)  %.0fs" % (i + 1, len(test_names),
                                            100 * (i + 1) / len(test_names),
                                            time.time() - t0))

    official_CL = np.array(official_CL)
    official_CD = np.array(official_CD)
    pressure_CL = np.array(pressure_CL)
    pres_CD = np.array(pres_CD); pres_CL = np.array(pres_CL)
    fric_CD = np.array(fric_CD); fric_CL = np.array(fric_CL)
    # 摩擦项占比（"C_D 不报"的量化依据）
    fric_share_CD = float(np.abs(fric_CD).mean() / np.abs(official_CD).mean())
    fric_share_CL = float(np.abs(fric_CL).mean() / np.abs(official_CL).mean())
    # 只算压力项时，C_D / C_L 相对官方全量的逐样本相对误差
    cd_den = np.maximum(np.abs(official_CD), 1e-9)
    cl_den = np.maximum(np.abs(official_CL), 1e-9)
    pres_only_CD_relerr = float((np.abs(pres_CD - official_CD) / cd_den).mean())
    pres_only_CL_relerr = float((np.abs(pres_CL - official_CL) / cl_den).mean())
    # 逐样本相对差（分母=|官方真值|，0 值用全局均值兜底避免除 0）
    denom = np.where(np.abs(official_CL) < 1e-6,
                     max(1e-6, np.mean(np.abs(official_CL[official_CL != 0]))),
                     np.abs(official_CL))
    rel_diff = np.abs(pressure_CL - official_CL) / denom
    mean_rel_diff = float(rel_diff.mean())
    branch = "a" if mean_rel_diff < 0.05 else "b"

    # 官方真值内部校验（对称翼型 AoA=0 应该 C_L≈0）
    sym_mask = np.array([decode_sim(n)["AoA_deg"] == 0 and
                         decode_sim(n)["naca"][0] == 0 for n in test_names])
    sym_cl_check = float(np.abs(official_CL[sym_mask]).mean()) if sym_mask.any() else None

    out = {
        "branch": branch,
        "mean_pressure_vs_official_relerr": mean_rel_diff,
        "branch_threshold": 0.05,
        "n_test": len(test_names),
        "official_C_L_test_mean": float(official_CL.mean()),
        "official_C_D_test_mean": float(official_CD.mean()),
        "pressure_CL_test_mean": float(pressure_CL.mean()),
        "C_L_pressure_relerr_test": mean_rel_diff,
        "C_L_official_relerr_check": mean_rel_diff,
        "symmetric_zero_aoa_CL_abs_mean": sym_cl_check,
        # ---- "C_D 不报" 的存档依据（论文口径：力系数 = 压力项 + 壁面剪应力项）----
        "friction_split": {
            "method": ("官方 metrics.py Compute_coefficients 口径下，把力系数拆成"
                       " 压力项 (0 - WP_int) 与 摩擦项 (Wss_int)，再按 AoA 旋转、"
                       "乘 2/U∞²；真实法向 + 真实 cell 弧长"),
            "friction_share_of_C_D": fric_share_CD,
            "friction_share_of_C_L": fric_share_CL,
            "pressure_only_C_D_relerr_vs_official": pres_only_CD_relerr,
            "pressure_only_C_L_relerr_vs_official": pres_only_CL_relerr,
            "pressure_only_C_D_mean": float(pres_CD.mean()),
            "pressure_only_C_L_mean": float(pres_CL.mean()),
            "friction_only_C_D_mean": float(fric_CD.mean()),
            "friction_only_C_L_mean": float(fric_CL.mean()),
            "conclusion": ("B 链路只还原壁面 Cp，只能算压力项：压力项能还原 C_L（相对误差 "
                           "%.4f），但摩擦项占 C_D 的 %.1f%%，压力项版 C_D 相对官方差 %.1f%%，"
                           "因此 C_D 不报。"
                           % (pres_only_CL_relerr, 100 * fric_share_CD,
                              100 * pres_only_CD_relerr)),
        },
        "note": ("branch a: 压力项主导升力，报 C_L_pressure_relerr + 官方真值校验残差；"
                 "branch b: 删除 C_L/C_D，B 只报口径无关指标（逐样本 relL2/比值/Spearman）"),
        "per_sample": {
            "names": test_names,
            "official_CL": official_CL.tolist(),
            "official_CD": official_CD.tolist(),
            "pressure_CL": pressure_CL.tolist(),
            "pressure_only_CD": pres_CD.tolist(),
            "pressure_only_CL": pres_CL.tolist(),
            "friction_only_CD": fric_CD.tolist(),
            "friction_only_CL": fric_CL.tolist(),
        },
        "wallcp_source": ap_wallcp,
        "task_root": ap_task_root,
    }
    out_path = os.path.join(ROOT, "04_results", "prediction",
                            "judge_lifting_full.json")
    # 工作站上 ROOT/04_results/ 不存在；不建目录会在跑完 200 个样本后
    # 于最后一行写盘时 FileNotFoundError（"跑了几分钟然后失败"的典型症状）
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    log("DONE branch=%s mean_rel_diff=%.4f -> %s" % (
        branch, mean_rel_diff, out_path))
    log("symmetric_aoa0|CL|=%.4f (应≈0)" % (sym_cl_check or float("nan")))
    sys.exit(0 if branch == "a" else 0)


if __name__ == "__main__":
    main()

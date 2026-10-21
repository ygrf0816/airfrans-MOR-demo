"""单样本加载：从 .vtu/.vtp 重建官方 airfrans 12 列，不依赖 airfrans.dataset.load 全量加载。

官方 12 列（probe 已逐列核对）：
  0:2  pos          <- internal.vtu points[:, :2]
  2:4  inlet_vel    <- 样本级常量 = 文件名 (U_inf, AoA) 转 (Ux, Uy)
  4    sdf          <- internal.vtu implicit_distance 取负
  5:7  normals      <- aerofoil.vtp Normals 按坐标匹配（内部点 = 0）
  7:9  velocity     <- internal.vtu U[:, :2]
  9    p            <- internal.vtu p
  10   nu_t         <- internal.vtu nut
  11   surface      <- 坐标匹配 aerofoil.vtp 表面点
"""
import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

TASK_ROOT = "/home/zch/airfrans/data/Dataset"


def decode_sim(name):
    """文件名 -> 参数。airFoil2D_SST_{Uinf}_{AoA}_{camber}_{thick}_{Q}[可选字段]"""
    parts = name.split("_")
    vals = parts[2:]
    return {
        "U_inf": float(vals[0]),
        "AoA_deg": float(vals[1]),
        "camber": float(vals[2]),
        "t_c": float(vals[3]),
        "Q": float(vals[4]),
        "naca": vals[2:],
    }


def load_sample(name, task_root=TASK_ROOT):
    """单样本 -> dict(x(7), y(4), pos(2), surf(bool), sdf, U_local, p, nut)"""
    d = os.path.join(task_root, name)
    iv = pv.read(os.path.join(d, name + "_internal.vtu"))
    av = pv.read(os.path.join(d, name + "_aerofoil.vtp"))

    P = np.asarray(iv.points[:, :2], dtype=np.float64)
    N = P.shape[0]
    U = np.asarray(iv.point_data["U"], dtype=np.float64)[:, :2]
    p = np.asarray(iv.point_data["p"], dtype=np.float64)
    nut = np.asarray(iv.point_data["nut"], dtype=np.float64)
    sdf = -np.asarray(iv.point_data["implicit_distance"], dtype=np.float64)

    par = decode_sim(name)
    a = np.deg2rad(par["AoA_deg"])
    Ux, Uy = par["U_inf"] * np.cos(a), par["U_inf"] * np.sin(a)
    inlet = np.tile(np.array([Ux, Uy]), (N, 1))

    # 表面点匹配：法向 + surface bool
    sp = np.asarray(av.points[:, :2], dtype=np.float64)
    sn = np.asarray(av.point_data["Normals"], dtype=np.float64)[:, :2]
    tree = cKDTree(P)
    dist, idx = tree.query(sp)
    tol = 1e-6 * max(1.0, np.abs(P).max())
    hit = dist < tol
    idx_hit = idx[hit]
    normals = np.zeros((N, 2))
    normals[idx_hit] = sn[hit]
    surf = np.zeros(N, dtype=bool)
    surf[idx_hit] = True

    x = np.concatenate([P, inlet, sdf[:, None], normals], axis=1).astype(np.float32)
    y = np.concatenate([U, p[:, None], nut[:, None]], axis=1).astype(np.float32)
    return {"name": name, "x": x, "y": y, "pos": P.astype(np.float32),
            "surf": surf, "par": par,
            "mesh": iv, "airfoil": av,
            "match_dist_max": float(dist[hit].max()) if hit.any() else -1.0,
            "n_surf_matched": int(hit.sum())}


if __name__ == "__main__":
    import json
    man = json.load(open(os.path.join(TASK_ROOT, "manifest.json"), encoding="utf-8"))
    for nm in man["full_test"][:2]:
        s = load_sample(nm)
        print(nm, "N=%d surf=%d match_max=%.2e" % (
            s["x"].shape[0], s["surf"].sum(), s["match_dist_max"]))
        print("  x[0]=", np.round(s["x"][0], 4), " y[0]=", np.round(s["y"][0], 4))

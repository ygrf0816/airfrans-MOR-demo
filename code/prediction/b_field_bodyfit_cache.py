"""B 链路场预测（wall-fitted 版，第 1/2 步）：近壁贴体 (s,d) 带缓存。

问题（已量化）：原 256x128 均匀笛卡尔网格的格距 dx≈0.025（弦长比），而边界层第一层
网格只有 ~1e-4，均匀网格在近壁区**根本没有分辨率**。把真值场本身插值回 CFD 网格，
近壁带就已经错 5~28（归一化 MSE x1e-2），远大于 A 链路整场误差（1.4）。B 的模型
（25.07）其实已经打到自己表示方式的下限，瓶颈是表示，不是模型。

做法：给 B 的场表示加一个**贴体近壁带**，与原有笛卡尔网格组成双区表示：
  近壁带 d < D1(=0.06)：弦向弧长站位 NS=512 × 壁法向 ND=64（对数分布 1e-5→0.06），
    在每个站位沿外法向射线采样 CFD 场 -> (NS,ND) 定长图像；
  远场 d >= D1：沿用现有 256x128 笛卡尔网格（vol_fields_1000.npz，不动）。
评估时按每个 CFD 节点的 d 选区：d<D1 用 (s,d) 双线性回读，否则用笛卡尔三角插值。

闸门实测（64 个训练样本，近壁带内 692 万节点）：
  理想下限（真值直接回读）        5.430
  rank-16 截断后回读              5.033   （= 理想下限的 0.93 倍 -> 降维装得下）
  rank-32 截断后回读              5.367
对照：同一批节点上纯笛卡尔表示的下限约 27.8（6 测试样本）-> 贴体把表示误差降约 5 倍。

产物：/home/user/airfrans/pod/vol_fields_bodyfit_1000.npz
  names (1000,)            顺序 = manifest full_train + full_test（与 vol_fields_1000 一致）
  ns, nd, d0, d1           贴体带定义（本文件里的 u/v/p/nu_t **就是** (s,d) 带上的值，
                           与 vol_fields_1000.npz 里同名的笛卡尔网格场是两个文件、互不冲突）
  u/v/p/nu_t               (1000, ns*nd) float32  物理量原值（贴近壁面 -> 远至 d1）
  near_bad                 (1000, ns*nd) bool     射线打到翼型内部（实测 0）
  surf_perim               (1000,) float32        该样本翼型周长
"""
import os, json, time, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from bodyfit import order_surface      # noqa: E402  表面排序与朝向统一的唯一实现

DATA = "/home/user/airfrans/data/Dataset"
OUT = "/home/user/airfrans/pod/vol_fields_bodyfit_1000.npz"
NS, ND = 512, 64
D0, D1 = 1e-5, 0.06
FIELDS = ["u", "v", "p", "nu_t"]
NWORK = int(os.environ.get("NWORK", "24"))


def _order_surface(S):
    """与评估侧共用 bodyfit.order_surface（含弦向朝向统一），避免两边口径漂移。"""
    return order_surface(S)


def one(nm):
    try:
        import pyvista as pv
        from scipy.spatial import cKDTree, Delaunay
        from scipy.interpolate import LinearNDInterpolator
        m = pv.read(os.path.join(DATA, nm, nm + "_internal.vtu"))
        P = np.asarray(m.points, float)[:, :2]
        sd = np.asarray(m.point_data["implicit_distance"], float)
        U = np.asarray(m.point_data["U"], float)
        V = np.stack([U[:, 0], U[:, 1], np.asarray(m.point_data["p"], float),
                      np.asarray(m.point_data["nut"], float)], 1)

        w = pv.read(os.path.join(DATA, nm, nm + "_aerofoil.vtp"))
        Q = _order_surface(np.asarray(w.points, float)[:, :2])
        seg = np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1)
        arcl = np.concatenate([[0.0], np.cumsum(seg)])
        per = arcl[-1]
        s_surf = arcl[:-1] / per

        Qm = np.roll(Q, 1, 0); Qp = np.roll(Q, -1, 0)
        tg = Qp - Qm
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
        nrm = np.column_stack([tg[:, 1], -tg[:, 0]])
        sdi = LinearNDInterpolator(P, sd, fill_value=0.0)
        e = 1e-4
        if (sdi(Q[:, 0] + e * nrm[:, 0], Q[:, 1] + e * nrm[:, 1])
                > sdi(Q[:, 0] - e * nrm[:, 0], Q[:, 1] - e * nrm[:, 1])).mean() > 0.5:
            nrm = -nrm                      # 统一为外法向

        s_st = np.linspace(0, 1, NS, endpoint=False)
        Qs = np.column_stack([np.interp(s_st, s_surf, Q[:, 0]),
                              np.interp(s_st, s_surf, Q[:, 1])])
        Ns = np.column_stack([np.interp(s_st, s_surf, nrm[:, 0]),
                              np.interp(s_st, s_surf, nrm[:, 1])])
        dl = np.geomspace(D0, D1, ND)
        PTS = (Qs[:, None, :] + dl[None, :, None] * Ns[:, None, :]).reshape(-1, 2)
        tri = Delaunay(P)
        near = np.empty((NS * ND, 4), np.float32)
        for k in range(4):
            near[:, k] = LinearNDInterpolator(tri, V[:, k], fill_value=np.nan)(
                PTS[:, 0], PTS[:, 1])
        bsd = sdi(PTS[:, 0], PTS[:, 1])
        bad = (~np.isfinite(near).all(1)) | (bsd > 0)
        near = np.nan_to_num(near)
        return nm, near, bad, float(per)
    except Exception as ex:
        return nm, None, None, repr(ex)[:200]


def main():
    from multiprocessing import Pool
    man = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))
    names = list(man["full_train"]) + list(man["full_test"])
    print("[bodyfit %s] %d 样本  NS=%d ND=%d d=[%g,%g]  %d workers"
          % (time.strftime("%H:%M:%S"), len(names), NS, ND, D0, D1, NWORK), flush=True)
    t0 = time.time()
    got, errs = {}, []
    with Pool(NWORK) as pool:
        for i, (nm, near, bad, per) in enumerate(pool.imap_unordered(one, names, chunksize=2)):
            if near is None:
                errs.append((nm, per))
            else:
                got[nm] = (near, bad, per)
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs  ok=%d err=%d  load=%.1f"
                      % (i + 1, len(names), time.time() - t0, len(got), len(errs),
                         os.getloadavg()[0]), flush=True)

    names_o = [n for n in names if n in got]
    arrs = {k: np.stack([got[n][0][:, j] for n in names_o]) for j, k in enumerate(FIELDS)}
    badm = np.stack([got[n][1] for n in names_o])
    perim = np.array([got[n][2] for n in names_o], np.float32)
    np.savez(OUT, names=np.array(names_o), ns=NS, nd=ND, d0=D0, d1=D1,
             surf_perim=perim, near_bad=badm, **arrs)
    print("[bodyfit %s] DONE n=%d bad_cells=%d (%.6f) errors=%d  %.0fs -> %s  %.0f MB"
          % (time.strftime("%H:%M:%S"), len(names_o), int(badm.sum()), badm.mean(),
             len(errs), time.time() - t0, OUT, os.path.getsize(OUT) / 1e6), flush=True)
    for k in FIELDS:
        v = arrs[k]
        print("   %-5s mean=%+.5g std=%.5g min=%.5g max=%.5g"
              % (k, v.mean(), v.std(), v.min(), v.max()), flush=True)
    for e in errs[:5]:
        print("   ERR", e, flush=True)


if __name__ == "__main__":
    main()

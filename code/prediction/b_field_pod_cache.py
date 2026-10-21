"""B 链路场预测（第 1/3 步）：体积场公共网格缓存。

把 1000 个样本的 4 个场 (u, v, p, nu_t) 从各自的 .vtu 插值到 256x128 公共网格，
只保留**所有样本都有效**的格（交集掩膜，保证定长向量、无 NaN），存 npz。

口径沿用 P2-01 / workstation_pod.py：
  - bbox (-0.5, 4.23) x (-0.5, 1.62)（已核定的 union bbox）
  - implicit_distance > 0 视为翼型内部，剔除
  - 每个样本**只读一次 .vtu**，四个场共用一次读取与一次格点坐标

为什么不用已有的 vol_p_300x100.npz / vol_u_300x100.npz：
  只有 p 与 |U| 两个量、只有 400 样本，覆盖不了官方四任务划分
  （scarce 200/200、reynolds 504/496、aoa 804/196），无法做"每任务全字段对标"。

产物：/home/user/airfrans/pod/vol_fields_1000.npz
  names (1000,)  样本名，顺序 = manifest 的 full_train + full_test
  mask  (32768,) bool  交集掩膜，True 表示该格参与
  u/v/p/nu_t  (1000, mask.sum()) float32  物理量原值（未标准化）
  nx, ny, x0, x1, y0, y1  网格定义（便于还原到 (ny, nx) 图像）
"""
import os, json, time, sys
import numpy as np

DATA = "/home/user/airfrans/data/Dataset"
OUT = "/home/user/airfrans/pod/vol_fields_1000.npz"
NX, NY = 256, 128
# 必须用**全域** bbox：A 链路的 MSE 是在全部 ~18 万 CFD 网格节点上算的，
# 而 P2-01 那套 (-0.5,4.23)x(-0.5,1.62) 是近场裁剪，上游远场完全没覆盖，
# 拿覆盖不全的网格去和 A 的全局 MSE 并列会低估误差。
# 这里取覆盖整流场的 bbox（网格外的点自然是 NaN，会被交集掩膜剔除）。
X0, X1, Y0, Y1 = -2.16, 4.23, -1.65, 1.65
FIELDS = ["u", "v", "p", "nu_t"]
NWORK = int(os.environ.get("NWORK", "8"))

_xs = np.linspace(X0, X1, NX)
_ys = np.linspace(Y0, Y1, NY)
_GX, _GY = np.meshgrid(_xs, _ys, indexing="xy")
GRID = np.column_stack([_GX.ravel(), _GY.ravel()])
NG = GRID.shape[0]


def one(nm):
    """单样本 -> 4 个场的格点值（翼型内为 NaN）。只读一次 vtu。"""
    try:
        import pyvista as pv
        from scipy.interpolate import LinearNDInterpolator
        m = pv.read(os.path.join(DATA, nm, nm + "_internal.vtu"))
        P = m.points
        pts = (P[:, 0], P[:, 1])
        sd = m.point_data["implicit_distance"]
        U = m.point_data["U"]
        raw = {"u": U[:, 0], "v": U[:, 1],
               "p": m.point_data["p"], "nu_t": m.point_data["nut"]}
        ps = LinearNDInterpolator(pts, sd, fill_value=np.nan)(GRID[:, 0], GRID[:, 1])
        inside = (ps > 0) & np.isfinite(ps)
        out = {}
        for k, v in raw.items():
            g = LinearNDInterpolator(pts, v, fill_value=np.nan)(GRID[:, 0], GRID[:, 1])
            if inside.any():
                g[inside] = np.nan
            out[k] = g
        return nm, out
    except Exception as e:
        return nm, {"_err": repr(e)[:200]}


def main():
    from multiprocessing import Pool
    man = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))
    names = list(man["full_train"]) + list(man["full_test"])
    print("[cache %s] %d samples x %d fields, %d workers"
          % (time.strftime("%H:%M:%S"), len(names), len(FIELDS), NWORK), flush=True)
    t0 = time.time()
    got = {}
    errs = []
    with Pool(NWORK) as pool:
        for i, (nm, o) in enumerate(pool.imap_unordered(one, names, chunksize=2)):
            if "_err" in o:
                errs.append((nm, o["_err"]))
            else:
                got[nm] = o
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs  ok=%d err=%d"
                      % (i + 1, len(names), time.time() - t0, len(got), len(errs)), flush=True)

    names_o = [n for n in names if n in got]
    arrs = {k: np.stack([got[n][k] for n in names_o]) for k in FIELDS}
    valid = np.ones((len(names_o), NG), dtype=bool)
    for k in FIELDS:
        valid &= np.isfinite(arrs[k])
    col = valid.all(axis=0)
    out = {k: arrs[k][:, col].astype(np.float32) for k in FIELDS}
    np.savez(OUT, names=np.array(names_o), mask=col, nx=NX, ny=NY,
             x0=X0, x1=X1, y0=Y0, y1=Y1, **out)
    print("[cache %s] DONE n=%d valid_cells=%d/%d (%.3f) errors=%d  %.0fs -> %s"
          % (time.strftime("%H:%M:%S"), len(names_o), int(col.sum()), NG,
             col.mean(), len(errs), time.time() - t0, OUT), flush=True)
    for k in FIELDS:
        v = out[k]
        print("   %-5s mean=%+.5g std=%.5g min=%.5g max=%.5g"
              % (k, v.mean(), v.std(), v.min(), v.max()), flush=True)
    for e in errs[:5]:
        print("   ERR", e, flush=True)


if __name__ == "__main__":
    main()

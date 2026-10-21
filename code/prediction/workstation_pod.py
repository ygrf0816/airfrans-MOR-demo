"""工作站上做的事：解压 9GB zip + 3 口径 POD 全量重算（B 链路目标坐标）。
运行在 user@<host>，依赖 /home/user/.conda/envs/pytorch + pyvista 0.49。

口径严格对齐 P2-02 / P3-01 已验证结论（不另起炉灶）：
- 壁面 Cp：p2_grid_wall.wall_cp（弧长归一化 s∈[0,1]，Cp=p_wall/q，q=0.5U∞²，
  前缘锚定由 cycle_from_edges 重建闭环 + s 前缘=0）
- 体积 p / |U|：P2-01 256×128 公共网格插值 + 壁面掩膜（implicit_distance>0 剔除）

输出 3 个 npz 到 /home/user/airfrans/pod/：
  wallcp_1000.npz / vol_p_300x100.npz / vol_u_300x100.npz
  每个含：基 Vt[:r]、S、mu、split_idx（官方 full 800/200 或 300/100 子集）

上传前需确保工作站 /home/user/airfrans/prediction/ 下有：
  vtkxml2.py / wall_force.py（P2-02 依赖，remote.py 已传）
本脚本 main 流程：unzip -> 读 manifest -> 3 口径 POD -> 各存 npz。
"""
import os, sys, json, time, glob
import numpy as np
import pyvista as pv
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/user/airfrans"
DATA = os.path.join(ROOT, "data")
DS_ZIP = os.path.join(DATA, "Dataset.zip")
DS_DIR = os.path.join(DATA, "Dataset")
POD_OUT = os.path.join(ROOT, "pod")
PRED = os.path.join(ROOT, "prediction")
os.makedirs(POD_OUT, exist_ok=True)
# 复用 P2 自算口径的读库
sys.path.insert(0, PRED)


def log(msg):
    print("[workstation_pod %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def extract_zip():
    if os.path.isdir(DS_DIR) and len(glob.glob(os.path.join(DS_DIR, "airFoil2D*"))) > 0:
        log("Dataset/ already extracted, skipping unzip")
        return
    log("Unzipping 9GB Dataset.zip -> %s (~20-40 min)" % DS_DIR)
    t0 = time.time()
    import zipfile
    with zipfile.ZipFile(DS_ZIP, "r") as z:
        for info in z.infolist():
            nm = info.filename
            if ("_internal.vtu" in nm or "_aerofoil.vtp" in nm or
                    "_freestream.vtp" in nm or nm.endswith("/")):
                z.extract(info, DATA)
    log("unzip done in %.1f min, n_dir=%d" %
        ((time.time() - t0) / 60,
         len(glob.glob(os.path.join(DS_DIR, "airFoil2D*")))))


# ============ P2-02 口径：壁面 Cp(s) ============
# 原样移植 03_code/p1/p2_grid_wall.py 的 wall_cp 逻辑（弧长归一化 + p/q）
# 依赖：vtkxml2.read_vtkxml / wall_force._cycle_from_edges
from vtkxml2 import b64_stream, _DT  # noqa: E402
from wall_force import _cycle_from_edges  # noqa: E402
import xml.etree.ElementTree as ET
import zlib, struct


def _lines(fp):
    """读 .vtp 的 <Lines> 边表 -> (conn, off)。复用 p2_grid_wall._lines。"""
    root = ET.parse(fp).getroot()
    piece = next(e for e in root.iter() if e.tag == "Piece")
    conn = off = None
    for da in piece.find("Lines").findall("DataArray"):
        raw = b64_stream(da.text or "")
        nb = struct.unpack_from("<I", raw, 0)[0]
        pos, o = 12, bytearray()
        for _ in range(max(1, nb)):
            cl = struct.unpack_from("<I", raw, pos)[0]; pos += 4
            o += zlib.decompress(raw[pos:pos + cl]); pos += cl
        arr = np.frombuffer(bytes(o), dtype=np.dtype(_DT[da.get("type")]))
        if da.get("Name") == "connectivity":
            conn = arr
        elif da.get("Name") == "offsets":
            off = arr
    return conn, off


def Uinf(sim):
    return float(sim.replace("airFoil2D_SST_", "").split("_")[0])


def wall_cp(sim, n=256, anchor_le=True):
    """P2-02 口径：翼型轮廓 n 个弧长归一化点，Cp = p_wall / (0.5 U∞²)。

    p 取自 _aerofoil.vtp 轮廓点自身（官方 wall 面）。

    前缘锚定（anchor_le=True，默认）：
      - 闭环起点取 x 最小点（前缘 LE）；
      - 若沿闭环走到一半时 y < 前一半的 y（即先走下表面），整条曲线翻转，
        保证前半=上表面、s=0 在前缘。
      - 这是 P2-02 已核实的 p2_wallcp_anchored.npz 口径（r99=6/r999=10），
        不锚定时（_cycle_from_edges 的 min(node) 起点）r99=25/r999=37（rank 差
        一个数量级，02_pod_energy.json 已记录）。
    """
    fp = os.path.join(DS_DIR, sim, sim + "_aerofoil.vtp")
    pts, pdata, _, _, _ = __import__("vtkxml2").read_vtkxml(fp)
    conn, off = _lines(fp)
    loop = _cycle_from_edges(conn, off, pts[:, :2])
    N = len(loop)
    X = np.vstack([pts[loop, :2], pts[loop[0], :2]])  # N+1 点（闭环闭合）
    p_wall = pdata["p"][loop]  # N 个值
    if not anchor_le:
        seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        s = s / s[-1]
        q = 0.5 * Uinf(sim) ** 2
        cp_wall = p_wall / q
        return np.interp(np.linspace(0, 1, n), s[:len(cp_wall)], cp_wall)
    # 前缘锚定：LE = argmin x（X 含闭合重复点，N+1 个）
    le = int(np.argmin(X[:, 0]))
    Xr = np.roll(X, -le, axis=0)  # index 0 = LE
    Nr = len(Xr)
    # 前半应走表面：若 Xr[Nr//2].y < Xr[Nr//2-1].y 说明先走了下表面 -> 翻转
    reverse = Xr[Nr // 2, 1] < Xr[Nr // 2 - 1, 1]
    if reverse:
        Xr = Xr[::-1]
    pr = np.roll(p_wall, -le)  # N 个值
    if reverse:
        pr = pr[::-1]
    Xr_N = Xr[:N]  # 去掉闭合重复点，N 个点
    seg = np.linalg.norm(
        np.diff(np.vstack([Xr_N, Xr_N[:1]]), axis=0), axis=1)  # N 段（闭环）
    s = np.concatenate([[0], np.cumsum(seg)])
    s = s[:-1]  # N 个值，和 pr 对齐
    s = s / s[-1]
    q = 0.5 * Uinf(sim) ** 2
    cp = pr / q
    return np.interp(np.linspace(0, 1, n), s, cp)


# ============ P2-01 口径：体积场 256×128 公共网格 ============
from scipy.interpolate import LinearNDInterpolator  # noqa: E402

NX, NY = 256, 128
X0, X1, Y0, Y1 = -0.5, 4.23, -0.5, 1.62  # P2-01 已核定的 union bbox
xs = np.linspace(X0, X1, NX)
ys = np.linspace(Y0, Y1, NY)
GX, GY = np.meshgrid(xs, ys, indexing="xy")
GRID = np.column_stack([GX.ravel(), GY.ravel()])


def grid_sample(sim, kind):
    """P2-01 口径：公共网格插值 + 壁面掩膜（implicit_distance>0 ⇒ 翼型内 ⇒ 剔除）。
    kind: 'p' 静压 / 'u' 速度模长 |U|。
    """
    fp = os.path.join(DS_DIR, sim, sim + "_internal.vtu")
    m = pv.read(fp)
    P = m.points
    sd = m.point_data["implicit_distance"]
    vals = m.point_data["p"] if kind == "p" else np.hypot(
        m.point_data["U"][:, 0], m.point_data["U"][:, 1])
    v = LinearNDInterpolator((P[:, 0], P[:, 1]), vals,
                             fill_value=np.nan)(GRID[:, 0], GRID[:, 1])
    ps = LinearNDInterpolator((P[:, 0], P[:, 1]), sd,
                              fill_value=np.nan)(GRID[:, 0], GRID[:, 1])
    inside = (ps > 0) & np.isfinite(ps)
    mask = np.isfinite(v) & ~inside
    v[mask] = v[mask]
    v[~mask] = np.nan
    return v, mask


def pod_full_svd(arr, n_tr):
    """train（前 n_tr 行）做 SVD，返回 Vt[:64] / S / mu / tr_idx / te_idx。"""
    mu = arr[:n_tr].mean(axis=0)
    U, S, Vt = np.linalg.svd(arr[:n_tr] - mu, full_matrices=False)
    tr_idx = list(range(n_tr))
    te_idx = list(range(n_tr, arr.shape[0]))
    return Vt[:64], S, mu, tr_idx, te_idx


def main():
    extract_zip()
    man = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))
    log("manifest: full_train=%d full_test=%d" %
        (len(man["full_train"]), len(man["full_test"])))

    # ============ 1) 壁面 Cp 全量 1000 ============
    all_names = man["full_train"] + man["full_test"]
    log("wall Cp: reading %d aerofoil.vtp (~30 min)" % len(all_names))
    t0 = time.time()
    CP = np.zeros((len(all_names), 256), dtype=np.float64)
    for i, sim in enumerate(all_names):
        CP[i] = wall_cp(sim, anchor_le=True)
        if i % 200 == 0:
            log("  wallcp %d/%d (%.1f%%)" % (i, len(all_names),
                                               100.0 * i / len(all_names)))
    log("wall Cp done, %.1f min" % ((time.time() - t0) / 60))
    Vt, S, mu, tr, te = pod_full_svd(CP, n_tr=len(man["full_train"]))
    np.savez(os.path.join(POD_OUT, "wallcp_1000.npz"),
             CP=CP, names=np.array(all_names),
             Vt_first64=Vt, S=S, mu=mu,
             tr_idx=np.array(tr), te_idx=np.array(te),
             split="full_800_200", anchor="leading_edge")
    log("wallcp_1000.npz written (LE-anchored), S[0:8]=%s" %
        np.round(S[:8], 3))

    # ============ 2) 体积 p / 体积 |U| 全量 300/100（P2-01 口径）============
    ft = list(man["full_train"])[:300]
    fte = list(man["full_test"])[:100]
    sel = ft + fte
    log("volume p/|U|: reading+interp %d samples (~40 min, P2-01 256×128 网格)"
        % len(sel))
    for kind, out_nm in [("p", "vol_p_300x100.npz"),
                         ("u", "vol_u_300x100.npz")]:
        t0 = time.time()
        arr = np.full((len(sel), GRID.shape[0]), np.nan, dtype=np.float64)
        valid = np.zeros((len(sel), GRID.shape[0]), dtype=bool)
        for i, sim in enumerate(sel):
            v, mask = grid_sample(sim, kind)
            arr[i] = v
            valid[i] = mask
            if i % 50 == 0:
                log("  vol_%s %d/%d (%.1f%%)" %
                    (kind, i, len(sel), 100.0 * i / len(sel)))
        n_valid = int(valid.mean() * GRID.shape[0])
        # 只保留每样本相同的有效格集合（并集掩膜，避免每样本形状不一致）
        col_valid = valid.all(axis=0)  # 全部样本都有效的格
        arr_v = arr[:, col_valid]
        Vt, S, mu, tr, te = pod_full_svd(
            np.nan_to_num(arr_v, nan=0.0), n_tr=len(ft))
        np.savez(os.path.join(POD_OUT, out_nm),
                 P=arr_v.astype(np.float32) if kind == "p"
                 else arr_v.astype(np.float32),
                 valid_frac=float(col_valid.mean()),
                 Vt_first32=Vt[:32], S=S, mu=mu,
                 tr_idx=np.array(tr), te_idx=np.array(te),
                 train_n=len(ft), test_n=len(fte),
                 split="full_300_100",
                 nx=NX, ny=NY, kind=kind)
        log("%s done, %.1f min, valid_frac=%.3f, S[0:16]=%s"
            % (out_nm, (time.time() - t0) / 60, col_valid.mean(),
               np.round(S[:16], 3)))

    log("ALL POD DONE. /home/user/airfrans/pod/:")
    for f in sorted(os.listdir(POD_OUT)):
        log("  %s  %.2f MB" % (f,
                               os.path.getsize(os.path.join(POD_OUT, f)) / 1e6))
    log("workstation_pod COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        log("FATAL\n" + traceback.format_exc())
        sys.exit(1)

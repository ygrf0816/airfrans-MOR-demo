"""把 B 链路的 256 点 Cp 曲线（前缘锚定、弧长归一化）放到弦向 x/c 横轴上。

关键：曲线值来自传入的 `b_cp256`（POD 还原的预测），几何只用来提供
"每个弧长站位对应哪个 x/c"，绝不从 .vtp 读压力（那是真值）。
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vtkxml2 as vx  # noqa: E402
from wall_force import _cycle_from_edges, _flat_int_da  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402


def _lines(fp):
    root = ET.parse(fp).getroot()
    piece = next(e for e in root.iter() if e.tag == "Piece")
    conn = off = None
    for da in piece.find("Lines").findall("DataArray"):
        if da.get("Name") == "connectivity":
            conn = _flat_int_da(da)
        elif da.get("Name") == "offsets":
            off = _flat_int_da(da)
    return conn, off


def anchored_geometry(name, task_root):
    """返回前缘锚定序下的 (X (N,2) 坐标, s (N,) 弧长归一化, x0, x1)。
    锚定规则 = workstation_pod.wall_cp(anchor_le=True)：s=0 在前缘、前半上表面。
    """
    fp = os.path.join(task_root, name, name + "_aerofoil.vtp")
    pts, _, _, _, _ = vx.read_vtkxml(fp)
    conn, off = _lines(fp)
    loop = _cycle_from_edges(conn, off, pts[:, :2])
    N = len(loop)
    Xc = np.vstack([pts[loop, :2], pts[loop[0], :2]])
    le = int(np.argmin(Xc[:, 0]))
    Xr = np.roll(Xc, -le, axis=0)
    Nr = len(Xr)
    reverse = Xr[Nr // 2, 1] < Xr[Nr // 2 - 1, 1]
    if reverse:
        Xr = Xr[::-1]
    Xr_N = Xr[:N]
    seg = np.linalg.norm(np.diff(np.vstack([Xr_N, Xr_N[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])[:-1] / np.cumsum(seg)[-1]
    return Xr_N, s, Xr_N[:, 0].min(), Xr_N[:, 0].max()


def b_curve_to_xchord(name, task_root, b_cp256, n_out=64):
    """b_cp256 (256,) 前缘锚定、弧长归一化的 Cp 曲线（来自 POD 还原）。
    返回 (xc[n_out], cp_up, cp_lo, x0, x1)：
      xc = 线性 x/c 站位；cp_up/cp_lo = 该站位上下表面的 B 预测 Cp。
    做法：256 点各自弧长站位 -> 插值取该处 x；再按 x 线性站位在 994 点几何上
    取上下表面法向位置，对 256 点曲线做插值。
    """
    Xr_N, s, x0, x1 = anchored_geometry(name, task_root)
    s256 = np.linspace(0, 1, len(b_cp256))
    x_at_s256 = np.interp(s256, s, Xr_N[:, 0])       # 256 点各自的物理 x
    ys_at_s256 = np.interp(s256, s, Xr_N[:, 1])      # 各自的 y（判上下表面）
    c = x1 - x0
    xs_out = np.linspace(0, 1, n_out)
    cp_up = np.full(n_out, np.nan)
    cp_lo = np.full(n_out, np.nan)
    for i, xx in enumerate(xs_out * c + x0):
        band = np.abs(x_at_s256 - xx) < c / n_out * 0.5
        if band.sum() == 0:
            continue
        ysb = ys_at_s256[band]
        cpb = b_cp256[band]
        # 同一 x 附近可能同时有上下表面；用 y 的符号/极值分
        if band.sum() == 1:
            cp_up[i] = cp_lo[i] = cpb[0]
        else:
            cp_up[i] = cpb[np.argmax(ysb)]
            cp_lo[i] = cpb[np.argmin(ysb)]
    return xs_out, cp_up, cp_lo, x0, x1


def main():
    import json
    task_root = "/home/user/airfrans/data/Dataset"
    pod = np.load("/home/user/airfrans/pod/wallcp_1000.npz", allow_pickle=True)
    CP, names = pod["CP"], pod["names"].tolist()
    man = json.load(open("/home/user/airfrans/data/manifest.json", encoding="utf-8"))
    for nm in man["full_test"][:3]:
        i = names.index(nm)
        xs, up, lo, x0, x1 = b_curve_to_xchord(nm, task_root, CP[i])
        print(nm, "x0,x1=", round(x0, 4), round(x1, 4),
              "up[:3]=", np.round(up[:3], 3), "lo[:3]=", np.round(lo[:3], 3))
    print("B_XCHORD_OK")


if __name__ == "__main__":
    main()

"""P2-01 MOR 公共网格 + 壁面掩膜 + 逐样本插值（自算，数据源 02_dataset/Dataset/*_internal.vtu）。
口径（自算 vs 对照见 09_p2_mor.md）：
  主线 = 壁面 Cp(s)：翼型轮廓 256 弧长归一化点 s∈[0,1]，Cp = p/(0.5 U∞²)（p 运动学 p/ρ，P1-05 已核实）。
        壁面 p 取自 _aerofoil.vtp 轮廓点自身（官方 wall 面，比 body 内插值更干净），几何=轮廓。
  对照 = 体积场 p / |U|：本脚本自建公共 NX×NY 规则网格（union bbox），
        用 scipy LinearNDInterpolator 逐样本插值，再**壁面掩膜**（该样本 implicit_distance>0 ⇒ 翼型内 ⇒ 剔除）。
  POD 拟合在 train（full_train）；报数在 test（红线：test 只报数）。
  重构误差两种归一化（随 r 单调下降）：
     relL2_centered = ||(X - mean_train) - P_r(X - mean_train)|| / ||X - mean_train||   （脉动归一）
     relL2_full     = ||X - mean_train - P_r(X - mean_train)|| / ||X - mean_train|| 同一公式，
     其中 P_r = V_r V_r^T 是前 r 个右奇异向量的正交投影。即 1-cum_energy 的 L2 形式。
"""
import os, sys, json, time, collections, zlib, struct
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'probe')))
import vtkxml2 as vx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wall_force import _cycle_from_edges
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
DATASET = os.path.join(ROOT, '02_dataset', 'Dataset')
OUT = os.path.join(ROOT, '04_results', 'analysis')
man = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
ALL = sorted(set().union(*[set(v) for v in man.values()]))
assert len(ALL) == 1000

NX, NY = 192, 96
X0, X1, Y0, Y1 = -0.5, 4.23, -0.5, 1.62
xs = np.linspace(X0, X1, NX); ys = np.linspace(Y0, Y1, NY)
GX, GY = np.meshgrid(xs, ys, indexing='xy')
GRID = np.column_stack([GX.ravel(), GY.ravel()])

def Uinf(sim):
    return float(sim.split('_')[2])

def _lines(vtp):
    root = ET.parse(vtp).getroot()
    piece = next(e for e in root.iter() if e.tag == 'Piece')
    conn = off = None
    for da in piece.find('Lines').findall('DataArray'):
        raw = vx.b64_stream(da.text or ''); nb = struct.unpack_from('<I', raw, 0)[0]
        pos, o = 12, bytearray()
        for _ in range(max(1, nb)):
            cl = struct.unpack_from('<I', raw, pos)[0]; pos += 4
            o += zlib.decompress(raw[pos:pos+cl]); pos += cl
        arr = np.frombuffer(bytes(o), dtype=np.dtype(vx._DT[da.get('type')]))
        if da.get('Name') == 'connectivity': conn = arr
        elif da.get('Name') == 'offsets': off = arr
    return conn, off

def wall_cp(sim, n=256):
    """壁面 Cp(s)：翼型轮廓 n 个弧长归一化点。p 取轮廓点自身（_aerofoil.vtp）。"""
    fp = os.path.join(DATASET, sim, sim + '_aerofoil.vtp')
    pts, pdata, _, _, _ = vx.read_vtkxml(fp)
    conn, off = _lines(fp)
    loop = _cycle_from_edges(conn, off, pts[:, :2])
    X = np.vstack([pts[loop, :2], pts[loop[0], :2]])
    p_wall = pdata['p'][loop]
    seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    s = s / s[-1]
    q = 0.5 * Uinf(sim)**2
    cp_wall = p_wall / q
    # p_wall 有 len(loop) 个（闭环），s 有 len(loop)+1 个（含闭合重复首点）
    # 取 s[:len(cp_wall)] 与之对齐
    return np.interp(np.linspace(0, 1, n), s[:len(cp_wall)], cp_wall)

def grid_sample(sim):
    """公共网格插值 p 与 |U|，返回 (p_g, Umag_g, mask_valid, t_read_s, t_interp_s)。"""
    fp = os.path.join(DATASET, sim, sim + '_internal.vtu')
    t0 = time.time(); m = pv.read(fp); P = m.points; t_read = time.time() - t0
    p = m.point_data['p']; U = m.point_data['U']; sd = m.point_data['implicit_distance']
    Umag = np.hypot(U[:, 0], U[:, 1])
    t1 = time.time()
    pp = LinearNDInterpolator((P[:,0], P[:,1]), p, fill_value=np.nan)(GRID[:,0], GRID[:,1])
    pu = LinearNDInterpolator((P[:,0], P[:,1]), Umag, fill_value=np.nan)(GRID[:,0], GRID[:,1])
    ps = LinearNDInterpolator((P[:,0], P[:,1]), sd, fill_value=np.nan)(GRID[:,0], GRID[:,1])
    t_interp = time.time() - t1
    inside = (ps > 0) & np.isfinite(ps)
    mask = np.isfinite(pp) & np.isfinite(pu) & ~inside
    return pp, pu, mask, t_read, t_interp

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', default='all', choices=['wallcp', 'grid', 'all'])
    a = ap.parse_args()
    if a.what in ('wallcp', 'all'):
        t0 = time.time()
        CP = np.zeros((1000, 256))
        for i, sim in enumerate(ALL):
            CP[i] = wall_cp(sim)
            if (i+1) % 100 == 0:
                print(f'  wallcp {i+1}/1000 {time.time()-t0:.0f}s', flush=True)
        np.savez(os.path.join(OUT, 'p2_wallcp_self.npz'), CP=CP, names=np.array(ALL))
        print(f'-> p2_wallcp_self.npz  ({time.time()-t0:.0f}s)')
    if a.what in ('grid', 'all'):
        # 子集：full_train 前 300 + full_test 前 100（按 ALL 字典序取，避免重叠=full_test 与 full_train 不相交）
        ft = list(man['full_train']); fte = list(man['full_test'])
        sel = ft[:300] + fte[:100]
        Pm = np.full((len(sel), GRID.shape[0]), np.nan)
        Um = np.full((len(sel), GRID.shape[0]), np.nan)
        V = np.zeros((len(sel), GRID.shape[0]), dtype=bool)
        tot_read = tot_interp = 0.0
        for i, sim in enumerate(sel):
            pp, pu, mask, tr, ti = grid_sample(sim)
            Pm[i] = pp; Um[i] = pu; V[i] = mask
            tot_read += tr; tot_interp += ti
            if (i+1) % 40 == 0:
                print(f'  grid {i+1}/{len(sel)} interp_sum={tot_interp:.1f}s read_sum={tot_read:.1f}s', flush=True)
        np.savez(os.path.join(OUT, 'p2_grid_self.npz'),
                 P=np.ascontiguousarray(Pm), U=np.ascontiguousarray(Um),
                 valid=np.ascontiguousarray(V), names=np.array(sel),
                 xs=np.ascontiguousarray(xs), ys=np.ascontiguousarray(ys),
                 grid_x=np.ascontiguousarray(GRID[:,0]), grid_y=np.ascontiguousarray(GRID[:,1]),
                 interp_s_per_sample=float(tot_interp/len(sel)),
                 read_s_per_sample=float(tot_read/len(sel)))
        print(f'-> p2_grid_self.npz  interp={tot_interp/len(sel):.2f}s/sample read={tot_read/len(sel):.2f}s/sample')

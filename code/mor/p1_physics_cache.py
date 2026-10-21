"""P1-05 物理校验 Phase 1：把每个 sim 的物理量算好缓存到 physics_cache.npz。
  - p 量纲判据（远场静压 vs 0 / 101325 / 101325/RHO）
  - 壁面压力积分 → Fx, Fy（用于 Cl/Cd）
  - Re 与边界层
  读盘 475 个 VTK 一次，~90s。
"""
import os, sys, json, math, time
import numpy as np
import pyvista as pv
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SAMPLES = os.path.join(ROOT, '04_results', 'analysis', 'samples')
OUT = os.path.join(ROOT, '04_results', 'analysis')
cache = os.path.join(OUT, 'physics_cache.npz')
P_REF = 1.01325e5   # Pa
RHO   = 1.1839      # kg/m3, airfrans T=298.15K
NU    = 1.56e-5

def decode(sim):
    p = sim.replace('airFoil2D_SST_', '').split('_')
    return {'U_inf': float(p[0]), 'AoA_deg': float(p[1]),
            'Re': float(p[0]) / NU,
            'naca': [float(x) for x in p[2:]],
            'series': '5digit' if len(p) >= 6 else '4digit'}

def wall_pressure_force(pts, p_vals):
    """在 internal.vtu 的壁面点（surface=True）上，把点按极角排序成闭合多边形，
    用 −∮ p n dl（n 为局部外法向）做 2D 压力力积分。pts:(N,2), p_vals:(N,)。
    保守/可复现：不依赖 aerofoil.vtp 的三角形连接（它是点云 n_cells==n_points）。"""
    P = np.asarray(pts, float)
    p = np.asarray(p_vals, float)
    cx, cy = P.mean(0)
    ang = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
    order = np.argsort(ang)
    P = P[order]; p = p[order]
    N = len(P)
    Pn = np.roll(P, -1, axis=0)          # 下一点
    # 有向面积（带符号，定内外法向）
    A2 = (P[:, 0] * Pn[:, 1] - Pn[:, 0] * P[:, 1])
    signed_area = 0.5 * A2.sum()
    # 每条边的法向（2D）：对边向量 e=(ex,ey)，法向 n=(ey, -ex) 或 (-ey, ex)；
    # 取“向外”：与多边形朝向一致。用 e 旋转 -90deg = (ey, -ex)。
    e = Pn - P
    # 外法向（顺时针多边形为外）：n = (dy, -dx) 归一
    edy, edx = e[:, 1], e[:, 0]
    n = np.stack([edy, -edx], axis=1)
    ln = np.linalg.norm(n, axis=1)
    n = np.where(ln[:, None] > 0, n / (ln[:, None] + 1e-300), 0.0)
    if signed_area > 0:   # 逆时针：外法向要取反
        n = -n
    p_edge = 0.5 * (p + np.roll(p, -1))
    F = -(p_edge[:, None] * n * ln[:, None]).sum(axis=0)
    return F[:2]

m = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
import collections
sim_splits = collections.defaultdict(set)
for split, names in m.items():
    for n in names: sim_splits[n].add(split)

sims = [os.path.basename(d) for d in sorted(os.listdir(SAMPLES))
        if os.path.isdir(os.path.join(SAMPLES, d)) and os.path.basename(d) in sim_splits]
sims = [s for s in sims if os.path.exists(os.path.join(SAMPLES, s, s + '_internal.vtu'))]
sims = [s for s in sims if os.path.exists(os.path.join(SAMPLES, s, s + '_aerofoil.vtp'))]
print(f'参与物理校验的 sim: {len(sims)}', flush=True)

data = {}
t0 = time.time()
n_ok = 0
sim_list = []
for i, sim in enumerate(sims, 1):
    d = decode(sim)
    internal = pv.read(os.path.join(SAMPLES, sim, sim + '_internal.vtu'))
    p_int = np.array(internal.point_data['p'])
    U_int = np.array(internal.point_data['U'])
    p_static = float(np.percentile(p_int, 98))
    p_med = float(np.median(p_int))
    p_min = float(p_int.min())
    p_max = float(p_int.max())
    # 弦长（internal 包围盒 x 范围，翼型在弦 1m 归一）
    P2 = internal.points[:, :2]
    chord = float(P2[:, 0].max() - P2[:, 0].min())
    # 壁面力：在 internal 的壁面点（surface=U[:,0]==0）上做闭合多边形积分
    U_int = np.array(internal.point_data['U'])
    surf_mask = (U_int[:, 0] == 0)
    if surf_mask.sum() < 8:
        print(f'  ! {sim} 壁面点不足 {surf_mask.sum()}', flush=True); continue
    surf_pts = internal.points[surf_mask, :2]
    surf_p = np.array(internal.point_data['p'])[surf_mask]
    Fx, Fy = wall_pressure_force(surf_pts, surf_p)
    rho = RHO; Uinf = d['U_inf']
    q = 0.5 * rho * Uinf * Uinf
    Cl = Fy / q if q > 0 else 0.0
    Cd = -Fx / q if q > 0 else 0.0
    AoA = math.radians(d['AoA_deg'])
    Cl_thin = 2 * math.pi * AoA
    data.update({
        f'd{ i:03d}_Uinf': Uinf, f'd{i:03d}_AoA': d['AoA_deg'], f'd{i:03d}_Re': d['Re'],
        f'd{i:03d}_p_static': p_static, f'd{i:03d}_p_med': p_med,
        f'd{i:03d}_p_min': p_min, f'd{i:03d}_p_max': p_max,
        f'd{i:03d}_chord': chord, f'd{i:03d}_Fx': Fx, f'd{i:03d}_Fy': Fy,
        f'd{i:03d}_Cl': Cl, f'd{i:03d}_Cd': Cd, f'd{i:03d}_Cl_thin': Cl_thin,
        f'd{i:03d}_q': q,
    })
    n_ok += 1
    sim_list.append(sim)
    if i % 60 == 0:
        print(f'  [{i}/{len(sims)}] {time.time()-t0:.0f}s', flush=True)
np.savez(cache, **data, sim_names=np.array(sim_list, dtype=object))
print(f'缓存写出 {cache} keys={len(data)}+sim_names ({time.time()-t0:.0f}s), ok={n_ok}', flush=True)
print('DONE_PHYSICS_P1')

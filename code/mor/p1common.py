"""P1 公共读取层：pyvista 读 _internal.vtu（含 cell_data）+ 文件名解码。
所有分析脚本 import 这里，保证口径一致（= 官方 airfrans.load() 的字段口径）。
"""
import os, sys, json, numpy as np
import pyvista as pv
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SAMPLES = os.path.join(ROOT, '04_results', 'analysis', 'samples')
DATASET = os.path.join(ROOT, '02_dataset', 'Dataset')
SEL = os.path.join(ROOT, '04_results', 'analysis', 'selected_names.json')
PREFIX = 'airFoil2D_SST_'

RHO = 1.1839      # kg/m3, airfrans 库 T=298.15K 值
NU_REF = 1.56e-5  # m2/s, 官方声明数据集生成时固定


def _internal_path(sim, src):
    return os.path.join(src, sim, f'{sim}_internal.vtu')


def decode(name):
    """文件名 -> dict: U_inf, AoA, naca(list 3 或 4 段), series(4或5)。"""
    p = [float(x) for x in name[len(PREFIX):].split('_')]
    naca = p[2:]
    return {
        'U_inf': p[0], 'AoA_deg': p[1],
        'naca': naca, 'series': '5digit' if len(naca) == 4 else '4digit',
        'Re': p[0] / NU_REF,   # 弦长 1m
    }


def load_internal(sim, src=SAMPLES):
    """读 _internal.vtu，返回 dict（点级 + 解码 + 派生）。src 可指 475 子样本或全量 Dataset。"""
    d = decode(sim)
    fp = _internal_path(sim, src)
    m = pv.read(fp)
    U = m.point_data['U']                  # (N,3)
    p = m.point_data['p']                  # (N,)
    nut = m.point_data['nut']
    sd = m.point_data['implicit_distance']
    N = m.n_points
    # 派生（官方口径）
    surface = (U[:, 0] == 0)
    sdf = -sd
    velocity = U[:, :2]
    return {
        'name': sim, **d, 'N': N,
        'pts': m.points, 'U': U, 'p': p, 'nut': nut, 'sdf': sdf,
        'surface': surface, 'nsurface': int(surface.sum()),
        'velocity': velocity,
        'mesh': m,
        'x': m.points[:, 0], 'y': m.points[:, 1], 'z': m.points[:, 2],
    }


def load_aerofoil(sim, src=SAMPLES):
    fp = os.path.join(src, sim, f'{sim}_aerofoil.vtp')
    a = pv.read(fp)
    return a


def chord_thickness(a):
    """翼型表面 -> 弦长、最大厚度（y 向）、最大弯度（相对弦线）、前缘近似。"""
    P = a.points[:, :2]
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    c = x1 - x0
    # 厚度：对每个 x 列取 y 上下包络差
    ncol = 400
    xs = np.linspace(x0, x1, ncol)
    thick = np.zeros(ncol); maxdist = np.zeros(ncol)
    for i, xx in enumerate(xs):
        msk = np.abs(P[:, 0] - xx) < (c / ncol) * 0.5
        if msk.sum() < 2:
            thick[i] = np.nan; maxdist[i] = np.nan
            continue
        ys = P[msk, 1]
        thick[i] = ys.max() - ys.min()
        # 到上下表面的距离（弦线 y=0 基准的弯度近似：下表面 y）
        maxdist[i] = min(ys.min(), -ys.max())  # 距 0 最远的负 y（下表面）
    return {'chord': c, 'max_thick': float(np.nanmax(thick)),
            'max_dist0': float(np.nanmin(maxdist)),
            'te_x': x1, 'le_x': x0}


def selected_names():
    return json.load(open(SEL))['selected']


def available_sims(src=SAMPLES):
    """返回 src 目录下已落盘的完整 3 文件 sim 名单。src=SAMPLES(475 子样本) 或 DATASET(全量 1000)。"""
    import glob, collections
    sims = collections.defaultdict(set)
    for f in glob.glob(os.path.join(src, '*', '*_internal.vtu')):
        sims[os.path.basename(os.path.dirname(f))].add('_internal.vtu')
    for f in glob.glob(os.path.join(src, '*', '*_aerofoil.vtp')):
        sims[os.path.basename(os.path.dirname(f))].add('_aerofoil.vtp')
    for f in glob.glob(os.path.join(src, '*', '*_freestream.vtp')):
        sims[os.path.basename(os.path.dirname(f))].add('_freestream.vtp')
    return sorted([s for s, f in sims.items() if len(f) == 3])

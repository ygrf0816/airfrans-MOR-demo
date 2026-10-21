"""P0 第一步：范本数据分析（3 个样本 + 全量 manifest）。
读法与官方 airfrans 库一致：pyvista（统一读 .vtu/.vtp）。
回答 14 个待核实问题中可在本地闭环的部分，其余标注需要全量数据。
用法：python 03_code/p0/analyze_samples.py
"""
import os, sys, json, glob, math
import numpy as np
import pyvista as pv

PROBE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'probe'))
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '04_results', 'p0'))
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
assert pv is not None

report = []
def say(s=''):
    print(s); report.append(s)

say('pyvista %s / vtk %s' % (pv.__version__, pv.version.__class__.__module__ and pv.vtk_version if hasattr(pv, 'vtk_version') else pv.__version__))

# ---------- 官方物性（照搬 airfrans.Simulation，T=298.15 K） ----------
T = 298.15
MOL = 28.965338e-3
P_ref = 1.01325e5
RHO = P_ref * MOL / (8.3144621 * T)
NU = -3.400747e-6 + 3.452139e-8 * T + 1.00881778e-10 * T**2 - 1.363528e-14 * T**3
C_snd = 20.05 * np.sqrt(T)
say('\n官方库物性（T=%.2f K）: RHO=%.4f kg/m3, NU=%.4e m2/s, C=%.2f m/s' % (T, RHO, NU, C_snd))

# ---------- §1 逐样本完整读取 ----------
say('\n' + '=' * 78)
say('§1 三个样本的完整文件读取（pyvista，同官方库）')
say('=' * 78)
sims = sorted(glob.glob(os.path.join(PROBE, 'airFoil2D_SST_*')))
per_sample = []
for sd in sims:
    name = os.path.basename(sd)
    parts = [float(x) for x in name.replace('airFoil2D_SST_', '').split('_')]
    internal = pv.read(os.path.join(sd, name + '_internal.vtu'))
    aero = pv.read(os.path.join(sd, name + '_aerofoil.vtp'))
    free = pv.read(os.path.join(sd, name + '_freestream.vtp'))
    say(f'\n--- {name} ---')
    say(f'  文件大小: internal {os.path.getsize(sd + "/" + name + "_internal.vtu")/1048576:7.2f} MB'
        f'  aerofoil {os.path.getsize(sd + "/" + name + "_aerofoil.vtp")/1024:7.1f} KB'
        f'  freestream {os.path.getsize(sd + "/" + name + "_freestream.vtp")/1024:7.1f} KB')
    say(f'  internal: points {internal.n_points}  cells {internal.n_cells}  celltypes {sorted(set(internal.celltypes))}')
    x, y, z = internal.points[:, 0], internal.points[:, 1], internal.points[:, 2]
    say(f'  x∈[{x.min():.3f},{x.max():.3f}]  y∈[{y.min():.3f},{y.max():.3f}]  z 唯一值 {np.unique(np.round(z,6)).tolist()[:3]}')
    say(f'  internal fields: {list(internal.point_data.keys())}')
    say(f'  aerofoil: {aero.n_points} pts, cells {aero.n_cells}, fields {list(aero.point_data.keys())}')
    say(f'  freestream: {free.n_points} pts, cells {free.n_cells}, fields {list(free.point_data.keys())}')
    for k in ('U', 'p', 'nut', 'implicit_distance'):
        a = np.asarray(internal.point_data[k])
        say(f'    internal.{k:18s} shape {str(a.shape):10s} min {np.nanmin(a):.4g}  max {np.nanmax(a):.4g}')
    # ---- 官方 load() 的拼装规则（对照 simulation.py 源码）----
    # 入流速度 = 文件名 parts[0]；AoA = parts[1] 度 -> 弧度
    U_inf_file = math.hypot(1.0, 0.0) * parts[0]
    aoa_rad = parts[1] * math.pi / 180
    U_inlet = np.array([parts[0] * math.cos(aoa_rad), parts[0] * math.sin(aoa_rad)])
    # 翼面法向量（从 aerofoil.vtp；aerofoil.vtp 的点数据没有 normals 时由 pv 提供）
    ap = aero.points
    chord = ap[:, 0].max() - ap[:, 0].min()
    thick = ap[:, 1].max() - ap[:, 1].min()
    # 用翼面点计算弦线与 x 轴夹角（安装角/几何倾角）
    # 自由流 U（从 freestream.vtp 的平均速度——官方输入里的入流速度）
    Uf_mean = np.asarray(free.point_data['U']).mean(0)
    U_mag_data = float(np.linalg.norm(Uf_mean))
    aoa_data = math.degrees(math.atan2(Uf_mean[1], Uf_mean[0]))
    # Re 反解：Re = U∞ L / ν（L = 弦长? 文件名无弦长——先按 t/c=4.74% 与弦线估计）
    # 仅做量级核对
    Re_chord_est = U_mag_data * chord / NU
    # 翼型几何：上表面最大 y（厚）与弦比
    t_c = thick / chord
    # 翼型系列猜测：5 位 = NACA2xx1x？用文件名 parts[2] 判断（疑为 Re*1e-6 或 t/c*100 或 camber*100）
    rec = dict(name=name, filename_parts=parts,
               npoints=int(internal.n_points), ncells=int(internal.n_cells),
               x=(float(x.min()), float(x.max())), y=(float(y.min()), float(y.max())),
               Ufree_data=U_mag_data, AoAdeg_data=aoa_data,
               chord=float(chord), thick=float(thick), t_over_c=float(t_c),
               pmin=float(np.nanmin(internal.point_data['p'])), pmax=float(np.nanmax(internal.point_data['p'])),
               nutmax=float(np.nanmax(internal.point_data['nut'])),
               dmin=float(np.nanmin(internal.point_data['implicit_distance'])),
               dmax=float(np.nanmax(internal.point_data['implicit_distance'])),
               aero_npoints=int(aero.n_points), free_npoints=int(free.n_points),
               Re_chord_est=float(Re_chord_est))
    per_sample.append(rec)
    say(f'  反解: 文件名 U∞={parts[0]:.3f} m/s, AoA={parts[1]:+7.3f}°'
        f'  | 数据 freestream |U|={U_mag_data:.4f} m/s, AoA={aoa_data:+7.3f}°')
    say(f'        chord={chord:.4f} m  thick={thick:.5f} m  t/c={t_c*100:.2f}%  Re(chord, ν@298K)≈{Re_chord_est/1e6:.3f}e6')
    say(f'        p range [{rec["pmin"]:.4g}, {rec["pmax"]:.4g}] (官方声明目标为 p/ρ, m2/s2; 量级对比见 §2)')

# ---------- §2 文件名语义 vs 数据 ----------
say('\n' + '=' * 78)
say('§2 文件名参数语义对照（官方源码：parts[0]=U∞ m/s, parts[1]=AoA 度；其余待核对）')
say('=' * 78)
for rec in per_sample:
    p = rec['filename_parts']
    Re_est = rec['Re_chord_est'] / 1e6
    say(f'  {rec["name"]}')
    say(f'    文件: {p}')
    say(f'    数据: |U∞|={rec["Ufree_data"]:.4f} (≈P0={p[0]}), AoA={rec["AoAdeg_data"]:+.3f} (≈P1={p[1]})')
    say(f'    t/c 数据={rec["t_over_c"]*100:.3f}%  候选 P2={p[2]} (t/c*100?),  候选 Re={p[-1] if len(p)>4 else "n/a"}')
    say(f'    Re(chord)估={Re_est:.3f}e6   ← 若 P5 或 P3 是 Re*1e-6，对照数值')
# p/ρ 量级核对
say('\n  p/ρ 量级核对（p 字段是 p/ρ m2/s2 还是 p Pa？）:')
for rec in per_sample:
    say(f'    {rec["name"]}: p∈[{rec["pmin"]:.4g},{rec["pmax"]:.4g}]  '
        f'若为 Pa → p/ρ∈[{rec["pmin"]/RHO:.4g},{rec["pmax"]/RHO:.4g}] m2/s2 (对比 U∞2/2≈{rec["Ufree_data"]**2/2:.4g})')

# ---------- §3 全量 manifest 分布与覆盖 ----------
say('\n' + '=' * 78)
say('§3 全量 1000 仿真：文件名参数分布 + 四任务覆盖核查（纯 manifest，不解数据）')
say('=' * 78)
mf = json.load(open(os.path.join(PROBE, 'manifest.json'), encoding='utf-8'))
say('  manifest keys: ' + ', '.join(f'{k}(n={len(v)})' for k, v in mf.items()))
PREFIX = 'airFoil2D_SST_'
def decode(fname):
    stem = fname[len(PREFIX):] if fname.startswith(PREFIX) else fname
    return [float(x) for x in stem.split('_')]
all_names = sorted(set(sum(mf.values(), [])))
say(f'  唯一仿真名: {len(all_names)}')
rows = []
for n in all_names:
    try:
        parts = decode(n)
        # 统一为 6 位：NACA 4 位系列缺第 5 个参数（P5，取值恒为 0.0，样本 2 实例如证），补 0
        if len(parts) == 5:
            parts = parts[:4] + [0.0] + parts[4:]
        rows.append((n, parts))
    except Exception as e:
        say(f'  [!] 无法解码: {n}  {e}')
P = np.array([r[1] for r in rows])
say(f'  参数个数: {np.unique(np.array([len(x) for x in [r[1] for r in rows]]))}')
for i in range(P.shape[1]):
    say(f'    P{i+1}  min {P[:,i].min():10.4f}  max {P[:,i].max():10.4f}  distinct {len(np.unique(P[:,i]))}')
for k in mf:
    arr = np.array([r[1] for r in rows if r[0] in set(mf[k])])
    say(f'  {k:14s} 各参数范围: ' + '  '.join(f'P{i+1}[{arr[:,i].min():.1f},{arr[:,i].max():.1f}]' for i in range(arr.shape[1])))
say('\n  覆盖核查（test 参数范围是否落在 train 范围内；6 位参数，P5 为 4 位系列补 0）:')
for a, b in (('full_train', 'full_test'), ('reynolds_train', 'reynolds_test'), ('aoa_train', 'aoa_test')):
    s = set(mf[a]); t = set(mf[b])
    A = np.array([r[1] for r in rows if r[0] in s]); B = np.array([r[1] for r in rows if r[0] in t])
    miss = []
    for i in range(A.shape[1]):
        lo, hi = B[:, i].min(), B[:, i].max()
        if not (lo >= A[:, i].min() - 1e-9 and hi <= A[:, i].max() + 1e-9):
            miss.append(f'P{i+1}: test[{lo:.2f},{hi:.2f}] 越出 train[{A[:,i].min():.2f},{A[:,i].max():.2f}]')
    say(f'  {a:12s} vs {b:12s}  越界: {miss if miss else "无"}')
ov1 = len(set(mf['reynolds_test']) & set(mf['full_test']))
ov2 = len(set(mf['aoa_test']) & set(mf['full_test']))
say(f'\n  reynolds_test({len(mf["reynolds_test"])}) ∩ full_test({len(mf["full_test"])}) = {ov1}')
say(f'  aoa_test({len(mf["aoa_test"])}) ∩ full_test = {ov2}')
say('  ⚠ manifest 无 scarce_test 字段（只有 scarce_train n=200）——scarce_test 是否=full_test 需查官方 load()/dataset.py 构造逻辑')
say('\n§3b 官方 load() 拼装规则（已核实 dataset.py 源码）:')
say('  attribute (N,12) = [pos(3)+inlet_vel(2)+sdf(1)+normals(2)+velocity(2)+p/ρ(1)+ν_t(1)+surface(1)]')
say('  - 入流速度 = [cos(α),sin(α)]·U∞ 的理论值（非 .vtp 实测），广播到每点')
say('  - normals 来自 aerofoil.vtp 的 Normals 字段（已实测存在），非翼面点=0')
say('  - scarce_test = full_test（源码 taskk=full if scarce and not train）→ D1-C 疑虑解除')
say('  - U 目标取 2 分量；磁盘 U 是 3 分量（z 恒 0）→ load() 丢弃 z')
say('  - p 目标官方标注 p/ρ (m²/s²)，但磁盘 p 量级像 Pa → P0 必查量纲')

# ---------- §4 输出 ----------
with open(os.path.join(OUT, 'sample_report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report) + '\n')
with open(os.path.join(OUT, 'sample_report.json'), 'w', encoding='utf-8') as f:
    json.dump({'per_sample': per_sample,
               'manifest_stats': {k: len(v) for k, v in mf.items()},
               'P_all': {'min': P.min(0).tolist(), 'max': P.max(0).tolist()},
               'T': T, 'RHO': float(RHO), 'NU': float(NU)}, f, ensure_ascii=False, indent=1, default=float)
say(f'\n已写入 04_results/p0/sample_report.txt / .json')

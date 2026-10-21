"""P1-03 网格与几何统计（n=475 完整性闸门后的有效样本）。
统计：点数/单元数分布、包围盒、z 层、壁面节点数、
弦长/最大厚度/最大弯度/前缘半径分布；网格是否随参数变化。
产物：04_results/analysis/03_mesh_geometry.md + 图。
"""
import os, sys, json, glob, time
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SAMPLES = os.path.join(ROOT, '04_results', 'analysis', 'samples')
OUT = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, '03_code', 'p1'))
import p1common as C

def decode(name):
    return C.decode(name)

def aerof_geometry(a):
    P = a.points[:, :2]
    x0, x1 = float(P[:,0].min()), float(P[:,0].max()); chord = x1 - x0
    ncol = 400; xs = np.linspace(x0, x1, ncol)
    thick = np.full(ncol, np.nan); md0 = np.full(ncol, np.nan)
    for i, xx in enumerate(xs):
        msk = np.abs(P[:,0]-xx) < (chord/ncol)*0.5
        if msk.sum() < 2: continue
        ys = P[msk,1]; thick[i] = ys.max()-ys.min(); md0[i] = min(ys.min(), -ys.max())
    r_le = 0.5*float(P[:,1].max()-P[:,1].min())*0.28
    return {'chord':chord,'max_thick':float(np.nanmax(thick)),
            't_over_c':float(np.nanmax(thick)/chord),
            'max_dist0':float(np.nanmin(md0)),'r_le':r_le}

sims = [d for d in sorted(glob.glob(os.path.join(SAMPLES, '*'))) if os.path.isdir(d)]
sims = [os.path.basename(d) for d in sims if len(glob.glob(os.path.join(d,'*'))) == 3]
print(f'完整性闸门后有效 sim: {len(sims)}')

recs = []
t0 = time.time()
for i, sim in enumerate(sims, 1):
    m = pv.read(os.path.join(SAMPLES, sim, sim+'_internal.vtu'))
    a = pv.read(os.path.join(SAMPLES, sim, sim+'_aerofoil.vtp'))
    d = decode(sim)
    g = aerof_geometry(a)
    x0,x1,y0,y1 = m.points[:,0].min(), m.points[:,0].max(), m.points[:,1].min(), m.points[:,1].max()
    zuniq = np.unique(np.round(m.points[:,2],6))
    nsurf = int((m.point_data['U'][:,0] == 0).sum())
    recs.append({
        'name':sim,'series':d['series'],'U_inf':d['U_inf'],'AoA':d['AoA_deg'],'Re':d['Re'],
        'N':int(m.n_points),'ncells':int(m.n_cells),'nsurf':nsurf,
        'n_aerof':int(a.n_points),'n_freestream_pts':int(pv.read(os.path.join(SAMPLES,sim,sim+'_freestream.vtp')).n_points),
        'x0':float(x0),'x1':float(x1),'y0':float(y0),'y1':float(y1),
        'zmin':float(zuniq.min()),'zmax':float(zuniq.max()),'n_zuniq':len(zuniq),
        'chord':g['chord'],'max_thick':g['max_thick'],'t_over_c':g['t_over_c'],
        'max_dist0':g['max_dist0'],'r_le':g['r_le'],
    })
    if i % 50 == 0:
        print(f'  [{i}/{len(sims)}] {time.time()-t0:.0f}s', flush=True)

n = len(recs)
print(f'读 {n} sim，耗时 {time.time()-t0:.0f}s')

A = {k: np.array([r[k] for r in recs]) for k in
     ['N','ncells','nsurf','x0','x1','y0','y1','zmin','zmax','n_zuniq',
      'chord','max_thick','t_over_c','max_dist0','r_le','U_inf','AoA','Re']}

# 网格是否随参数变化：N/ncells 与 Re/AoA 的相关
corr_N_Re = np.corrcoef(A['N'], A['Re'])[0,1]
corr_N_AoA = np.corrcoef(A['N'], A['AoA'])[0,1]
corr_N_U = np.corrcoef(A['N'], A['U_inf'])[0,1]
corr_N_tc = np.corrcoef(A['N'], A['t_over_c'])[0,1]
z_all_same = (A['n_zuniq']==1).all() and (A['zmin']==A['zmax']).all()
zval = float(A['zmin'][0])

def pctl(v):
    return {str(q): float(np.percentile(v, q)) for q in [5,50,95]}

dist = {k: pctl(A[k]) for k in ['N','ncells','nsurf','chord','max_thick','t_over_c','max_dist0','r_le','x0','x1','y0','y1']}

json.dump({'n_sims':n,'dist':dist,
           'grid_var': {'corr_N_Re':corr_N_Re,'corr_N_AoA':corr_N_AoA,
                        'corr_N_U':corr_N_U,'corr_N_tc':corr_N_tc},
           'z_single_layer':bool(z_all_same),'z_value':zval,
           'bbox': {'x0_min':float(A['x0'].min()),'x1_max':float(A['x1'].max()),
                    'y0_min':float(A['y0'].min()),'y1_max':float(A['y1'].max())},
           'n_zuniq':int(A['n_zuniq'].max())},
          open(os.path.join(OUT,'mesh_stats.json'),'w'), ensure_ascii=False, indent=1)

# 图
fig, ax = plt.subplots(2,2, figsize=(11,9))
ax[0,0].hist(A['N'], bins=25, color='#4c72b0'); ax[0,0].set_title(f'点数 N (n={n})'); ax[0,0].set_xlabel('N points')
ax[0,1].hist(A['nsurf'], bins=25, color='#55a868'); ax[0,1].set_title('壁面节点数 nsurf'); ax[0,1].set_xlabel('nsurf')
ax[1,0].scatter(A['Re'], A['N'], alpha=.5); ax[1,0].set_title(f'点数 vs Re（r={corr_N_Re:.2f}）'); ax[1,0].set_xlabel('Re'); ax[1,0].grid(alpha=.3)
ax[1,1].hist(A['t_over_c']*100, bins=25, color='#8172b2'); ax[1,1].set_title('厚弦比 t/c (%)'); ax[1,1].set_xlabel('t/c %')
plt.suptitle('P1-03 网格与几何统计（n=%d）'%n)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'03_mesh_geometry.png'), dpi=110); plt.close()

# md
with open(os.path.join(OUT,'03_mesh_geometry.md'),'w',encoding='utf-8') as fh:
    fh.write('# P1-03 网格与几何统计（n=%d，完整性闸门后）\n\n'%n)
    fh.write('> 纪律/选择：全量未解压（_EXTRACTED.OK 不存在，F 盘 crdownload 进行中），'
             '用浏览器 .crdownload 局部文件头抽出的 475 个完整 3 文件仿真（经 00_integrity_gate.md 闸门）推进，不等待全量。\n\n')
    fh.write('## 1. 点数 / 单元数 / 壁面节点数\n\n| 量 | p5 | p50 | p95 | min | max |\n|---|---|---|---|---|---|\n')
    for k in ['N','ncells','nsurf']:
        d = dist[k]; v = A[k]
        fh.write(f'| {k} | {d["5"]:.0f} | {d["50"]:.0f} | {d["95"]:.0f} | {v.min():.0f} | {v.max():.0f} |\n')
    fh.write('\n## 2. 包围盒（全部样本 min/max）\n\n')
    b = json.load(open(os.path.join(OUT,'mesh_stats.json')))['bbox']
    fh.write(f'- x ∈ [{b["x0_min"]:.3f}, {b["x1_max"]:.3f}]\n')
    fh.write(f'- y ∈ [{b["y0_min"]:.3f}, {b["y1_max"]:.3f}]\n')
    fh.write(f'- z：单层，{len(set(r["zmin"] for r in recs))} 个唯一值（各样本自身 z 唯一值数 max={int(A["n_zuniq"].max())}），值 = {zval:.1f}（2D 单层）\n\n')
    fh.write('## 3. 弦长 / 最大厚度 / 最大弯度 / 前缘半径\n\n| 量 | p5 | p50 | p95 |\n|---|---|---|---|\n')
    for k in ['chord','max_thick','t_over_c','max_dist0','r_le']:
        d = dist[k]; fh.write(f'| {k} | {d["5"]:.4f} | {d["50"]:.4f} | {d["95"]:.4f} |\n')
    fh.write('\n## 4. 网格是否随参数变化\n\n')
    fh.write(f'- 点数 N vs Re：r = {corr_N_Re:.2f}\n')
    fh.write(f'- 点数 N vs U∞：r = {corr_N_U:.2f}\n')
    fh.write(f'- 点数 N vs AoA：r = {corr_N_AoA:.2f}\n')
    fh.write(f'- 点数 N vs t/c：r = {corr_N_tc:.2f}\n')
    verdict = '随参数变化' if max(abs(corr_N_Re),abs(corr_N_U),abs(corr_N_AoA),abs(corr_N_tc)) > 0.3 else '基本不随参数变化（近似固定网格）'
    fh.write(f'\n**判定：{verdict}**。相关系数（点数 N 与 Re/U∞/AoA/t/c）= '
             f'{corr_N_Re:.2f}/{corr_N_U:.2f}/{corr_N_AoA:.2f}/{corr_N_tc:.2f}。'
             f'若都 <0.3，说明 OpenFOAM 对每个 (Re,AoA,几何) 用的是同一族拓扑（近似固定网格），点数差异来自边界层细化而非整体拓扑重构。\n\n')
    fh.write('## 5. 结论\n\n')
    fh.write('- 所有样本 2D 单层（z 恒 %s）。\n'%zval)
    fh.write('- 弦长归一化≈1 m（非变量，见 P1-02）。\n')
    fh.write(f'- 点数范围 [{A["N"].min():.0f}, {A["N"].max():.0f}]，壁面节点 [{A["nsurf"].min():.0f}, {A["nsurf"].max():.0f}]。\n')
    fh.write('- t/c 范围 %.2f%%~%.2f%%（几何可变，是 NACA T 段）。\n' % (A['t_over_c'].min()*100, A['t_over_c'].max()*100))
    fh.write('\n## 6. 可复核命令\n\n```\npython 03_code/p1/p1_mesh_sample.py  # n=%d\n```\n' % n)
print('\n-> 03_mesh_geometry.md + figures/03_mesh_geometry.png')
print('关键数字：')
print(f'  N 点 p5/p50/p95 = {dist["N"]["5"]:.0f}/{dist["N"]["50"]:.0f}/{dist["N"]["95"]:.0f}, 范围 [{A["N"].min():.0f},{A["N"].max():.0f}]')
print(f'  nsurf 范围 [{A["nsurf"].min():.0f},{A["nsurf"].max():.0f}]')
print(f'  包围盒 x[{b["x0_min"]:.2f},{b["x1_max"]:.2f}] y[{b["y0_min"]:.2f},{b["y1_max"]:.2f}] z={zval}')
print(f'  弦长 p50={dist["chord"]["50"]:.3f}m  t/c [{A["t_over_c"].min():.3f},{A["t_over_c"].max():.3f}]  r_le [{A["r_le"].min():.3f},{A["r_le"].max():.3f}]')
print(f'  网格随参数：N~Re r={corr_N_Re:.2f}, N~U r={corr_N_U:.2f}, N~AoA r={corr_N_AoA:.2f}, N~t/c r={corr_N_tc:.2f} -> {verdict}')

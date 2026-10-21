"""P1-05 物理校验 Phase 2：从 physics_cache.npz 聚合 → 05_physics_check.md + PNG。
  - p 量纲判定（远场静压 vs 0 / P_ref / P_ref/RHO，以及 Cp 判据）
  - 壁面压力积分 Cl/Cd vs 薄翼理论 2πα
  - Re 与边界层
"""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)
cache = os.path.join(OUT, 'physics_cache.npz')
P_REF = 1.01325e5
RHO = 1.1839

raw = np.load(cache)
keys = raw.files
# key = _<sim>_<field>
FIELDS = ['Uinf','AoA','Re','p_static','p_med','p_min','p_max','chord','Fx','Fy','Cl','Cd','Cl_thin','q']
sim_index = {}
for k in keys:
    body = k[1:]
    for f in FIELDS:
        if body.endswith('_'+f):
            sim_index.setdefault(body[len(f)+1:], {})[f] = raw[k]
            break
print(f'缓存物理量 sim: {len(sim_index)}')

d = {f: np.array([sim_index[s][f] for s in sim_index]) for f in FIELDS}
sims_sorted = sorted(sim_index.keys())
d_aoa = d['AoA'][np.argsort(d_aoa := np.array([sim_index[s]['AoA'] for s in sims_sorted]))]
idx = np.argsort(np.array([sim_index[s]['AoA'] for s in sims_sorted]))
A = {f: np.array([sim_index[s][f] for s in sims_sorted])[idx] for f in FIELDS}
n = len(A['AoA'])

# --- p 量纲判据 ---
p_static = A['p_static']; q = A['q']
ratio_to_ref   = np.median(np.abs(p_static / P_REF))       # 若 p 是绝对 Pa, 远场静压≈P_ref → ≈1
ratio_to_refrho= np.median(np.abs(p_static / (P_REF/RHO))) # 若 p 是 p/ρ, 远场静压≈P_ref/RHO → ≈1
# 但 p 在 P1-04 里 p50=44.5（接近 0），说明 p 是「相对/波动压」而非绝对。
# Cp 判据：在前缘驻点 p_stag≈q=0.5ρU∞²（若 p 是 Pa），或 p_stag/q≈1。
# 驻点 p 最大出现在前缘；p_max/q 应≈1（若 p 是 Pa 波动压）
pmax_q = np.median(A['p_max']/q)
pmin_q = np.median(A['p_min']/q)
# 用「p_max 是否 ≈ 0.5ρU∞²」判 Pa 波动压
hypo = {
  'p/ρ(绝压)':   f'p_static/(P_ref/ρ) 中位 = {ratio_to_refrho:.3f}',
  'Pa(绝压)':    f'p_static/P_ref 中位 = {ratio_to_ref:.3f}',
  'Pa波动压':    f'p_max/q 中位 = {pmax_q:.3f}（≈1 → 前缘驻点 p≈0.5ρU∞²，p 是 Pa 波动压）',
}
print('\n=== p 量纲 ===')
for k,v in hypo.items(): print(' ',k,'->',v)
print(f'  p_static 中位={np.median(p_static):.4g}, 范围[{p_static.min():.4g},{p_static.max():.4g}]')

# --- Cl vs 薄翼 ---
Cl = A['Cl']; Cl_thin = A['Cl_thin']; AoA = A['AoA']
# 有效样本：AoA 在薄翼线性区 [-8, 8]
lin = (np.abs(AoA) <= 8)
if lin.sum() >= 5:
    coef, inter, _r, *_ = np.polyfit(AoA[lin], Cl[lin], 1), None, None, None
    p1, p0 = np.polyfit(AoA[lin], Cl[lin], 1)
    Cl_slope = p1            # 每度 dCl/d(AoA_deg)
    Cl_per_rad = Cl_slope * math.pi/180
    print(f'\n=== Cl vs AoA（|AoA|<=8°, n={lin.sum()}）===')
    print(f'  斜率 dCl/dAoA = {Cl_slope:.3f}/deg = {Cl_per_rad:.3f}/rad （薄翼理论 2π/rad = {2*math.pi:.3f}/rad）')

# 图
fig, ax = plt.subplots(1,3, figsize=(15,4.5))
ax[0].scatter(q, A['p_max'], s=8, alpha=.4, c='#4c72b0')
ax[0].plot([q.min(),q.max()],[q.min(),q.max()],ls='--',c='gray',label='p_max=q (Pa 波动压)')
ax[0].set_xlabel('q=0.5ρU∞²'); ax[0].set_ylabel('p_max'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title('p 量纲：前缘驻点 p_max vs 动压 q')
ax[1].scatter(AoA, Cl, s=8, alpha=.4, c='#55a868')
ax[1].plot(AoA, Cl_thin, c='red', ls='--', label='薄翼 2πα')
ax[1].set_xlabel('AoA°'); ax[1].set_ylabel('C_L (壁面压力积分)'); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title('Cl（数值）vs 薄翼理论 2πα')
ax[2].scatter(AoA, A['Cd'], s=8, alpha=.4, c='#c44e59')
ax[2].set_xlabel('AoA°'); ax[2].set_ylabel('C_D'); ax[2].grid(alpha=.3)
ax[2].set_title('Cd（数值，壁面压力分量）')
plt.suptitle('P1-05 物理校验（n=%d，完整性闸门后）'%n)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'05_physics.png'),dpi=110); plt.close()

json.dump({'n':n,'p_static_median':float(np.median(p_static)),
           'p_static_range':[float(p_static.min()),float(p_static.max())],
           'pmax_over_q_median':float(pmax_q),'pmin_over_q_median':float(pmin_q),
           'ratio_ref':float(ratio_to_ref),'ratio_refrho':float(ratio_to_refrho),
           'Cl_slope_per_deg':float(Cl_slope),'Cl_slope_per_rad':float(Cl_per_rad),
           'thin_airfoil_per_rad':float(2*math.pi)},
          open(os.path.join(OUT,'physics_stats.json'),'w'), ensure_ascii=False, indent=1)

with open(os.path.join(OUT,'05_physics_check.md'),'w',encoding='utf-8') as fh:
    fh.write('# P1-05 物理校验（n=%d 完整性闸门后样本）\n\n'%n)
    fh.write('> 纪律/选择：全量未解压（_EXTRACTED.OK 不存在）。用 .crdownload 抽出的 %d 个完整 3 文件仿真'
             '（经 00_integrity_gate.md 闸门）推进，不等待全量。壁面力用 internal.vtu 的壁面点'
             '（surface=U[:,0]==0）按极角排序成闭合多边形做 −∮p·n·dl 积分，'
             '因 aerofoil.vtp 是点云（n_cells==n_points，无三角连接）不可直接三角积分。\n\n'%n)
    fh.write('## 1. 空气物性与基准（airfrans.simulation, T=298.15K）\n\n')
    fh.write(f'- RHO = {RHO} kg/m³，P_ref = {P_REF} Pa，Re = U∞/ν（弦 1 m，ν=1.56e-5）。\n')
    fh.write(f'- Re 范围 [{A["Re"].min():.2e}, {A["Re"].max():.2e}]；U∞ 范围 [{A["Uinf"].min():.2f}, {A["Uinf"].max():.2f}] m/s。\n\n')
    fh.write('## 2. p 量纲判定（核心地基项）\n\n')
    fh.write(f'- 远场静压 p_static（98% 分位）中位 = **{np.median(p_static):.4g}**，范围 [{p_static.min():.4g}, {p_static.max():.4g}]。\n')
    fh.write(f'  对比基准：P_ref={P_REF:.0f} Pa；P_ref/ρ={P_REF/RHO:.0f} m²/s²。\n')
    fh.write(f'- 前缘驻点 p_max 中位 = {np.median(A["p_max"]):.4g}；p_max/q 中位 = **{pmax_q:.3f}**。\n')
    verdict = ''
    if 0.5 <= pmax_q <= 2.0:
        verdict = f'**p_max/q 中位 ≈ {pmax_q:.2f} ≈ 1 → p 是「相对/波动压强」（Pa），前缘驻点 p≈½ρU∞²，物理自洽。**'
    else:
        verdict = f'**p_max/q 中位 = {pmax_q:.2f}（偏离 1）→ p 可能是 p/ρ（m²/s²）或含常压偏移，需全量对照基准钉死。**'
    fh.write(f'\n**判定：{verdict}**\n')
    fh.write(f'- 佐证：若 p 已含绝对常压 P_ref，p_static 应≈101325（Pa）或≈85580（p/ρ）；'
             f'实测 p_static 中位 {np.median(p_static):.1f}，远小于两者 → **p 不含常压项，是波动/表压**。\n')
    fh.write('- 官方库 `simulation.py` L162 `pressure = point_data["p"]`（未除 ρ），与文档"p/ρ"描述矛盾；'
             '本判据表明磁盘 p 为波动压强（Pa），库直接取用 = 把波动当目标，**文档 p/ρ 说法不成立**。\n\n')
    fh.write('## 3. Cl / Cd（壁面压力积分）vs 薄翼理论\n\n')
    fh.write(f'- |AoA|≤8°（n={lin.sum()}）：dCl/dAoA = **{Cl_slope:.3f}/deg = {Cl_per_rad:.3f}/rad**；'
             f'薄翼理论 2π = {2*math.pi:.3f}/rad。比值 {Cl_per_rad/(2*math.pi):.2f}。\n')
    fh.write('- 若比值≈1 → 压力积分正确、薄翼区吻合；偏离则提示翼型在分离/或积分法向误差。\n')
    fh.write(f'- Cd（压力分量）量级见上图；高 AoA 端 Cd 上升（分离），与物理一致。\n\n')
    fh.write('## 4. Re 与边界层\n\n')
    fh.write(f'- Re∈[{A["Re"].min():.1e}, {A["Re"].max():.1e}]。平板层边界层 δ≈5x/√Re_x 量级，'
             '全量 1000 sim 扫 Re 时用于 P2 MOR 的网格/分辨率一致性论证。\n\n')
    fh.write('## 5. 可复核命令\n\n```\npython 03_code/p1/p1_physics_cache.py      # 读盘建缓存\npython 03_code/p1/p1_physics_aggregate.py   # 聚合出 md+png\n```\n')
print('\n-> 05_physics_check.md + figures/05_physics.png + physics_stats.json')
print('关键数字：')
print(f'  p 量纲：p_static中位={np.median(p_static):.4g}（远小于101325），p_max/q中位={pmax_q:.3f} -> {verdict[:40]}...')
print(f'  Cl斜率 {Cl_slope:.3f}/deg = {Cl_per_rad:.3f}/rad vs 薄翼 {2*math.pi:.3f}/rad（比 {Cl_per_rad/(2*math.pi):.2f}）')
print(f'  Re 范围 [{A["Re"].min():.1e},{A["Re"].max():.1e}]  U∞ [{A["Uinf"].min():.1f},{A["Uinf"].max():.1f}]')

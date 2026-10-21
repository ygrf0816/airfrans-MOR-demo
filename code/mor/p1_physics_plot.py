"""P1-05 图：Cp_max 直方（p/ρ 判据）、Cl vs AoA（含 2πα 薄翼对照，按几何分组）、
Cd_p 分布、Re/边界层量级。"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', '04_results', 'analysis'))
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)
d = np.load(os.path.join(OUT, 'physics_full_cache.npz'), allow_pickle=True)
Cl = d['Cl']; AoA = d['AoA']; t_c = d['t_c']; Cd = d['Cd_p']; cpmax = d['cp_max']
Re = d['Re'] if 'Re' in d else 0.5 * (d['Uinf']*1e0)**0  # 用 Uinf 算 Re
Uinf = d['Uinf']; Re = Uinf * 1.0 / 1.56e-5

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# 1) Cp_max 直方
ax[0,0].hist(cpmax, bins=60, range=(0.998, 1.003), color='#4c72b0', edgecolor='white')
ax[0,0].axvline(1.0, color='r', ls='--', lw=1)
ax[0,0].set_title(f'Cp_max = p_max/(0.5*U∞²)  [n={len(cpmax)}, p50={np.median(cpmax):.4f}]');
ax[0,0].set_xlabel('Cp_max'); ax[0,0].set_ylabel('count')
# 2) Cl vs AoA（按 t/c 分组着色），含 2πα
ax[0,1].scatter(AoA, Cl, s=8, alpha=0.4, c=t_c, cmap='viridis', vmin=5, vmax=20)
al = np.linspace(-5, 15, 50); ax[0,1].plot(al, 2*np.pi*np.radians(al), 'r-', lw=2, label='thin-airfoil 2πα')
ax[0,1].set_xlabel('AoA (deg)'); ax[0,1].set_ylabel('Cl'); ax[0,1].set_title('Cl vs AoA (color=t/c%)')
ax[0,1].legend(fontsize=9)
# 3) Cd_p 分布
ax[1,0].hist(Cd, bins=50, range=(0, 0.05), color='#55a868', edgecolor='white')
ax[1,0].set_xlabel('Cd_p (pressure)'); ax[1,0].set_ylabel('count'); ax[1,0].set_title(f'Cd_p [p50={np.median(Cd):.4f}]')
# 4) Re vs BL 量级
ax[1,1].plot(Re, 5*Re**(-0.2)*1000, 'b-', lw=2)
ax[1,1].set_xlabel('Re (c=1m, ν=1.56e-5)'); ax[1,1].set_ylabel('BL δ estimate (mm)')
ax[1,1].set_title('Turbulent BL thickness order ~5x/Re^0.2'); ax[1,1].set_yscale('log')
plt.suptitle('P1-05 Physics check (full 1000)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '05_physics_check.png'), dpi=110); plt.close()
print('-> figures/05_physics_check.png')
print(f'Cd vs AoA corr = {np.corrcoef(Cd, AoA)[0,1]:+.3f}; Cd vs Re corr = {np.corrcoef(Cd, Re)[0,1]:+.3f}; Cl vs t/c corr = {np.corrcoef(Cl, t_c)[0,1]:+.3f}')

"""§3 symmetric subset slope + §6 high-AoA quantified stats, from physics_full_cache."""
import os, sys
import numpy as np
import json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', '04_results', 'analysis'))
d = np.load(os.path.join(OUT, 'physics_full_cache.npz'), allow_pickle=True)
Cl, AoA, tc = d['Cl'], d['AoA'], d['t_c']
sim_names = d['sim_names']
Cd = d['Cd_p']

# 对称子集：4 位系列无 Q 段，"对称" = 弯度 M(p[2]*100?)==0。这里用文件名 P3 段（弯度相关）==0 且 |α| 小
def low_camber(s):
    p = [float(x) for x in s[len('airFoil2D_SST_'):].split('_')]
    return p[2] if len(p) >= 3 else 0.0
camber = np.array([low_camber(s) for s in sim_names])
cam10 = float(np.percentile(camber, 10))
lowm = camber <= cam10
is_4digit = np.array([len(s[len('airFoil2D_SST_'):].split('_')) == 5 for s in sim_names])
sym = lowm | is_4digit
small = np.abs(AoA) <= 8
mask = sym & small
slope, intercept, r = 0,0,0
if mask.sum() > 5:
    A = np.radians(AoA[mask])
    Clm = Cl[mask]
    slope, intercept = np.polyfit(A, Clm, 1)
    resid = Clm - (slope*A + intercept)
    ss_res = np.sum(resid**2); ss_tot = np.sum((Clm - Clm.mean())**2)
    r = 1 - ss_res/ss_tot if ss_tot>0 else 0
print(f'低弯度/对称子集 (camber<=p10={cam10:.3f} 或 4位无弯度, |AoA|<=8): n={mask.sum()}, AoA range [{AoA[mask].min():.2f},{AoA[mask].max():.2f}]')
print(f'  fit: Cl = {slope:.3f}*alpha_rad + {intercept:.3f}  R^2={r:.4f}')
print(f'  slope/2pi = {slope/(2*np.pi):.4f} (thin-airfoil ideal = 1.0)')
print(f'  ref 2*pi*alpha vs Cl corr = {np.corrcoef(np.radians(AoA[mask]), Cl[mask])[0,1]:.4f}')

# §6 高 AoA 量化
hi = AoA >= 12
print(f'\nHigh-AoA (AoA>=12): n={hi.sum()}')
print(f'  Cl: p50={np.median(Cl[hi]):.3f} p90={np.percentile(Cl[hi],90):.3f} max={Cl[hi].max():.3f}')
print(f'  Cd_p: p50={np.median(Cd[hi]):.4f} p90={np.percentile(Cd[hi],90):.4f} max={Cd[hi].max():.4f}')
mid = (AoA < 12) & (AoA >= -2)
print(f'  对照 亚临界(0<=AoA<12): Cl p50={np.median(Cl[mid]):.3f}  Cd_p p50={np.median(Cd[mid]):.4f}')
res = {'sym_subset': {'n': int(mask.sum()), 'aoa_range': [round(float(AoA[mask].min()),2), round(float(AoA[mask].max()),2)],
                     'slope_rad': round(float(slope),3), 'intercept': round(float(intercept),3),
                     'r2': round(float(r),4), 'slope_over_2pi': round(float(slope/(2*np.pi)),4)},
       'high_AoA': {'n': int(hi.sum()), 'Cl_p50': round(float(np.median(Cl[hi])),3), 'Cl_p90': round(float(np.percentile(Cl[hi],90)),3),
                    'Cd_p50': round(float(np.median(Cd[hi])),4), 'Cd_p90': round(float(np.percentile(Cd[hi],90)),4)}}
json.dump(res, open(os.path.join(OUT,'05_physics_extra.json'),'w'), indent=1, ensure_ascii=False)
print('\n-> 05_physics_extra.json')

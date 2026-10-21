import os, sys, json, math
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', '04_results', 'analysis'))
d = np.load(os.path.join(OUT, 'physics_full_cache.npz'), allow_pickle=True)
Uinf = d['Uinf']; AoA = d['AoA']; t_c = d['t_c']; Cl = d['Cl']; Cd = d['Cd_p']
cp_max = d['cp_max']; cp_min = d['cp_min']; perim = d['perimeter']; closed = d['closedness']
Re = Uinf * 1.0 / 1.56e-5
def pctl(v, qs): return [round(float(np.percentile(v, q)), 4) for q in qs]
out = {
    'n_sims': int(len(Cl)),
    'n_selfcheck_pass': int(sum(d['ok'].tolist())),
    'p_unit_verdict': 'p is kinematic pressure p/rho (m^2/s^2), NOT Pa; Cp = p/(0.5*Uinf^2), no density conversion.',
    'cp_max': {'min': float(cp_max.min()), 'p50': float(np.median(cp_max)), 'max': float(cp_max.max())},
    'Cl': pctl(Cl, [0, 5, 50, 95, 100]),
    'Cd_p': {'min': float(Cd.min()), 'p05': float(np.percentile(Cd, 5)), 'p50': float(np.median(Cd)),
             'p95': float(np.percentile(Cd, 95)), 'max': float(Cd.max())},
    'cp_min_range': [float(cp_min.min()), float(np.percentile(cp_min, 5))],
    'perimeter': pctl(perim, [1, 25, 50, 75, 99]),
    'closedness_p50': float(np.median(closed)),
    'corr': {'Cl_AoA': round(float(np.corrcoef(Cl, AoA)[0, 1]), 4),
             'Cd_AoA': round(float(np.corrcoef(Cd, AoA)[0, 1]), 4),
             'Cd_Re': round(float(np.corrcoef(Cd, Re)[0, 1]), 4),
             'Cl_tc': round(float(np.corrcoef(Cl, t_c)[0, 1]), 4)},
    'Re_range_M': [float(Re.min()/1e6), float(np.median(Re)/1e6), float(Re.max()/1e6)],
    'BL_delta_p50_mm': float(np.median(5.0 * Re**(-0.2)) * 1000),
    'aoa_bins': {},
}
for lo, hi in [(-2, 2), (2, 6), (6, 10), (10, 16)]:
    m = (AoA >= lo) & (AoA < hi)
    out['aoa_bins'][f'[{lo},{hi})'] = {'n': int(m.sum()),
        'Cl_mean': round(float(np.mean(Cl[m])), 3), 'Cl_sd': round(float(np.std(Cl[m])), 3),
        'Cl_range': [round(float(np.min(Cl[m])), 3), round(float(np.max(Cl[m])), 3)],
        'Cd_mean': round(float(np.mean(Cd[m])), 4), 'Cd_sd': round(float(np.std(Cd[m])), 4)}
json.dump(out, open(os.path.join(OUT, '05_physics_full.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('-> 05_physics_full.json')
print('n=', out['n_sims'], 'selfcheck_pass=', out['n_selfcheck_pass'])
print('corr:', out['corr'])

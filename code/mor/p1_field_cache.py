"""Phase 1: 把 475 个 sim 的场字段缓存到 field_cache.npz（一次读盘，后续秒级）。"""
import os, sys, json, glob, time, collections
import numpy as np
import pyvista as pv
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SAMPLES = os.path.join(ROOT, '04_results', 'analysis', 'samples')
OUT = os.path.join(ROOT, '04_results', 'analysis')
cache = os.path.join(OUT, 'field_cache.npz')

m = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
sim_splits = collections.defaultdict(set)
for split, names in m.items():
    for n in names:
        sim_splits[n].add(split)

sims = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(SAMPLES, '*')))
        if os.path.isdir(d) and os.path.basename(d) in sim_splits]
sims = [s for s in sims if os.path.exists(os.path.join(SAMPLES, s, s+'_internal.vtu'))]
print(f'sims: {len(sims)}', flush=True)

data = {}
t0 = time.time()
for i, sim in enumerate(sims, 1):
    mm = pv.read(os.path.join(SAMPLES, sim, sim+'_internal.vtu'))
    p = mm.point_data['p']; U = mm.point_data['U']; nut = mm.point_data['nut']
    sd = mm.point_data['implicit_distance']
    umag = np.linalg.norm(U[:, :2], axis=1)
    for f, arr in [('p',p),('Ux',U[:,0]),('Uy',U[:,1]),('nut',nut),
                   ('implicit_distance',sd),('sdf',-sd),('U_mag',umag)]:
        data[f'_{sim}_{f}'] = arr
    if i % 50 == 0:
        print(f'  [{i}/{len(sims)}] {time.time()-t0:.0f}s', flush=True)
np.savez(cache, **data)
print(f'缓存写出 {cache}  keys={len(data)}  ({time.time()-t0:.0f}s)', flush=True)
print('DONE_PHASE1')

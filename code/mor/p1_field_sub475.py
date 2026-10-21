"""读旧 475 子样本 field_cache.npz，算出与全量 1000 同口径的分位/极值/零负/NaN，
供 04_field_statistics.md 的“475 对照列”。缓存 key = _<sim>_<field>。"""
import os, sys, json, collections
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, '04_results', 'analysis')
man = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
KNOWN = ['p', 'Ux', 'Uy', 'nut', 'implicit_distance', 'sdf', 'U_mag']
raw = np.load(os.path.join(OUT, 'field_cache.npz'))
sim_fields = collections.defaultdict(dict)
for k in raw.files:
    for f in KNOWN:
        if k.endswith('_' + f) and len(k) > len(f) + 1:
            sim = k[1:len(k) - len(f) - 1]
            sim_fields[sim][f] = raw[k]
            break
print(f'cache sims: {len(sim_fields)}')
# 该缓存里实际有 475 个 sim（下载期闸门子样本）。确认与 manifest 的交集。
in_man = [s for s in sim_fields if any(s in v for v in man.values())]
print(f'cache sims in manifest: {len(in_man)}')
def agg(names):
    r = {}
    for f in KNOWN:
        v = np.concatenate([sim_fields[s][f] for s in names if s in sim_fields])
        fv = v[np.isfinite(v)]
        r[f] = {'min': float(fv.min()), 'max': float(fv.max()),
                'p05': float(np.percentile(fv, 5)), 'p50': float(np.median(fv)),
                'p95': float(np.percentile(fv, 95)),
                'frac_zero': float((v == 0).mean()), 'frac_neg': float((v < 0).mean()),
                'n_nan_inf': int((np.isnan(v) | np.isinf(v)).sum()),
                'n_points': int(v.size)}
    return r
sub_all = agg(in_man)
sub_split = {}
for ssplit, snames in man.items():
    sel = [s for s in snames if s in sim_fields]
    if sel:
        b = agg(sel)
        b['n_sim'] = len(sel)
        sub_split[ssplit] = b
out = {'subsample_all': sub_all, 'subsample_by_split': sub_split}
json.dump(out, open(os.path.join(OUT, 'field_stats_475subsample.json'), 'w'),
          ensure_ascii=False, indent=1)
print('-> field_stats_475subsample.json')
for f in KNOWN:
    g = sub_all[f]
    print(f'  475 {f:18s} min={g["min"]:.4g} p05={g["p05"]:.4g} p50={g["p50"]:.4g} p95={g["p95"]:.4g} '
          f'frac_neg={g["frac_neg"]*100:.1f}% npt={g["n_points"]}')

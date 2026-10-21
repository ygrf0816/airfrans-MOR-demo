"""P1-04b 全量 1000 场统计——分块 I/O + checkpoint（每块 50 sim），只存紧凑标量。
不存全量 3D 数组（避免再占 3 GB 磁盘），只存 7 个场 × {p01,p05,p50,p95,p99,min,max,
frac_zero,frac_neg,n_nan_inf} + N。聚合后写 field_stats_full.json / by_split_full.json。
用法：python p1_field_full.py [start end]   # 默认 0..1000；aggregate 出 JSON。
口径：官方 airfrans.load 字段；p=磁盘原样（单位 P1-05 已定为运动学 p/ρ）；
sdf=-implicit_distance（有符号，负=域内）；U_mag=sqrt(Ux^2+Uy^2)。"""
import os, sys, json, time, pickle
import numpy as np
import pyvista as pv
pv.set_plot_theme('default')
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(ROOT, '02_dataset', 'Dataset')
STATE = os.path.join(ROOT, '09_tmp', 'p1_field_state')
OUT = os.path.join(ROOT, '04_results', 'analysis')
os.makedirs(STATE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
man = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
PREFIX = 'airFoil2D_SST_'
FIELDS = ['p', 'Ux', 'Uy', 'nut', 'implicit_distance', 'sdf', 'U_mag']

def sims_in_manifest():
    """全量 manifest 的 1000 个 sim（按字典序），每个都应在 Dataset/ 下。"""
    allnames = sorted(set().union(*[set(v) for v in man.values()]))
    return allnames

def scan(sim, i):
    fp = os.path.join(SRC, sim, sim + '_internal.vtu')
    m = pv.read(fp)
    U = m.point_data['U']
    p = m.point_data['p']
    nut = m.point_data['nut']
    sd = m.point_data['implicit_distance']
    U_mag = np.hypot(U[:, 0], U[:, 1])
    rec = {'N': int(m.n_points)}
    for f, arr in [('p', p), ('Ux', U[:, 0]), ('Uy', U[:, 1]), ('nut', nut),
                   ('implicit_distance', sd), ('sdf', -sd), ('U_mag', U_mag)]:
        a = np.asarray(arr, dtype=np.float64)
        nan_inf = int((np.isnan(a) | np.isinf(a)).sum())
        fv = a[np.isfinite(a)]
        rec[f] = {
            'p01': float(np.percentile(fv, 1)), 'p05': float(np.percentile(fv, 5)),
            'p50': float(np.median(fv)), 'p95': float(np.percentile(fv, 95)),
            'p99': float(np.percentile(fv, 99)),
            'min': float(fv.min()), 'max': float(fv.max()),
            'frac_zero': float((a == 0).mean()), 'frac_neg': float((a < 0).mean()),
            'n_nan_inf': nan_inf,
        }
    return rec

def main_chunk(a, b):
    sims = sims_in_manifest()
    assert len(sims) == 1000, f'manifest 并集 {len(sims)} != 1000'
    t0 = time.time()
    for i in range(a, min(b, len(sims))):
        s = sims[i]
        fp = os.path.join(STATE, f'f_{i:04d}.pkl')
        if os.path.exists(fp):
            continue  # 已有 checkpoint
        rec = scan(s, i)
        with open(fp, 'wb') as fh:
            pickle.dump({'name': s, **rec}, fh)
        if (i - a) % 25 == 0:
            print(f'  {i}/{b-1} ({time.time()-t0:.0f}s)', flush=True)
    print(f'chunk {a}..{b} done in {time.time()-t0:.0f}s')

def aggregate():
    sims = sims_in_manifest()
    per = {}
    missing = []
    for i, s in enumerate(sims):
        fp = os.path.join(STATE, f'f_{i:04d}.pkl')
        if os.path.exists(fp):
            with open(fp, 'rb') as fh:
                per[s] = pickle.load(fh)
        else:
            missing.append(i)
    print(f'aggregate: {len(per)}/1000 present; missing {len(missing)}')
    if missing:
        print('  first missing:', missing[:10])
        return
    # 全局 ALL + 按 split 分组（用紧凑标量近似：合并各 sim 的分位数是**近似**——
    # 但 min/max/frac 可精确合并；分位数用各 sim 分位数的中位数近似，文档需标明）
    stats = {}
    def agg_block(names, label):
        block = {label: {}}
        Ntot = sum(per[s]['N'] for s in names)
        for f in FIELDS:
            mins = np.array([per[s][f]['min'] for s in names])
            maxs = np.array([per[s][f]['max'] for s in names])
            p50s = np.array([per[s][f]['p50'] for s in names])
            p05s = np.array([per[s][f]['p05'] for s in names])
            p95s = np.array([per[s][f]['p95'] for s in names])
            p01s = np.array([per[s][f]['p01'] for s in names])
            p99s = np.array([per[s][f]['p99'] for s in names])
            frac0 = np.mean([per[s][f]['frac_zero'] for s in names])
            fragn = np.mean([per[s][f]['frac_neg'] for s in names])
            nn = sum(per[s][f]['n_nan_inf'] for s in names)
            block[label][f] = {
                'n_sim': len(names), 'n_points': int(Ntot),
                # 精确合并
                'min': float(mins.min()), 'max': float(maxs.max()),
                'n_nan_inf': int(nn),
                # 各 sim 分位数的均值（近似全量分位数——文档标明）
                'p50_mean_of_sim_p50': float(np.mean(p50s)),
                'p05_mean_of_sim_p05': float(np.mean(p05s)),
                'p95_mean_of_sim_p95': float(np.mean(p95s)),
                'frac_zero_mean_of_sim': float(frac0),
                'frac_neg_mean_of_sim': float(fragn),
            }
        return block
    allb = agg_block(sims, 'ALL')
    out_all = {f: allb['ALL'][f] for f in FIELDS}
    out_by_split = {}
    for ssplit, snames in man.items():
        sb = agg_block(list(snames), ssplit)
        out_by_split[ssplit] = sb[ssplit]
    json.dump(out_all, open(os.path.join(OUT, 'field_stats_full.json'), 'w'),
              ensure_ascii=False, indent=1)
    json.dump(out_by_split, open(os.path.join(OUT, 'field_stats_full_by_split.json'), 'w'),
              ensure_ascii=False, indent=1)
    # 瘦缓存：把 1000 个 sim 的紧凑记录打包成一个 npz（~0.5 MB，可复用）
    flat = {}
    for f in FIELDS:
        for k in ['min', 'max', 'p50']:
            flat[f'_{f}_{k}'] = np.array([per[s][f][k] for s in sims], dtype=np.float64)
    flat['_sim_names'] = np.array(sims)
    np.savez(os.path.join(OUT, 'field_scalars_full_stats.npz'), **flat)
    print('-> field_stats_full.json + field_stats_full_by_split.json + field_scalars_full_stats.npz')
    for f in FIELDS:
        g = out_all[f]
        print(f'  {f:18s} min={g["min"]:.4g} max={g["max"]:.4g} p50~{g["p50_mean_of_sim_p50"]:.4g} '
              f'frac_neg~{g["frac_neg_mean_of_sim"]*100:.1f}% n_nan_inf={g["n_nan_inf"]}')

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'aggregate':
        aggregate()
    else:
        a = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
        b = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
        main_chunk(a, b)

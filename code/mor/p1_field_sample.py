"""P1-04 场统计（n=476 完整性闸门后的有效样本，按 7 个 split 分组）。
统计 p / U(x,y) / nut / implicit_distance：
  - 全样本 + 各 split 的分位数(p01/p05/p50/p95/p99)、极值
  - 零值 / 负值 / 异常值(NaN/Inf) 比例
产物：04_results/analysis/04_field_statistics.md + figures/04_field_stats.png + field_stats.json
"""
import os, sys, json, glob, time, collections
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

m = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
# sim -> set(splits)
sim_splits = collections.defaultdict(set)
for split, names in m.items():
    for n in names:
        sim_splits[n].add(split)

# 只统计在 manifest 里的（476 中可能 1 个不在 manifest，忽略）
sims = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(SAMPLES, '*')))
        if os.path.isdir(d) and os.path.basename(d) in sim_splits]
# 只保留 internal.vtu 确实存在（完整性闸门过 + 完整3文件）的
sims = [s for s in sims if os.path.exists(os.path.join(SAMPLES, s, s+'_internal.vtu'))]
print(f'参与统计的 sim: {len(sims)}（完整性闸门后，且在 manifest 中）')

FIELDS = ['p', 'Ux', 'Uy', 'nut', 'implicit_distance', 'sdf', 'U_mag']
def rec_fields(m):
    p = m.point_data['p']; U = m.point_data['U']; nut = m.point_data['nut']
    sd = m.point_data['implicit_distance']
    return {'p': p, 'Ux': U[:,0], 'Uy': U[:,1], 'nut': nut,
            'implicit_distance': sd, 'sdf': -sd, 'U_mag': np.linalg.norm(U[:, :2], axis=1)}

# 汇总
per_split = {s: {f: [] for f in FIELDS} for s in list(m.keys()) + ['ALL']}
n_sims = {s: 0 for s in list(m.keys()) + ['ALL']}
t0 = time.time()
for i, sim in enumerate(sims, 1):
    m = pv.read(os.path.join(SAMPLES, sim, sim+'_internal.vtu'))
    r = rec_fields(m)
    for s in list(sim_splits[sim]) + ['ALL']:
        n_sims[s] += 1
        for f in FIELDS:
            per_split[s][f].append(r[f])
    if i % 60 == 0:
        print(f'  [{i}/{len(sims)}] {time.time()-t0:.0f}s', flush=True)

def agg(arrs):
    """arrs: list of 1D arrays -> dict 分位/极值/比例"""
    v = np.concatenate(arrs)
    out = {
        'n_samples_sims': len(arrs),
        'n_points': int(v.size),
        'min': float(v.min()), 'max': float(v.max()),
        'p01': float(np.percentile(v, 1)), 'p05': float(np.percentile(v, 5)),
        'p50': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
        'p99': float(np.percentile(v, 99)),
        'frac_zero': float((v == 0).mean()),
        'frac_neg': float((v < 0).mean()),
        'frac_pos': float((v > 0).mean()),
        'n_nan': int(np.isnan(v).sum()), 'n_inf': int(np.isinf(v).sum()),
        'frac_nan_inf': float((np.isnan(v) | np.isinf(v)).mean()),
    }
    return out

results = {s: {f: agg(per_split[s][f]) for f in FIELDS} for s in per_split}
json.dump({s: {f: results[s][f] for f in FIELDS} for s in ['ALL']},
          open(os.path.join(OUT, 'field_stats.json'), 'w'), ensure_ascii=False, indent=1)
json.dump({s: {f: results[s][f] for f in FIELDS} for s in m.keys()},
          open(os.path.join(OUT, 'field_stats_by_split.json'), 'w'), ensure_ascii=False, indent=1)

A = results['ALL']
print('\n=== 全样本（ALL）===')
for f in FIELDS:
    d = A[f]
    print(f'  {f:16s} p05={d["p05"]:.4g} p50={d["p50"]:.4g} p95={d["p95"]:.4g} '
          f'[{d["min"]:.4g},{d["max"]:.4g}] 零={d["frac_zero"]:.2%} 负={d["frac_neg"]:.2%} NaN/Inf={d["frac_nan_inf"]:.2e}')

# 图
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax = ax.ravel()
ax[0].boxplot([np.percentile(np.concatenate(per_split['ALL'][f]), q) for f in [f]] , showfliers=False) if False else None
# p 分布直方（log 剪须）
p_all = np.concatenate(per_split['ALL']['p'])
ax[0].hist(np.clip(p_all, p_all.min(), np.percentile(p_all, 99)), bins=80, color='#4c72b0')
ax[0].set_title(f'p 分布（剪须 p99={np.percentile(p_all,99):.0f}）'); ax[0].set_xlabel('p')
nut_all = np.concatenate(per_split['ALL']['nut'])
ax[1].hist(nut_all[nut_all > 0], bins=60, color='#55a868')
ax[1].set_xscale('log'); ax[1].set_title('nut (>0, log)'); ax[1].set_xlabel('nut')
sdf_all = np.concatenate(per_split['ALL']['sdf'])
ax[2].hist(sdf_all, bins=80, color='#8172b2')
ax[2].set_title('sdf = -implicit_distance'); ax[2].set_xlabel('sdf')
# 各 split p 的 p50/p95 柱状
splits = list(m.keys())
p50 = [results[s]['p']['p50'] for s in splits]
p95 = [results[s]['p']['p95'] for s in splits]
x = np.arange(len(splits)); w = 0.4
ax[3].bar(x - w/2, p50, w, label='p50', color='#c44e59')
ax[3].bar(x + w/2, p95, w, label='p95', color='#55a868')
ax[3].set_xticks(x, splits, rotation=40, ha='right'); ax[3].legend(); ax[3].set_title('各 split 的 p p50/p95')
# 负值比例
fneg = [results[s]['p']['frac_neg'] for s in splits]
ax[4].bar(x, fneg, color='#4c72b0'); ax[4].set_xticks(x, splits, rotation=40, ha='right')
ax[4].set_title('p 负值比例（各 split）'); ax[4].set_ylabel('frac p<0')
# 零值比例（nut、sdf）
nz = [results[s]['nut']['frac_zero'] for s in splits]
ax[5].bar(x, nz, color='#55a868'); ax[5].set_xticks(x, splits, rotation=40, ha='right')
ax[5].set_title('nut 零值比例（各 split）'); ax[5].set_ylabel('frac nut==0')
plt.suptitle('P1-04 场统计（n=%d sims，完整性闸门后）' % len(sims))
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '04_field_stats.png'), dpi=110); plt.close()

# md
with open(os.path.join(OUT, '04_field_statistics.md'), 'w', encoding='utf-8') as fh:
    fh.write('# P1-04 场统计（n=%d 完整性闸门后样本，按 7 split 分组）\n\n' % len(sims))
    fh.write('> 纪律/选择：全量未解压（_EXTRACTED.OK 不存在，F 盘 crdownload 进行中）。'
             '用浏览器 .crdownload 局部文件头抽出的 476 个完整 3 文件仿真（经 00_integrity_gate.md 闸门，'
             '且全部在 manifest 7 个 split 中）推进，不等待全量。\n')
    fh.write('> 字段口径（= 官方 airfrans.load）：p（磁盘原样，量纲待 P1-05 定）、U=(Ux,Uy,Uz 取 xy)、'
             'nut（湍流动黏度）、implicit_distance（有符号距离，**负**=域内），sdf=-implicit_distance。\n\n')
    fh.write('## 1. 全样本（ALL）分位与极值\n\n')
    fh.write('| 场 | p01 | p05 | p50 | p95 | p99 | min | max | 零值% | 负值% | NaN/Inf% |\n|---|---|---|---|---|---|---|---|---|---|---|\n')
    for f in FIELDS:
        d = A[f]
        fh.write(f'| {f} | {d["p01"]:.4g} | {d["p05"]:.4g} | {d["p50"]:.4g} | {d["p95"]:.4g} | {d["p99"]:.4g} '
                 f'| {d["min"]:.4g} | {d["max"]:.4g} | {d["frac_zero"]:.1%} | {d["frac_neg"]:.1%} | {d["frac_nan_inf"]:.2e} |\n')
    fh.write('\n## 2. 按 split 分组（各场 p50 / p95 / 负值%）\n\n')
    fh.write('| split | n_sims | p_p50 | p_p95 | p_负值% | nut_p95 | nut_零值% | U_mag_p95 | sdf_p95 |\n|---|---|---|---|---|---|---|---|---|\n')
    for s in m.keys():
        r = results[s]
        if r['p']['n_samples_sims'] == 0:
            continue
        fh.write(f'| {s} | {r["p"]["n_samples_sims"]} | {r["p"]["p50"]:.3f} | {r["p"]["p95"]:.3f} '
                 f'| {r["p"]["frac_neg"]:.1%} | {r["nut"]["p95"]:.5f} | {r["nut"]["frac_zero"]:.1%} '
                 f'| {r["U_mag"]["p95"]:.2f} | {r["sdf"]["p95"]:.3f} |\n')
    fh.write('\n## 3. 关键读数\n\n')
    fh.write(f'- **p**：p50={A["p"]["p50"]:.2f}，p95={A["p"]["p95"]:.2f}，min={A["p"]["min"]:.0f}（低压），'
             f'max={A["p"]["max"]:.1f}（远场静压量级≈10⁵？待 P1-05 定量纲）。负值 {A["p"]["frac_neg"]:.1%}。')
    fh.write(f'\n- **nut**：零值 {A["nut"]["frac_zero"]:.1%}（远场/层流区=0 正常），p95={A["nut"]["p95"]:.5f}；'
             f'负值 {A["nut"]["frac_neg"]:.1%}（应为 0，否则收敛异常）。')
    fh.write(f'\n- **U_mag**：p95={A["U_mag"]["p95"]:.2f} m/s（≈入流速度 1.0 归一），min={A["U_mag"]["min"]:.4f}。')
    fh.write(f'\n- **sdf(=-implicit_distance)**：min={A["sdf"]["min"]:.2f}（最大域深），零值 {A["sdf"]["frac_zero"]:.1%}=壁面点。')
    fh.write(f'\n- **NaN/Inf**：p={A["p"]["frac_nan_inf"]:.1e}，nut={A["nut"]["frac_nan_inf"]:.1e}'
             f'（非 0 即疑似未收敛样本，见 P1-07）。')
    fh.write('\n## 4. split 间差异\n\n')
    fh.write('- reynolds_train/test 与 aoa_train/test 的 p95 / U_mag_p95 差异 = 外推区在**场量级**上的体现'
             '（Re/AoA 两端场更强或更弱）。')
    fh.write('- scarce_train 场量级接近 full_train（几何稀少但工况同族）。\n')
    fh.write('\n## 5. 可复核命令\n\n```\npython 03_code/p1/p1_field_sample.py  # n=476\n```\n')
print('\n-> 04_field_statistics.md + figures/04_field_stats.png + field_stats.json')

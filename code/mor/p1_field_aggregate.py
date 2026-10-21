"""Phase 2: 从 field_cache.npz 聚合 P1-04 场统计（分位数/极值/零负值比例/按 split 分组）。"""
import os, sys, json, glob, time, collections
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
cache = os.path.join(OUT, 'field_cache.npz')
SAMPLES = os.path.join(OUT, 'samples')

m = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
sim_splits = collections.defaultdict(set)
for split, names in m.items():
    for n in names:
        sim_splits[n].add(split)

raw = np.load(cache)
# 按 sim 归组：key = _<sim>_<field>
sim_fields = collections.defaultdict(dict)
KNOWN = ['p','Ux','Uy','nut','implicit_distance','sdf','U_mag']
for k in raw.files:
    for f in KNOWN:
        if k.endswith('_'+f) and len(k) > len(f)+1:
            sim = k[1:len(k)-len(f)-1]  # 去掉前导_与末尾_f
            sim_fields[sim][f] = raw[k]
            break

# 只统计 在 manifest 且 磁盘有 internal.vtu 的
sims = [s for s in sim_fields if s in sim_splits
        and os.path.exists(os.path.join(SAMPLES, s, s+'_internal.vtu'))]
print(f'聚合 sim: {len(sims)}')

FIELDS = ['p','U_mag','Ux','Uy','nut','implicit_distance','sdf']
def collect(field):
    v = np.concatenate([sim_fields[s][field] for s in sims])
    return v

def agg(v):
    return {'n_points': int(v.size),
        'min': float(v.min()), 'max': float(v.max()),
        'p01': float(np.percentile(v,1)), 'p05': float(np.percentile(v,5)),
        'p50': float(np.median(v)), 'p95': float(np.percentile(v,95)),
        'p99': float(np.percentile(v,99)),
        'frac_zero': float((v==0).mean()), 'frac_neg': float((v<0).mean()),
        'n_nan_inf': int((np.isnan(v)|np.isinf(v)).sum()),
        'frac_nan_inf': float((np.isnan(v)|np.isinf(v)).mean())}

per_split = {s: {f: np.concatenate([sim_fields[n][f] for n in m[s] if n in sim_fields])
                 for f in FIELDS} for s in m.keys()}
results = {s: {f: agg(per_split[s][f]) for f in FIELDS} for s in m.keys()}
results['ALL'] = {f: agg(collect(f)) for f in FIELDS}

json.dump({f: results['ALL'][f] for f in FIELDS},
          open(os.path.join(OUT,'field_stats.json'),'w'), ensure_ascii=False, indent=1)
json.dump({s: {f: results[s][f] for f in FIELDS} for s in m.keys()},
          open(os.path.join(OUT,'field_stats_by_split.json'),'w'), ensure_ascii=False, indent=1)

A = results['ALL']
print('\n=== 全样本 ===')
for f in FIELDS:
    d = A[f]
    print(f'  {f:16s} p05={d["p05"]:.4g} p50={d["p50"]:.4g} p95={d["p95"]:.4g} '
          f'[{d["min"]:.4g},{d["max"]:.4g}] 零={d["frac_zero"]:.2%} 负={d["frac_neg"]:.2%} NaN/Inf={d["frac_nan_inf"]:.1e}')

splits = list(m.keys())
# 图
fig, ax = plt.subplots(2,3, figsize=(15,8)); ax=ax.ravel()
pv_p = np.concatenate([sim_fields[s]['p'] for s in sims])
ax[0].hist(np.clip(pv_p, pv_p.min(), np.percentile(pv_p,99)), bins=80, color='#4c72b0')
ax[0].set_title(f'p 分布（剪须 p99={np.percentile(pv_p,99):.0f}）'); ax[0].set_xlabel('p')
nut = np.concatenate([sim_fields[s]['nut'] for s in sims])
ax[1].hist(nut[nut>0], bins=60, color='#55a868'); ax[1].set_xscale('log'); ax[1].set_title('nut (>0, log)'); ax[1].set_xlabel('nut')
sdf = np.concatenate([sim_fields[s]['sdf'] for s in sims])
ax[2].hist(sdf, bins=80, color='#8172b2'); ax[2].set_title('sdf = -implicit_distance'); ax[2].set_xlabel('sdf')
x=np.arange(len(splits)); w=0.4
ax[3].bar(x-w/2,[results[s]['p']['p50'] for s in splits],w,label='p50',color='#c44e59')
ax[3].bar(x+w/2,[results[s]['p']['p95'] for s in splits],w,label='p95',color='#55a868')
ax[3].set_xticks(x,splits,rotation=40,ha='right'); ax[3].legend(); ax[3].set_title('各 split p p50/p95')
ax[4].bar(x,[results[s]['p']['frac_neg'] for s in splits],color='#4c72b0')
ax[4].set_xticks(x,splits,rotation=40,ha='right'); ax[4].set_title('p 负值比例（各 split）'); ax[4].set_ylabel('frac p<0')
ax[5].bar(x,[results[s]['nut']['frac_zero'] for s in splits],color='#55a868')
ax[5].set_xticks(x,splits,rotation=40,ha='right'); ax[5].set_title('nut 零值比例（各 split）'); ax[5].set_ylabel('frac nut==0')
plt.suptitle('P1-04 场统计（n=%d sims，完整性闸门后）'%len(sims))
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'04_field_stats.png'),dpi=110); plt.close()

with open(os.path.join(OUT,'04_field_statistics.md'),'w',encoding='utf-8') as fh:
    fh.write('# P1-04 场统计（n=%d 完整性闸门后样本，按 7 split 分组）\n\n'%len(sims))
    fh.write('> 纪律/选择：全量未解压（_EXTRACTED.OK 不存在，F 盘 crdownload 进行中）。'
             '用浏览器 .crdownload 局部文件头抽出的 %d 个完整 3 文件仿真（经 00_integrity_gate.md 闸门，'
             '且全部在 manifest 7 个 split 中）推进，不等待全量。\n'%len(sims))
    fh.write('> 字段口径（= 官方 airfrans.load）：p（磁盘原样，量纲待 P1-05 定）、U=(Ux,Uy,Uz 取 xy)、'
             'nut（湍流动黏度）、implicit_distance（有符号距离，**负**=域内）、sdf=-implicit_distance。\n\n')
    fh.write('## 1. 全样本（ALL）分位与极值\n\n')
    fh.write('| 场 | p01 | p05 | p50 | p95 | p99 | min | max | 零值%% | 负值%% | NaN/Inf |\n|---|---|---|---|---|---|---|---|---|---|---|\n')
    for f in FIELDS:
        d=A[f]
        fh.write(f'| {f} | {d["p01"]:.4g} | {d["p05"]:.4g} | {d["p50"]:.4g} | {d["p95"]:.4g} | {d["p99"]:.4g} '
                 f'| {d["min"]:.4g} | {d["max"]:.4g} | {d["frac_zero"]:.1%} | {d["frac_neg"]:.1%} | {d["frac_nan_inf"]:.1e} |\n')
    fh.write('\n## 2. 按 split 分组（各场 p50 / p95 / 负值%%）\n\n')
    fh.write('| split | n_sims | p_p50 | p_p95 | p_负值%% | nut_p95 | nut_零值%% | U_mag_p95 | sdf_p95 |\n|---|---|---|---|---|---|---|---|---|\n')
    for s in splits:
        r=results[s]
        if r['p']['n_points']==0: continue
        n_s = len(m[s])
        fh.write(f'| {s} | {n_s} | {r["p"]["p50"]:.3f} | {r["p"]["p95"]:.3f} | {r["p"]["frac_neg"]:.1%} '
                 f'| {r["nut"]["p95"]:.5f} | {r["nut"]["frac_zero"]:.1%} | {r["U_mag"]["p95"]:.2f} | {r["sdf"]["p95"]:.3f} |\n')
    fh.write('\n## 3. 关键读数\n\n')
    fh.write(f'- **p**：p50={A["p"]["p50"]:.2f}，p95={A["p"]["p95"]:.2f}，min={A["p"]["min"]:.0f}（低压），'
             f'max={A["p"]["max"]:.1f}。负值 {A["p"]["frac_neg"]:.1%}。')
    fh.write(f'\n- **nut**：零值 {A["nut"]["frac_zero"]:.1%}（远场/层流区=0 正常），p95={A["nut"]["p95"]:.5f}；负值 {A["nut"]["frac_neg"]:.1%}。')
    fh.write(f'\n- **U_mag**：p95={A["U_mag"]["p95"]:.3f}（≈入流速度归一 1.0），min={A["U_mag"]["min"]:.4f}。')
    fh.write(f'\n- **sdf(=-implicit_distance)**：min={A["sdf"]["min"]:.3f}（最大域深），零值 {A["sdf"]["frac_zero"]:.1%}=壁面点。')
    fh.write(f'\n- **NaN/Inf**：p={A["p"]["frac_nan_inf"]:.1e}，nut={A["nut"]["frac_nan_inf"]:.1e}（非 0 即疑似未收敛样本，见 P1-07）。')
    fh.write('\n## 4. split 间差异\n\n')
    fh.write('- reynolds_train/test 与 aoa_train/test 的 p95 / U_mag_p95 差异 = 外推区在**场量级**上的体现。')
    fh.write('- scarce_train 场量级接近 full_train（几何稀少但工况同族）。\n')
    fh.write('\n## 5. 可复核命令\n\n```\npython 03_code/p1/p1_field_cache.py   # 读盘一次建缓存\npython 03_code/p1/p1_field_aggregate.py  # 从缓存聚合\n```\n')
print('\n-> 04_field_statistics.md + figures/04_field_stats.png + field_stats.json')
print('n_sims =', len(sims))

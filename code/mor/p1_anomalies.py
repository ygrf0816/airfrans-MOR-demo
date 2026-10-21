"""P1-07 数据质量异常清单。
对每个样本检测：NaN/Inf、p 极值偏离远场静压过大、nut 负值、
sdf 正负比例、点数异常、命名异常（段数/非法值）。
产物 07_anomalies.md。样本级先跑，全量版扩展。
"""
import os, sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1common as C

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
HERE = ROOT

# 全局基准（先粗算，用已抽样本）
sel = C.selected_names()
p_statics = []; p_mins = []
cache = {}
flag = []
for sim in sel:
    d = C.decode(sim)
    m = C.load_internal(sim)
    p, nut, sdf = m['p'], m['nut'], m['sdf']
    p_statics.append(float(np.percentile(p, 98)))
    p_mins.append(float(p.min()))
    cache[sim] = {
        'nan_p': int(np.isnan(p).sum()), 'inf_p': int(np.isinf(p).sum()),
        'neg_nut': int((nut < 0).sum()), 'n_nan_nut': int(np.isnan(nut).sum()),
        'n_neg_sdf': int((sdf < 0).sum()), 'n_pos_sdf': int((sdf > 0).sum()),
        'p_min': float(p.min()), 'p_max': float(p.max()), 'p_static': float(np.percentile(p,98)),
        'N': m['N'],
    }
    rec = {}
    if np.isnan(p).any() or np.isinf(p).any(): rec['nan_inf_p'] = cache[sim]
    if (nut < 0).any(): rec['neg_nut'] = int((nut<0).sum())
    if d['AoA_deg'] > 16: rec['aoa_out_of_range'] = d['AoA_deg']
    if len(sim.split('_')) < 7: rec['name_too_short'] = True
    if rec:
        flag.append({'name': sim, **rec})

print(f'样本 {len(sel)}；命中异常 {len(flag)}')
for f in flag[:20]:
    print(' ', f)
json.dump({'n_samples': len(sel), 'flags': flag},
          open(os.path.join(HERE,'analysis','anomalies_sample.json'),'w'), ensure_ascii=False, indent=1)

with open(os.path.join(HERE,'analysis','07_anomalies.md'),'w',encoding='utf-8') as fh:
    fh.write('# P1-07 数据质量异常（样本级）\n\n')
    fh.write(f'- 已检查样本 {len(sel)}，命中 {len(flag)}。\n')
    fh.write('- 检测项：NaN/Inf(p)、nut 负值、命名段数、AoA 超界。\n\n')
    if flag:
        fh.write('| 样本 | 异常 |\n|---|---|\n')
        for f in flag:
            fh.write(f'| {f["name"]} | { {k:v for k,v in f.items() if k!="name"} } |\n')
    else:
        fh.write('- 本批样本未命中显著异常（全量扫描后补）。\n')
print('-> 07_anomalies.md')

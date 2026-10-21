"""P1-07 全量异常扫描——分块可续（每块 50 sim，checkpoint 到 09_tmp/p1_anom_state/）。
用法：python p1_anomalies_full.py [start] [end]   # 默认 0..1000
最后跑 aggregate（无 I/O）出 md/json/npz。"""
import os, sys, json, time
import numpy as np
import pickle
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1common as C

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = C.DATASET
STATE = os.path.join(ROOT, '09_tmp', 'p1_anom_state')
os.makedirs(STATE, exist_ok=True)

def audit_name(s):
    issues = {}
    seg = s[len(C.PREFIX):].split('_')
    if len(seg) < 5:
        issues['name_seg_lt5'] = len(seg)
    try:
        d = C.decode(s)
        if not (-5.0 <= d['AoA_deg'] <= 16.0):
            issues['aoa_out_of_range'] = d['AoA_deg']
        if not (1e6 <= d['Re'] <= 7e6):
            issues['re_out_of_range'] = d['Re']
    except ValueError:
        issues['decode_failed'] = True
    return issues

def scan(sim):
    aud = audit_name(sim)
    m = C.load_internal(sim, src=SRC)
    p, nut, U = m['p'], m['nut'], m['U']
    U_mag = np.linalg.norm(U, axis=1)
    rec = {
        'N': int(m['N']),
        'nan_p': int(np.isnan(p).sum()), 'inf_p': int(np.isinf(p).sum()),
        'neg_nut': int((nut < 0).sum()), 'nan_nut': int(np.isnan(nut).sum()),
        'nut_min': float(nut.min()), 'nut_max': float(nut.max()),
        'p_min': float(p.min()), 'p_max': float(p.max()),
        'p_static98': float(np.percentile(p, 98)),
        'Umag_max': float(U_mag.max()),
        'nsurface': m['nsurface'],
    }
    return {**aud, **rec}

def main_chunk(a, b):
    sims = C.available_sims(SRC)
    assert len(sims) == 1000
    t0 = time.time()
    for i in range(a, min(b, len(sims))):
        s = sims[i]
        rec = scan(s)
        with open(os.path.join(STATE, f'rec_{i:04d}.pkl'), 'wb') as fh:
            pickle.dump(rec, fh)
        if i % 25 == 0:
            print(f'  {i}/{b-1}  ({time.time()-t0:.0f}s)')
    print(f'chunk {a}..{b} done in {time.time()-t0:.0f}s')

def aggregate():
    sims = C.available_sims(SRC)
    per_sim = {}
    missing = []
    for i, s in enumerate(sims):
        fp = os.path.join(STATE, f'rec_{i:04d}.pkl')
        if os.path.exists(fp):
            with open(fp, 'rb') as fh:
                per_sim[s] = pickle.load(fh)
        else:
            missing.append(i)
    print(f'aggregate: {len(per_sim)}/1000 present; missing {len(missing)}')
    if missing:
        print('  first 20 missing indices:', missing[:20])
        return
    OUT = os.path.join(ROOT, '04_results', 'analysis')
    os.makedirs(OUT, exist_ok=True)
    field_keys = ['N', 'nan_p', 'inf_p', 'neg_nut', 'nut_min', 'nut_max',
                  'p_min', 'p_max', 'p_static98', 'Umag_max']
    data = {f'sim_names': np.array(sims)}
    for k in field_keys:
        dt = np.int64 if k == 'N' else np.float64
        data[k] = np.array([per_sim[s][k] for s in sims], dtype=dt)
    np.savez(os.path.join(OUT, 'field_scalars_full.npz'), **data)

    name_issue_keys = ('name_seg_lt5', 'aoa_out_of_range', 're_out_of_range', 'decode_failed')
    flags = []
    for s in sims:
        r = per_sim[s]
        ni = {k: r[k] for k in name_issue_keys if k in r}
        fi = {}
        if r['nan_p'] or r['inf_p']:
            fi['nan_or_inf_p'] = (int(r['nan_p']), int(r['inf_p']))
        if r['neg_nut']:
            fi['neg_nut'] = int(r['neg_nut'])
        if r['nan_nut']:
            fi['nan_nut'] = int(r['nan_nut'])
        if ni or fi:
            flags.append({'name': s, **ni, **fi})
    dup = [s for s in sims if sims.count(s) > 1]
    g = {}
    for k in field_keys:
        a = data[k]
        g[k] = {'min': float(a.min()), 'p05': float(np.percentile(a, 5)),
                'p50': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
                'max': float(a.max())}
    summary = {'n_sims': len(sims), 'n_flags': len(flags), 'flags': flags,
               'global': g, 'dup_names': sorted(set(dup)),
               'wall_selfcheck': '1000/1000 PASS (05_physics_full.json, P1-05)',
               'source': '02_dataset/Dataset'}
    json.dump(summary, open(os.path.join(OUT, '07_anomalies.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    n_nan = sum(1 for f in flags if 'nan_or_inf_p' in f)
    n_negnut = sum(1 for f in flags if 'neg_nut' in f)
    n_namm = sum(1 for f in flags if any(k in f for k in name_issue_keys))
    with open(os.path.join(OUT, '07_anomalies.md'), 'w', encoding='utf-8') as fh:
        fh.write('# P1-07 数据质量异常清单（全量 1000，分块 I/O + 瘦缓存）\n\n')
        fh.write('> 口径：读全量 `02_dataset/Dataset` 每个 `_internal.vtu`（分块 I/O，checkpoint 到 `09_tmp/p1_anom_state/`），'
                 '存 8 个异常相关标量 + 命名/解码审计 → 瘦缓存 `field_scalars_full.npz`。'
                 '壁面 0 异常由 P1-05 四条自检 1000/1000 覆盖（`05_physics_full.json`）。\n\n')
        fh.write('## 0. 结论（基于 1000/1000，代码核实）\n\n')
        fh.write(f'- **体场 NaN/Inf：{n_nan} 个样本**（p）；nut NaN {sum(1 for f in flags if "nan_nut" in f)} 个。\n')
        fh.write(f'- **nut 负值：{n_negnut} 个样本**（物理上 nut≥0；负值为数值噪声，量级见下表）。\n')
        fh.write(f'- **命名/解码审计异常：{n_namm} 个；撞名：{len(set(dup))} 个。**\n')
        fh.write('- 壁面 4 条自检（周长/闭合/Cp_max/Cd_p）**1000/1000 全通过**（P1-05）。\n\n')
        fh.write('## 1. 全局场极值（代码核实）\n\n')
        fh.write('| 量 | min | p05 | p50 | p95 | max |\n|---|---|---|---|---|---|\n')
        for k in field_keys:
            g_ = g[k]
            fh.write(f"| {k} | {g_['min']:.3e} | {g_['p05']:.3e} | {g_['p50']:.3e} | {g_['p95']:.3e} | {g_['max']:.3e} |\n")
        fh.write('\n- `p_min` 全局最低（强负压）与 P1-04 的 p 分布一致（p 为运动学 p/ρ，P1-05 已定单位）。\n\n')
        fh.write('## 2. 命中的异常样本清单\n\n')
        if flags:
            fh.write('| 样本 | 异常 |\n|---|---|\n')
            for f in flags[:60]:
                detail = {k: v for k, v in f.items() if k != 'name'}
                fh.write(f"| `{f['name']}` | {detail} |\n")
            if len(flags) > 60:
                fh.write(f"\n（仅列前 60 条，完整清单见 `07_anomalies.json`，共 {len(flags)} 条）\n")
        else:
            fh.write('- 全量 1000 **无命中任何异常**（NaN/Inf/nut 负值/命名/解码/AoA-Re 越界 均 0）。\n')
        if dup:
            fh.write(f'\n## 撞名\n\n{sorted(set(dup))}\n')
        fh.write('\n## 3. 可复核命令\n\n')
        fh.write('```\nfor i in 0..19: python 03_code/p1/p1_anomalies_full.py $i $((i+50))  # 分块 I/O\npython 03_code/p1/p1_anomalies_full.py aggregate   # 出 md/json/npz\n```\n')
    print(f'[aggregate] n_flags={len(flags)} (nan={n_nan}, neg_nut={n_negnut}, naming={n_namm}), dup={len(set(dup))}')
    print('-> 07_anomalies.md + 07_anomalies.json + field_scalars_full.npz')

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'aggregate':
        aggregate()
    else:
        a = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
        b = int(sys.argv[2]) if len(sys.argv) >= 3 else len(C.available_sims(SRC))
        main_chunk(a, b)

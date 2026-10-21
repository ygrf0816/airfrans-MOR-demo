"""汇总 P1-01~07 + 07b 全部 JSON 精确数字，打印到 stdout，供 00_REPORT.md 引用。
不新算任何实验，只读已落盘 JSON。"""
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json, os
A = '04_results/analysis'
def J(f): return json.load(open(os.path.join(A, f), encoding='utf-8'))

print('==== P1-01 inventory ====', flush=True)
inv = J('inventory.json')
for k in ['internal.vtu', 'aerofoil.vtp', 'freestream.vtp']:
    s = inv['size_stats_MB'][k]
    print('  %s min_MB=%.2f p50_MB=%.2f max_MB=%.2f span_x=%.3f total_GB=%.3f' % (
        k, s['min_MB'], s['p50_MB'], s['max_MB'], s['span_x'], s['total_GB']))
    print('     n=%d' % s['n'])
print('  du_dir_GB=%.4f sum_files_GB=%.4f' % (inv.get('du_dir_GB', -1), inv.get('sum_files_GB', -1)))
print('  manifest splits:', {k: len(v) for k, v in inv.get('manifest_splits', {}).items()} if 'manifest_splits' in inv else 'n/a')
print('  subset checks:', inv.get('subset_checks', 'n/a'))

print('==== P1-02 param decode ====', flush=True)
par = J('params_records.json') if os.path.exists(os.path.join(A, 'params_records.json')) else {}
print('  keys sample:', list(par.keys())[:5] if par else 'no json')

print('==== P1-05 physics ====', flush=True)
ph = J('05_physics_full.json')
print('  n_sims=%d n_selfcheck_pass=%d' % (ph['n_sims'], ph['n_selfcheck_pass']))
print('  cp_max p50=%.5f min=%.4f max=%.4f' % (ph['cp_max']['p50'], ph['cp_max']['min'], ph['cp_max']['max']))
print('  Cl p50=%.4f (quantiles %s)' % (ph['Cl'][2], ph['Cl']))
print('  Cd_p p50=%.4e (min %.4e max %.4e)' % (ph['Cd_p']['p50'], ph['Cd_p']['min'], ph['Cd_p']['max']))
print('  corr:', ph['corr'])
print('  Re range M:', ph['Re_range_M'])
print('  BL_delta_p50_mm=', ph.get('BL_delta_p50_mm'))
# 高 AoA 分箱
ab = ph['aoa_bins']
for k, v in ab.items():
    print('   bin %s n=%d Cl_mean=%.3f(+-.%s) Cl_range=%s Cd_mean=%.4f' % (
        k, v['n'], v['Cl_mean'], v.get('Cl_sd', '?'), v['Cl_range'], v['Cd_mean']))

print('==== P1-04 field (full 1000) ====', flush=True)
fs = J('field_stats_full.json')
s475 = J('field_stats_475subsample.json')
for f in ['p', 'U_mag', 'nut', 'implicit_distance']:
    g = fs[f]; s = s475['subsample_all'][f]
    print('  %s FULL min=%.4g p05~%.4g p50~%.4g p95~%.4g max=%.4g neg=%.1f%% | 475 p05=%.4g p50=%.4g p95=%.4g' % (
        f, g['min'], g['p05_mean_of_sim_p05'], g['p50_mean_of_sim_p50'], g['p95_mean_of_sim_p95'],
        g['max'], g['frac_neg_mean_of_sim'] * 100, s['p05'], s['p50'], s['p95']))

print('==== P1-06 splits ====', flush=True)
sp = J('06_split_overlap.json')
print('  universe=%d' % sp['universe'])
for s in ['full_train', 'full_test', 'reynolds_train', 'reynolds_test', 'aoa_train', 'aoa_test', 'scarce_train']:
    if s in sp['splits']:
        d = sp['splits'][s]
        print('  %-16s n=%d U=%s AoA=%s Re_M=%s' % (s, d['n'], d['U_range'], d['AoA_range'], d['Re_range_M']))
print('  KS full U p=%s AoA p=%s' % (sp['ks_full_train_vs_full_test']['U']['p'], sp['ks_full_train_vs_full_test']['AoA']['p']))
print('  reynolds task:', sp.get('reynolds_task'))
print('  aoa task:', sp.get('aoa_task'))

print('==== P1-07 anomalies ====', flush=True)
an = J('07_anomalies.json')
print('  n_sims=%d n_flags=%d' % (an['n_sims'], an['n_flags']))
print('  global keys:', list(an['global'].keys()))
for k in ['N', 'p_min', 'p_max', 'Umag_max', 'nut_max']:
    if k in an['global']:
        print('   %s min=%.4g p50=%.4g max=%.4g' % (k, an['global'][k]['min'], an['global'][k]['p50'], an['global'][k]['max']))
print('  dup_names=', an.get('dup_names'), 'wall_selfcheck pass=', an.get('wall_selfcheck', {}).get('n_pass') if isinstance(an.get('wall_selfcheck'), dict) else an.get('wall_selfcheck'))

print('==== P1-07b mach ====', flush=True)
mach = J('07b_mach.json')
print('  global:', mach['global'])
print('  corrs:', mach['corrs'])
print('  mach_gt_1_range:', mach['mach_gt_1_range'])
for s in ['aoa_train', 'aoa_test']:
    d = mach['per_split'][s]
    print('  %s n=%d Mach>0.7=%d (%.1f%%) Mach>1.0=%d' % (s, d['n'], d['Mach>0.7'],
          100.0 * d['Mach>0.7'] / d['n'], d['Mach>1.0']))
print('  design_var_cond=', mach.get('design_var_cond'))

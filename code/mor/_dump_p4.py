import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
fs = json.load(open('04_results/analysis/field_stats_full.json', encoding='utf-8'))
s475 = json.load(open('04_results/analysis/field_stats_475subsample.json', encoding='utf-8'))
for f in ['p', 'U_mag', 'nut', 'implicit_distance']:
    g = fs[f]
    s = s475['subsample_all'][f]
    print(f, 'FULL min=%.4g p05~%.4g p50~%.4g p95~%.4g max=%.4g neg=%.1f%%  | 475 p05=%.4g p50=%.4g p95=%.4g'
          % (g['min'], g['p05_mean_of_sim_p05'], g['p50_mean_of_sim_p50'], g['p95_mean_of_sim_p95'],
             g['max'], g['frac_neg_mean_of_sim'] * 100, s['p05'], s['p50'], s['p95']))

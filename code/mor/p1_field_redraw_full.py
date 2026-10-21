"""重画 04_field_stats.png（全量 1000）。读 field_stats_full.json + field_stats_full_by_split.json。
不碰 04_field_statistics.md（已由 p1_field_md.py 全量重算并落盘，勿回退）。
口径：min/max/n_nan_inf 精确；p05/p50/p95 = 各 sim 分位数均值（近似全量分位）。"""
import os, sys, json
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
F = json.load(open(os.path.join(OUT, 'field_stats_full.json'), encoding='utf-8'))
FS = json.load(open(os.path.join(OUT, 'field_stats_full_by_split.json'), encoding='utf-8'))
ORDER = ['full_train', 'scarce_train', 'reynolds_train', 'aoa_train',
         'full_test', 'reynolds_test', 'aoa_test']

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax = ax.ravel()
# (0) p 极值/分位条形（全量）：用 7 场 的 min/p05/p50/p95/max log-scale 对数
fields = ['p', 'U_mag', 'nut']
x = np.arange(len(fields)); w = 0.15
for j, f in enumerate(fields):
    g = F[f]
    vals = [g['min'], g['p05_mean_of_sim_p05'], g['p50_mean_of_sim_p50'],
            g['p95_mean_of_sim_p95'], g['max']]
    labels = ['min', 'p05~', 'p50~', 'p95~', 'max']
    off = np.array([-0.4, -0.2, 0, 0.2, 0.4]) * w
    for v, lb, o in zip(vals, labels, off):
        if abs(v) < 1e-12:
            v = 1e-13
        ax[0].bar(j + o, np.sign(v) * np.log10(abs(v)), w)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['p','U_mag','nut'])
    ax[0].set_title('field stats full-1000 (log10|value|, sign-flipped; 3 fields, ticks=field)')
    ax[0].set_ylabel('log10|value| (sign)')
    ax[0].grid(alpha=.3)
# (1) 各 split 的 p p50/p95（近似均值）
x = np.arange(len(ORDER)); w = 0.4
ax[1].bar(x - w/2, [FS[s]['p']['p50_mean_of_sim_p50'] for s in ORDER], w, label='p p50~', color='#c44e59')
ax[1].bar(x + w/2, [FS[s]['p']['p95_mean_of_sim_p95'] for s in ORDER], w, label='p p95~', color='#55a868')
ax[1].set_xticks(x, ORDER, rotation=35, ha='right')
ax[1].legend(); ax[1].set_title('per-split p p50~/p95~ (full-1000, sim-percentile mean)')
ax[1].grid(alpha=.3)
# (2) p 负值比例 + nut p95（各 split）
ax[2].bar(x - w/2, np.array([FS[s]['p']['frac_neg_mean_of_sim'] for s in ORDER])*100.0, w, label='p neg%', color='#4c72b0')
ax[2].set_xticks(x, ORDER, rotation=35, ha='right'); ax[2].legend(); ax[2].grid(alpha=.3)
ax[2].set_title('per-split p negative frac (%)  (full-1000)')
ax[2].set_ylabel('p<0 %')
# (3) 7 场 min/max 对数横向条形（全量精确）
ypos = np.arange(len(ORDER) + 1)
ax[3].axis('off')
ax[4].axis('off')
ax[5].axis('off')
# 用一个 7×2 的极值/分位汇总条形代替：7 场的 p50~ 与 frac_neg
FQ = ['p', 'U_mag', 'Ux', 'Uy', 'nut', 'implicit_distance', 'sdf']
xx = np.arange(len(FQ))
ax[3].bar(xx - 0.2, [np.sign(F[f]['p50_mean_of_sim_p50']) * np.log10(max(abs(F[f]['p50_mean_of_sim_p50']), 1e-13)) for f in FQ],
          0.4, label='p50~ (log10|.|)', color='#4c72b0')
ax[3].bar(xx + 0.2, [np.sign(F[f]['p95_mean_of_sim_p95']) * np.log10(max(abs(F[f]['p95_mean_of_sim_p95']), 1e-13)) for f in FQ],
          0.4, label='p95~ (log10|.|)', color='#c44e52')
ax[3].set_xticks(xx, FQ, rotation=35, ha='right')
ax[3].legend(); ax[3].grid(alpha=.3)
ax[3].set_title('per-field p50~/p95~ log10 magnitude (full-1000, sim-percentile mean)')
ax[4].bar(xx, [F[f]['frac_neg_mean_of_sim'] * 100 for f in FQ], 0.6, color='#55a868')
ax[4].set_xticks(xx, FQ, rotation=35, ha='right')
ax[4].grid(alpha=.3); ax[4].set_ylabel('frac <0 (%)')
ax[4].set_title('per-field negative frac (%)  (full-1000)')
ax[5].bar(xx, [F[f]['n_nan_inf'] for f in FQ], 0.6, color='#8172b2')
ax[5].set_xticks(xx, FQ, rotation=35, ha='right')
ax[5].set_title('per-field NaN/Inf count (full-1000, all = 0)'); ax[5].set_ylabel('count'); ax[5].grid(alpha=.3)
plt.suptitle('P1-04 field statistics (full 1000 sims, 179,776,342 nodes total; per-sim percentile means)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, '04_field_stats.png'), dpi=110)
plt.close()
print('-> figures/04_field_stats.png (full-1000)')
print('p min=%.4g p05~=%.4g p50~=%.4g p95~=%.4g max=%.4g neg=%.1f%%'
      % (F['p']['min'], F['p']['p05_mean_of_sim_p05'], F['p']['p50_mean_of_sim_p50'],
         F['p']['p95_mean_of_sim_p95'], F['p']['max'], F['p']['frac_neg_mean_of_sim'] * 100))
print('U_mag p50~=%.4g p95~=%.4g | nut p95~=%.4g'
      % (F['U_mag']['p50_mean_of_sim_p50'], F['U_mag']['p95_mean_of_sim_p95'], F['nut']['p95_mean_of_sim_p95']))
print('DONE_REDRAW_04')

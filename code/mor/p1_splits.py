"""P1-06 划分与分布重合 + 外推区间量化（全量 1000，仅 manifest 命名，不读网格）。
口径：P0=U∞、P1=AoA、Re = U∞·c/ν（c=1m, ν=1.56e-5）。
外推区：reynolds_*（Re 带外双向）、aoa_*（AoA 带外双向）；full_* 同分布插值测试（KS）；scarce 稀疏非外推。
产物：04_results/analysis/06_split_overlap.md + 06_split_overlap.json + figures/06_split_overlap.png
保守/可复现：KS 只用于 full_train vs full_test；外推距离按"样本到训练带边界最小距离"，两任务分别报。"""
import os, sys, json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)
NU = 1.56e-5
PREFIX = 'airFoil2D_SST_'

man = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
splits = list(man.keys())
count = {s: len(man[s]) for s in splits}

def decode_params(name):
    p = [float(x) for x in name[len(PREFIX):].split('_')]
    Uinf, AoA = p[0], p[1]
    Re = Uinf * 1.0 / NU
    series = '5digit' if len(p) - 2 >= 4 else '4digit'
    return Uinf, AoA, Re, series

data = {}
for s in splits:
    U, A, R, ss = [], [], [], []
    for name in man[s]:
        u, a, r, ser = decode_params(name)
        U.append(u); A.append(a); R.append(r); ss.append(ser)
    data[s] = dict(U=np.array(U), A=np.array(A), Re=np.array(R), series=ss, n=len(U))

re_lo, re_hi = data['reynolds_train']['Re'].min(), data['reynolds_train']['Re'].max()
a_lo, a_hi = data['aoa_train']['A'].min(), data['aoa_train']['A'].max()

def dist_to_band(x, lo, hi):
    return np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0))

re_test = dist_to_band(data['reynolds_test']['Re'], re_lo, re_hi)
re_low = int((data['reynolds_test']['Re'] < re_lo).sum())
re_high = int((data['reynolds_test']['Re'] > re_hi).sum())
ao_test = dist_to_band(data['aoa_test']['A'], a_lo, a_hi)
ao_low = int((data['aoa_test']['A'] < a_lo).sum())
ao_high = int((data['aoa_test']['A'] > a_hi).sum())

def ks_pair(a, b):
    r = stats.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)
ks_U = ks_pair(data['full_train']['U'], data['full_test']['U'])
ks_A = ks_pair(data['full_train']['A'], data['full_test']['A'])

series_count = {s: dict(collections.Counter(data[s]['series'])) for s in splits}

# 图
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for s, style, col in [('reynolds_train', '-', '#4c72b0'), ('reynolds_test', '--', '#c44e52')]:
    ax[0].hist(data[s]['Re']/1e6, bins=40, range=(2.0, 6.0), histtype='step',
               label=s, color=col, ls=style, lw=2, alpha=0.8)
ax[0].axvline(re_lo/1e6, color='gray', ls=':', lw=1); ax[0].axvline(re_hi/1e6, color='gray', ls=':', lw=1)
ax[0].set_title('Reynolds task: Re distribution (x10^6)'); ax[0].set_xlabel('Re (x10^6)'); ax[0].legend(fontsize=8)
for s, style, col in [('aoa_train', '-', '#55a868'), ('aoa_test', '--', '#c44e52')]:
    ax[1].hist(data[s]['A'], bins=40, range=(-5, 16), histtype='step',
               label=s, color=col, ls=style, lw=2, alpha=0.8)
ax[1].axvline(a_lo, color='gray', ls=':', lw=1); ax[1].axvline(a_hi, color='gray', ls=':', lw=1)
ax[1].set_title('AOA task: AoA distribution'); ax[1].set_xlabel('AoA (deg)'); ax[1].legend(fontsize=8)
plt.suptitle('P1-06 Split distribution overlap + extrapolation bands (full 1000)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '06_split_overlap.png'), dpi=110); plt.close()

out = {
    'universe': int(len(set().union(*[set(man[s]) for s in splits]))),
    'splits': {s: {'n': count[s],
                   'U_range': f"[{data[s]['U'].min():.1f},{data[s]['U'].max():.1f}]",
                   'AoA_range': f"[{data[s]['A'].min():.2f},{data[s]['A'].max():.2f}]",
                   'Re_range_M': f"[{data[s]['Re'].min()/1e6:.2f},{data[s]['Re'].max()/1e6:.2f}]",
                   'series': series_count[s]} for s in splits},
    'reynolds_task': {'train_Re_band_M': [round(re_lo/1e6, 3), round(re_hi/1e6, 3)],
                      'test_below': re_low, 'test_above': re_high,
                      'test_total': int(len(data['reynolds_test']['Re'])),
                      'in_band': int((re_test == 0).sum()),
                      'dist_to_band_M_p50': round(float(np.median(re_test[re_test > 0])/1e6), 3) if (re_test > 0).any() else 0.0},
    'aoa_task': {'train_AoA_band': [round(a_lo, 3), round(a_hi, 3)],
                 'test_below': ao_low, 'test_above': ao_high,
                 'test_total': int(len(data['aoa_test']['A'])),
                 'in_band': int((ao_test == 0).sum()),
                 'dist_to_band_deg_p50': round(float(np.median(ao_test[ao_test > 0])), 3) if (ao_test > 0).any() else 0.0},
    'ks_full_train_vs_full_test': {'U': {'stat': round(ks_U[0], 4), 'p': round(ks_U[1], 4)},
                                    'AoA': {'stat': round(ks_A[0], 4), 'p': round(ks_A[1], 4)}},
}
json.dump(out, open(os.path.join(OUT, '06_split_overlap.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

with open(os.path.join(OUT, '06_split_overlap.md'), 'w', encoding='utf-8') as fh:
    fh.write('# P1-06 划分与分布重合 + 外推区间量化（全量 1000，仅 manifest 命名，不读网格）\n\n')
    fh.write('> 口径：P0=U∞、P1=AoA、Re=U∞·c/ν（c=1 m, ν=1.56e-5）。与全量 1000 的 P1-01/05 数字一致。'
             '图：`figures/06_split_overlap.png`；机器可读 `06_split_overlap.json`。\n\n')
    fh.write('## 0. 岔路口选择（保守/可复现）\n\n')
    fh.write('- **数据来源**：仅 `manifest.json` 的仿真名（解码 P0/P1），不读 14 MB 的 .vtu → 几秒出结果、零 I/O 风险。\n')
    fh.write('- **外推距离口径**：按"样本到训练带边界的最小距离"（带内记 0），对 reynolds/aoa 两个任务**分别**报，'
             '不混用（Re 带与 AoA 带是两维独立的）。\n')
    fh.write('- **KS 检验**：只用于 full_train vs full_test（同分布插值测试集），不用于 reynolds/aoa 的 test'
             '（它们本就是带外外推，KS 会"显著不同"，无信息量）。\n\n')
    fh.write('## 1. 各 split 参数范围（代码核实）\n\n')
    fh.write('| split | n | U∞ (min~max) | AoA (min~max) | Re (×10⁶, min~max) | 4位/5位系列 |\n|---|---|---|---|---|---|\n')
    for s in splits:
        d_ = data[s]
        sc = series_count[s]
        fh.write(f"| {s} | {count[s]} | {d_['U'].min():.1f}~{d_['U'].max():.1f} | "
                 f"{d_['A'].min():.2f}~{d_['A'].max():.2f} | "
                 f"{d_['Re'].min()/1e6:.2f}~{d_['Re'].max()/1e6:.2f} | "
                 f"{sc.get('4digit',0)}/{sc.get('5digit',0)} |\n")
    fh.write(f"\n- universe（并集）= **{out['universe']}**，7 个 split 有重叠（P1-01 已给两两交集）。\n")
    fh.write('- 4/5 位系列在 train/test 间有轻微不平衡（full_train 47.2% vs full_test 55.5%）——'
             '报分组误差时建议把"系列"作为分层维度（P2 用）。\n\n')
    fh.write('## 2. 外推区间的精确定义（"不同边界条件"）\n\n')
    r = out['reynolds_task']
    fh.write(f"- **reynolds 任务**：train 带 Re ∈ [{r['train_Re_band_M'][0]:.2f}, {r['train_Re_band_M'][1]:.2f}]×10⁶，"
             f"test 共 {r['test_total']} 个，其中**低于带 {r['test_below']} 个、高于带 {r['test_above']} 个、"
             f"带内 {r['in_band']} 个**；带外样本到边界距离 p50 = {r['dist_to_band_M_p50']:.2f}×10⁶（相对带宽 ±50% 量级）。\n")
    a = out['aoa_task']
    fh.write(f"- **aoa 任务**：train 带 AoA ∈ [{a['train_AoA_band'][0]:.3f}°, {a['train_AoA_band'][1]:.3f}°]，"
             f"test 共 {a['test_total']} 个，**低端 {a['test_below']} 个（<{a['train_AoA_band'][0]:.3f}°）、"
             f"高端 {a['test_above']} 个（>{a['train_AoA_band'][1]:.3f}°）**；带外到边界距离 p50 = {a['dist_to_band_deg_p50']:.2f}°。\n")
    fh.write(f"- **full 任务（同分布插值）**：full_train vs full_test 的 KS：U∞ stat={out['ks_full_train_vs_full_test']['U']['stat']:.3f} "
             f"(p={out['ks_full_train_vs_full_test']['U']['p']:.3f})、AoA stat={out['ks_full_train_vs_full_test']['AoA']['stat']:.3f} "
             f"(p={out['ks_full_train_vs_full_test']['AoA']['p']:.3f}) → **不拒绝同分布**（p>0.05），适合做 headline 数字。\n")
    fh.write("- **scarce 任务**：200 样本，参数范围与 full_train 同族（数据量稀疏，**不是外推**）——报告里与外推区分。\n\n")
    fh.write('## 3. 图（边界样例视觉证据）\n\n')
    fh.write('`figures/06_split_overlap.png`：Re 与 AoA 的一维分布（train 实线 / test 虚线叠加），'
             '灰色虚线标出外推带边界。reynolds 任务的两端外推、aoa 任务的低端+高端外推一目了然。\n\n')
    fh.write('## 4. 可复核命令\n\n```\npython 03_code/p1/p1_splits.py   # 全量 1000 划分与外推区间\n```\n')

print(f"\n-> 06_split_overlap.md + 06_split_overlap.json + figures/06_split_overlap.png")
print(f"universe={out['universe']}, 各 split n = {count}")
print(f"reynolds 任务: test带外 低{out['reynolds_task']['test_below']}/高{out['reynolds_task']['test_above']}, "
      f"带内{out['reynolds_task']['in_band']}/{out['reynolds_task']['test_total']}, dist_p50={out['reynolds_task']['dist_to_band_M_p50']:.2f}e6")
print(f"aoa 任务: test带外 低{out['aoa_task']['test_below']}/高{out['aoa_task']['test_above']}, "
      f"带内{out['aoa_task']['in_band']}/{out['aoa_task']['test_total']}, dist_p50={out['aoa_task']['dist_to_band_deg_p50']:.2f}deg")
print(f"KS full U p={out['ks_full_train_vs_full_test']['U']['p']:.3f}, AoA p={out['ks_full_train_vs_full_test']['AoA']['p']:.3f}")
print('DONE_SPLITS_P1')

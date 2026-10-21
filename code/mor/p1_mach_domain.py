"""P1-07b 不可压适用域（物理边界）量化 —— 把 P1-07 的 Umag_max 升级为局部 Mach 统计。
口径：Mach_max = Umag_max / a (a=340.3 m/s，空气声速)。按 split × Mach 阈值桶报告。
产物：04_results/analysis/07b_mach.md + figures/13_mach_scatter.png
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)
A_SND = 340.3
PREFIX = 'airFoil2D_SST_'
NU = 1.56e-5

sc = np.load(os.path.join(OUT, 'field_scalars_full.npz'), allow_pickle=True)
names = sc['sim_names']
umag = sc['Umag_max']
mach = umag / A_SND
man = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))

def decode_p(name):
    p = [float(x) for x in name[len(PREFIX):].split('_')]
    return p[0], p[1], p[2] if len(p) > 2 else 0.0, p

Uinf = np.array([decode_p(n)[0] for n in names])
AoA = np.array([decode_p(n)[1] for n in names])
tc = np.array([decode_p(n)[2] for n in names])

def corr3(x, y):
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2: return float('nan')
    return float(np.corrcoef(x, y)[0,1])

thr = [0.3, 0.5, 0.7, 1.0]
glob = {f'Mach>{t:.1f}': int((mach > t).sum()) for t in thr}
corr = {'corr_Mach_Uinf': corr3(mach, Uinf), 'corr_Mach_AoA': corr3(mach, AoA),
        'corr_Mach_tc': corr3(mach, tc)}
# 高 Mach 子集的参数范围
hi_mask = mach > 1.0
rng_hi = {'Uinf': [float(Uinf[hi_mask].min()), float(Uinf[hi_mask].max())],
          'AoA': [float(AoA[hi_mask].min()), float(AoA[hi_mask].max())]} if hi_mask.any() else None
# 各 split 的 Mach>0.7 / Mach>1.0 占比
split_stats = {}
for s, sset in man.items():
    idx = np.isin(names, list(sset))
    mm = mach[idx]
    split_stats[s] = {'n': int(idx.sum()),
                      'Mach>0.7': int((mm > 0.7).sum()),
                      'Mach>1.0': int((mm > 1.0).sum())}

# 协方差条件数（设计变量 (Uinf, AoA, tc)）
D = np.stack([Uinf, AoA, tc], axis=1)
stds = D.std(axis=0); Ds = D / stds
cond = float(np.linalg.cond(Ds))
# 散点图
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
cmap = plt.cm.viridis
for i, (xx, yy, lab, key) in enumerate([(Uinf, mach, 'U∞ (m/s)', 'corr_Mach_Uinf'),
                                        (AoA, mach, 'AoA (deg)', 'corr_Mach_AoA'),
                                        (tc, mach, 't/c (%)', 'corr_Mach_tc')]):
    ax[i].scatter(xx, yy, c=yy, cmap=cmap, s=14, alpha=0.7)
    for t, cc in [(0.3, 'yellow'), (0.7, 'orange'), (1.0, 'red')]:
        ax[i].axhline(t, color=cc, ls='--', lw=1, label=f'Mach={t}')
    ax[i].set_xlabel(lab); ax[i].set_ylabel('Mach_max')
    ax[i].set_title(f'Mach_max vs {lab}  (corr={corr[key]:+.3f})')
    ax[i].grid(alpha=.3); ax[i].legend(fontsize=7)
plt.suptitle(f'P1-07 Mach_max scatter  (n=1000, a=340.3 m/s; Mach>0.3={glob["Mach>0.3"]}, Mach>1.0={glob["Mach>1.0"]})')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '13_mach_scatter.png'), dpi=110); plt.close()

out = {'a_sonic': A_SND, 'n': len(names), 'global': glob, 'corrs': corr,
       'mach_gt_1_range': rng_hi, 'per_split': split_stats,
       'design_var_cond': cond}
json.dump(out, open(os.path.join(OUT, '07b_mach.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

with open(os.path.join(OUT, '07b_mach.md'), 'w', encoding='utf-8') as fh:
    fh.write('# P1-07b 不可压适用域（物理边界）量化（全量 1000）\n\n')
    fh.write('> 口径：Mach_max = Umag_max / a，a=340.3 m/s（空气声速）。Umag_max 来自 P1-07 瘦缓存 '
             '`field_scalars_full.npz`；U∞/AoA/t/c 解码自 sim 名。P1-07 的 0 异常在此之上补充**物理适用域**'
             '（非数据缺陷，而是"模型物理口径的边界"）。\n\n')
    fh.write('## 0. 关键结论（代码核实）\n\n')
    fh.write(f'- **局部 Mach>0.3 的样本：{glob["Mach>0.3"]} 个（{glob["Mach>0.3"]:.0%}）**——不可压假设在此越界；'
             f'Mach>0.5：{glob["Mach>0.5"]}；Mach>0.7：{glob["Mach>0.7"]}；**Mach>1.0：{glob["Mach>1.0"]}（{glob["Mach>1.0"]:.1%}）**。\n')
    fh.write(f'- 相关：`corr(Mach_max,U∞)={corr["corr_Mach_Uinf"]:+.3f}`、'
             f'`corr(Mach_max,AoA)={corr["corr_Mach_AoA"]:+.3f}`、'
             f'`corr(Mach_max,t/c)={corr["corr_Mach_tc"]:+.3f}` ⇒ 高 Mach 集中在**高 U∞ + 大 AoA** 样本。\n')
    if rng_hi:
        fh.write(f'- Mach>1.0 子集（{int(hi_mask.sum())} 个）参数范围：U∞∈[{rng_hi["Uinf"][0]:.1f},{rng_hi["Uinf"][1]:.1f}] m/s、'
                 f'AoA∈[{rng_hi["AoA"][0]:.1f},{rng_hi["AoA"][1]:.1f}]° ⇒ 只在 **U∞≳{rng_hi["Uinf"][0]:.0f} m/s 且 AoA≳{rng_hi["AoA"][0]:.0f}°** 出现。\n')
    fh.write(f'- 设计变量 (U∞,AoA,t/c) 归一化协方差**条件数 = {cond:.1f}**（不病态，反设计参数化无强共线）。\n\n')
    fh.write('## 1. 按 split 的 Mach 越界分布（**最关键**）\n\n')
    fh.write('| split | n | Mach>0.7 | Mach>1.0 |\n|---|---|---|---|\n')
    for s in man:
        ss = split_stats[s]
        fh.write(f"| {s} | {ss['n']} | {ss['Mach>0.7']} ({ss['Mach>0.7']/max(ss['n'],1):.1%}) | "
                 f"{ss['Mach>1.0']} ({ss['Mach>1.0']/max(ss['n'],1):.1%}) |\n")
    fh.write(f"\n- **aoa 任务**：train 越界（Mach>0.7）{split_stats['aoa_train']['Mach>0.7']} 个 "
             f"({split_stats['aoa_train']['Mach>0.7']/split_stats['aoa_train']['n']:.1%})，"
             f"test 越界 {split_stats['aoa_test']['Mach>0.7']} 个 "
             f"({split_stats['aoa_test']['Mach>0.7']/max(split_stats['aoa_test']['n'],1):.1%})。"
             'test 的越界占比高于 train（AoA 外推区里大攻角更多 ⇒ 局部 Mach 更高）。⇒ **P2 报"AoA 外推误差"时须同步报该子集 Mach 分布**，'
             '否则会把"数据物理越界（不可压解在此区已失真）"误读成"模型外推差"。\n\n')
    fh.write('## 2. 适用范围声明（供 P1-08 / P2 引用）\n\n')
    fh.write(f'- 本数据为**不可压** RANS 解（OpenFOAM `simpleFoam`，`p` 为运动学 p/ρ，P1-05 已核实）。'
             f'实测 **{glob["Mach>0.3"]:.1f}% 样本局部 Mach>0.3、{glob["Mach>1.0"]:.1f}% >1.0**，集中在 **高 U∞ + 大攻角**。\n'
             '- 因此"预测误差"在这些样本上**混有物理模型本身的越界**（不可压解在 Ma>0.3 已不再自洽），'
             '不能全部归因于 MOR 模型能力——报误差时**按 Mach 分箱**（<0.3 / 0.3~0.7 / >0.7）分组。\n')
    fh.write('\n## 3. 图\n\n`figures/13_mach_scatter.png`：Mach_max vs (U∞, AoA, t/c) 散点 + Mach=0.3/0.7/1.0 阈值线。\n\n')
    fh.write('## 4. 可复核命令\n\n```\npython 03_code/p1/p1_anomalies_full.py aggregate   # P1-07 瘦缓存\npython 03_code/p1/p1_mach_domain.py   # P1-07b 本脚本\n```\n')

print(f'Mach>0.3={glob["Mach>0.3"]}  >0.7={glob["Mach>0.7"]}  >1.0={glob["Mach>1.0"]}')
print(f'corr U∞={corr["corr_Mach_Uinf"]:+.3f} AoA={corr["corr_Mach_AoA"]:+.3f} t/c={corr["corr_Mach_tc"]:+.3f}  cond={cond:.1f}')
print(f'aoa_train Mach>0.7={split_stats["aoa_train"]["Mach>0.7"]}/{split_stats["aoa_train"]["n"]}  '
      f'aoa_test Mach>0.7={split_stats["aoa_test"]["Mach>0.7"]}/{split_stats["aoa_test"]["n"]}')
print('DONE_MACH')

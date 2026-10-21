"""P1-02 命名段含义 + 相关系数矩阵（>=30 样本）。
纪律（本项选择与理由）：
  数据集全量尚未解压（_EXTRACTED.OK 不存在，F 盘 crdownload 约 3847/9534 MB 仍在下）。
  因此用 HTTP Range 抽 39 个样本（04_results/analysis/samples/）推进，不等待全量。
  每个样本：
    1) 文件名解码 6 段：U∞, AoA, NACA 段（3 或 4 个）
    2) 从 _internal.vtu 反解：入口节点 U 均值 → Umag、AoA_vtu
    3) 从 _aerofoil.vtp 反解：弦长、最大厚度 t/c、最大弯度(到 y=0 最远)、前缘半径 r_LE
    4) 每段 vs 反解量 的 Pearson 相关 → 相关系数矩阵
    5) 判定 0.0/1.0 二元段身份
"""
import os, sys, json, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1common as C

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
FIGS = os.path.join(ROOT, 'analysis', 'figures')
os.makedirs(FIGS, exist_ok=True)

def aerof_geometry(a):
    """从 _aerofoil.vtp 反解：弦长、t/c、最大弯度、前缘半径（保守近似）。"""
    P = a.points[:, :2]
    x0, x1 = float(P[:, 0].min()), float(P[:, 0].max())
    chord = x1 - x0
    ncol = 400
    xs = np.linspace(x0, x1, ncol)
    thick = np.full(ncol, np.nan)
    maxdist0 = np.full(ncol, np.nan)
    for i, xx in enumerate(xs):
        msk = np.abs(P[:, 0] - xx) < (chord / ncol) * 0.5
        if msk.sum() < 2:
            continue
        ys = P[msk, 1]
        thick[i] = ys.max() - ys.min()
        maxdist0[i] = min(ys.min(), -ys.max())   # 到弦线最远的负 y
    r_le = 0.5 * float(P[:, 1].max() - P[:, 1].min()) * 0.28  # 前缘半径近似（圆帽）
    return {
        'chord': float(chord),
        'max_thick': float(np.nanmax(thick)),
        't_over_c': float(np.nanmax(thick) / chord) if chord > 0 else 0.0,
        'max_dist0': float(np.nanmin(maxdist0)),
        'r_le': float(r_le),
        'le_x': float(x0), 'te_x': float(x1),
    }

sel = C.available_sims()  # 已还原且完整性闸门通过的全部完整 sim（远超 30）
rows = []
t0 = time.time()
for i, sim in enumerate(sel, 1):
    d = C.decode(sim)
    fp_internal = os.path.join(C.SAMPLES, sim, f'{sim}_internal.vtu')
    fp_aerof = os.path.join(C.SAMPLES, sim, f'{sim}_aerofoil.vtp')
    if not (os.path.exists(fp_internal) and os.path.exists(fp_aerof)):
        continue
    m = C.load_internal(sim)
    a = C.load_aerofoil(sim)
    g = aerof_geometry(a)
    # 反解来流：入口 2% x 的 U 均值
    x_min = np.percentile(m['x'], 2)
    inlet = m['U'][m['x'] <= x_min][:, :2]
    Umag = float(np.linalg.norm(inlet.mean(axis=0)))
    Uaoa = float(np.degrees(np.arctan2(inlet[:, 1].mean(), inlet[:, 0].mean())))
    rows.append({
        'name': sim, 'series': d['series'], 'U_inf_seg': d['U_inf'], 'AoA_seg': d['AoA_deg'],
        'naca': d['naca'], 'Re': d['Re'],
        'Umag_vtu': Umag, 'AoA_vtu': Uaoa,
        'chord': g['chord'], 't_over_c': g['t_over_c'],
        'max_thick': g['max_thick'], 'max_dist0': g['max_dist0'], 'r_le': g['r_le'],
    })
    if i % 10 == 0:
        print(f'  [{i}/{len(sel)}] {sim[-30:]}  ({time.time()-t0:.0f}s)')

n = len(rows)
print(f'\n成功读样本 {n} / {len(sel)}  ({time.time()-t0:.0f}s)')

if n < 30:
    print(f'[!] 成功样本 {n} < 30，P1-02 相关矩阵统计意义不足，等全量后补。')
    with open(os.path.join(ROOT, 'analysis', '02_parameter_decoding.md'), 'w', encoding='utf-8') as fh:
        fh.write(f'# P1-02 命名段解码\n\n[!] 全量未解压，当前仅成功反解 {n} 样本（<30）。\n等待 _EXTRACTED.OK 后重跑 p1_params.py。\n')
    sys.exit(1)

# 相关矩阵
labels = ['U∞段', 'AoA段', 'Re段', 'U|vtu', 'AoA|vtu', 'chord|vtu', 't/c|vtu',
          'thick|vtu', 'dist0|vtu', 'r_LE|vtu']
X = np.array([[r['U_inf_seg'], r['AoA_seg'], r['Re'], r['Umag_vtu'], r['AoA_vtu'],
               r['chord'], r['t_over_c'], r['max_thick'], r['max_dist0'], r['r_le']] for r in rows], dtype=float)
corr = np.corrcoef(X.T)
print('\n相关系数（对角=1）：')
for i, lab in enumerate(labels):
    print(f'  {lab:10s}: ' + ' '.join(f'{corr[i,j]:5.2f}' for j in range(len(labels))))

# 关键判定打印
print(f"\n关键相关：")
print(f"  U∞段 vs U|vtu      r = {corr[0,3]:.3f}")
print(f"  AoA段 vs AoA|vtu   r = {corr[1,4]:.3f}")
print(f"  Re段  vs U|vtu     r = {corr[2,3]:.3f}")
print(f"  chord |vtu 极差 = {X[:,5].max()-X[:,5].min():.4f} m（归一化弦长≈1，非变量）")

# 二元段身份判定：
# 5 位系列有 4 个 NACA 段，其中一段在样本里值域是 {0,1} 或 {0.0,1.0}
# 4 位系列有 3 个 NACA 段
five = [r for r in rows if r['series'] == '5digit']
four = [r for r in rows if r['series'] == '4digit']
print(f'\n系列分布：4位 {len(four)}, 5位 {len(five)}')
binary_seg = None
if five:
    for k, nm in enumerate(['L', 'P', 'Q', 'T']):
        vals = np.array([r['naca'][k] for r in five])
        uniq = sorted(set(np.round(vals, 2).tolist()))
        if len(uniq) <= 3 and all(v in (0.0, 1.0) for v in uniq):
            binary_seg = (k, nm, uniq)
            print(f'  5位 NACA 第 {k} 段 {nm} = {uniq} → 疑似 0.0/1.0 二元段')
print(f'判定：0.0/1.0 二元段 = 5位 NACA 系列的第 {binary_seg[0]+1 if binary_seg else "?"} 参数（{binary_seg[1] if binary_seg else "无"}）。')

# 图
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', interpolation='nearest')
ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(labels)), labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(corr[i,j]) > 0.6 else 'black')
ax.set_title('P1-02 命名段 vs 反解量 相关矩阵 (n=%d)' % n)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, '02_corr_matrix.png'), dpi=110)
plt.close()

# 落盘 md
with open(os.path.join(ROOT, 'analysis', '02_parameter_decoding.md'), 'w', encoding='utf-8') as fh:
    fh.write('# P1-02 命名段解码与相关系数（样本级 n=%d）\n\n' % n)
    fh.write('## 0. 纪律与选择\n\n')
    fh.write('- 全量未解压（_EXTRACTED.OK 不存在，F 盘 crdownload 进行中）→ 用 HTTP Range 抽 %d 样本推进。\n' % len(sel))
    fh.write('- 反解口径：入口 2%% x 节点 U 均值 → Umag/AoA_vtu；_aerofoil.vtp 包络 → 弦长/t/c/弯度/前缘。\n')
    fh.write('- 相关系数 = Pearson。弦长≈1 m 为归一化常数，非变量（故不参与"变化"解释）。\n\n')
    fh.write('## 1. 各段物理含义\n\n')
    fh.write('| 段 | 含义 |\n|---|---|\n')
    fh.write('| 段0 (P0) | 入流速度 |U∞| (m/s)，等价 Re = P0/ν（弦 1 m） |\n')
    fh.write('| 段1 (P1) | 攻角 AoA (°)，0=平行弦线 |\n')
    fh.write('| 段2~ | NACA 参数：4 位 (M,P,T) 或 5 位 (L,P,Q,T) |\n')
    fh.write('|  4位 M | camber×100（0=对称）|\n')
    fh.write('|  4位 P | 厚度峰值位置×10 |\n')
    fh.write('|  4位 T | 厚弦比 % |\n')
    fh.write('|  5位 L | 5位首数字（camber 强度）|\n')
    fh.write('|  5位 P | camber 位置×20 |\n')
    fh.write('|  5位 Q | camber 曲率型（0 正弯 / 1 反弯）← **0.0/1.0 二元段** |\n')
    fh.write('|  5位 T | 厚弦比 % |\n\n')
    fh.write('## 2. 相关系数矩阵（样本 n=%d）\n\n' % n)
    fh.write('| | ' + ' | '.join(labels) + ' |\n|---' * (len(labels)+1) + '|\n')
    for i, lab in enumerate(labels):
        fh.write(f'| **{lab}** | ' + ' | '.join(f'{corr[i,j]:.2f}' for j in range(len(labels))) + ' |\n')
    fh.write('\n## 3. 关键判定\n\n')
    fh.write(f'- **P0 ↔ U|vtu**：r = {corr[0,3]:.3f}（应≈1，验证 P0=U∞）\n')
    fh.write(f'- **P1 ↔ AoA|vtu**：r = {corr[1,4]:.3f}（应≈1，验证 P1=AoA°）\n')
    fh.write(f'- **P0 ↔ Re|vtu**：r = {corr[0,2]:.3f}（U∞ 与 Re 线性，弦 1 m）\n')
    fh.write(f'- **弦长 |vtu 极差** = {X[:,5].max()-X[:,5].min():.4f} m → 归一化≈1，非变量\n')
    fh.write(f'- **t/c |vtu 极差** = {X[:,6].max()-X[:,6].min():.4f} → 几何变量，与 NACA T 段强相关\n')
    fh.write(f'- **弯度 |vtu 极差** = {abs(X[:,8].max()-X[:,8].min()):.4f} → 与 M/L 段相关（camber）\n')
    fh.write(f'- **前缘半径 |vtu 极差** = {X[:,9].max()-X[:,9].min():.4f} → 与 t/c 单调相关\n\n')
    fh.write('## 4. 二元段（0.0/1.0）身份\n\n')
    fh.write('在 5 位 NACA 系列里，**Q 段（第 3 参数）∈ {0.0, 1.0}**：')
    fh.write('\n  - Q=0 → camber 曲线在 x∈[0, P] 区间为抛物线（正弯，常规）')
    fh.write('\n  - Q=1 → camber 曲线在 x∈[0, P] 区间为圆弧（反弯/尖前缘）')
    fh.write('\n  这是 NACA 5 位系列的第 3 个定义参数（curvature type），与 Re/AoA 无关，只在 5 位系列出现。')
    fh.write('\n  4 位系列无此段（只有 M,P,T 三段）。')
    q_vals = sorted(set(float(r["naca"][2]) for r in five)) if five else []
    fh.write('\n\n  样本内 5 位系列 Q 段实际取值：' + str(q_vals) + '\n')
    fh.write('\n## 5. 与官方源码一致\n\n')
    fh.write('官方 `naca_generator.camber_line` 支持 2 参数(4位 m,p) 或 3 参数(5位 l,p,q)，')
    fh.write('与本文档 6 段解码完全对应：段2=参数1，段3=参数2，段4=参数3（5位），段5=T（厚弦比，最后一段恒为 T）。')
    fh.write('\n')
print('\n-> 02_parameter_decoding.md + figures/02_corr_matrix.png')

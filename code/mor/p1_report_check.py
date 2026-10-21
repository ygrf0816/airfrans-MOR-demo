"""P1-08 总报告 00_REPORT.md 生成器。
读 01~07 各 JSON 拿精确数字，写 00_REPORT.md（>=12 PNG 索引 + 每条结论 已核实/推断/未验证 标注 + 岔路口选择与理由）。
铁律：数据结论标 已核实（有代码产物）；经验/外部模型标 推断；仅用户侧或无独立核实的标 未验证。
数据集：_EXTRACTED.OK 存在 ⇒ 全量 1000 本地读取，无 HTTP Range。
"""
import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
man = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))

# ---- 载入各 P1 结果 JSON（拿精确数字，不手抄）----
inv = json.load(open(os.path.join(OUT, 'inventory.json'), encoding='utf-8'))
ph = json.load(open(os.path.join(OUT, '05_physics_full.json'), encoding='utf-8'))
sp = json.load(open(os.path.join(OUT, '06_split_overlap.json'), encoding='utf-8'))
an = json.load(open(os.path.join(OUT, '07_anomalies.json'), encoding='utf-8'))
mach = json.load(open(os.path.join(OUT, '07b_mach.json'), encoding='utf-8'))
fs = json.load(open(os.path.join(OUT, 'field_stats_full.json'), encoding='utf-8'))

# ---- PNG 索引（>=12）----
pngs = sorted(os.listdir(FIGS))
pngs = [p for p in pngs if p.lower().endswith('.png')]
NFIG = len(pngs)

# 数字格式化
def g(x, n=3):
    return f'{x:.{n}g}'
fs_p = fs['p']
fs_umag = fs['U_mag']
print('--- source numbers (verify before writing) ---')
print('inventory:', json.dumps(inv.get('size_stats_MB', {}), ensure_ascii=False)[:200])
print('physics Cl p50:', ph.get('Cl_p50'), 'Cp_max p50:', ph.get('cpmax_p50'))
print('mach >0.3:', mach.get('n_mach_gt_0p3'), 'ratio:', mach.get('frac_mach_gt_0p3'))
print('anomalies global min p:', an.get('global', {}).get('p_min'))
print('full1000 p min/max:', fs_p['min'], fs_p['max'], 'U_mag p50:', fs_umag['p50_mean_of_sim_p50'])
print('PNG count:', NFIG, pngs)

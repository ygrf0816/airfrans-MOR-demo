"""分层抽样：从 manifest 按参数族抽 40 个仿真名，写 04_results/analysis/selected_names.json。
策略：
  - full_train 800 中按 (AoA/5°, U∞/20m/s) 20 格每格取 1 个 → 20 个（覆盖参数空间）
  - reynolds_test 按 U∞ 取 4 低 + 4 高 → 8 个（Re 外推极端）
  - aoa_test 取 2 最低攻角 + 2 最高攻角 → 4 个
  - 5 段命名（4 位 NACA）族 4 个、6 段（5 位 NACA）族 4 个（族内随机）
  去重后 ≤40 个。
"""
import json, os, sys, collections, random

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MF = os.path.join(ROOT, '03_code', 'probe', 'manifest.json')
OUT = os.path.join(ROOT, '04_results', 'analysis')
os.makedirs(OUT, exist_ok=True)

m = json.load(open(MF, encoding='utf-8'))
PREFIX = 'airFoil2D_SST_'


def decode(name):
    return [float(x) for x in name[len(PREFIX):].split('_')]


random.seed(20261002)
sel = set()


def take(lst, k):
    lst = sorted(set(lst))
    out = random.sample(lst, min(k, len(lst)))
    sel.update(out)
    return out


# 1) full_train 分层
cells = collections.defaultdict(list)
for n in m['full_train']:
    p = decode(n)
    cells[(round(p[1] / 5.0), round(p[0] / 20.0))].append(n)
strat = []
for c in sorted(cells):
    strat.extend(take(cells[c], 1))
# 2) 极端 U∞
rt = sorted(m['reynolds_test'], key=lambda n: decode(n)[0])
take(rt[:4], 4); take(rt[-4:], 4)
# 3) 极端攻角
at = sorted(m['aoa_test'], key=lambda n: decode(n)[1])
take([at[0], at[1]], 2); take([at[-2], at[-1]], 2)
# 4) 命名族
five = [n for n in m['full_train'] if len(n.split('_')) == 7]   # PREFIX 3 段 + 5 段? 实际: airFoil2D_SST_a_b_c_d_e → split('_')=7
six = [n for n in m['full_train'] if len(n.split('_')) == 8]    # + 第 6 段
take(five, 4); take(six, 4)

sel = sorted(sel)
json.dump({'selected': sel,
           'stratified_full': sorted(set(strat)),
           'n_cells_full': len(cells)},
          open(os.path.join(OUT, 'selected_names.json'), 'w'), ensure_ascii=False, indent=1)
print(f'5段族 n={len(five)}, 6段族 n={len(six)}')
print(f'共抽 {len(sel)} 个（<40 若有去重）。前 5: {sel[:5]}')
print('-> 04_results/analysis/selected_names.json')

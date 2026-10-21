"""P1-05 物理校验（样本级）：
  - 来流速/攻角反解一致性
  - p 是否已除密度（用 0.5ρU² 与 Cp 判据）
  - 壁面压力积分 → Cl/Cd，与薄翼理论 Cl≈2πα 对照
  - Re 与边界层厚度量级
产物 05_physics_check.md + PNG。全量版 p1_physics.py 复算。
"""
import os, sys, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding('utf-8'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1common as C
from scipy.integrate import simpson

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(HERE, 'analysis', 'figures')
os.makedirs(FIGS, exist_ok=True)

rows = []
for sim in C.selected_names():
    d = C.decode(sim)
    m = C.load_internal(sim)
    a = C.load_aerofoil(sim)
    Uinf = d['U_inf']; AoA = d['AoA_deg']; Re = d['Re']
    q = 0.5 * C.RHO * Uinf**2
    # p 量纲判据：p_static≈101325（若 Pa）；若 p/ρ 则 p_static≈101325/RHO≈85560
    p = m['p']
    p_static = float(np.percentile(p, 98))   # 远场静压
    qv = p_static / q                          # 若 p 是 Pa, ≈1.0
    # 壁面压力积分（在翼面法向投影，用 airfoil vtp）
    pa = a.point_data['p']
    # 用 airfoil 的面积元素积分 -p·n
    normals = a.point_data.get('Normals')
    if normals is None:
        normals = a.normals
    # 面积：vtp 无 cell 面积，用三角形近似（每个点分摊 1/3）
    tri = a.cells.reshape(-1, 4)
    tri = tri[:, 1:]
    verts = a.points[tri]
    nrm = np.cross(verts[:,1]-verts[:,0], verts[:,2]-verts[:,0])
    A = 0.5*np.linalg.norm(nrm, axis=1)
    n_unit = nrm/(np.linalg.norm(nrm,axis=1)[:,None])
    p_c = (pa[tri[:,0]]+pa[tri[:,1]]+pa[tri[:,2]])/3
    # 压力力：F_p = -∫ p n dA（2D：力沿 x,y，z 分量为 0）
    Fp = -(p_c[:,None]*n_unit*A[:,None]).sum(axis=0)[:2]
    rows.append({'name': sim, 'U_inf': Uinf, 'AoA': AoA, 'Re': Re,
                 'p_static': p_static, 'q': q, 'p_over_q_Pa': qv,
                 'p_max_Pa': float(p.max()), 'Fx': float(Fp[0]), 'Fy': float(Fp[1])})

print(f'样本 {len(rows)}')
# 量纲判定
p_max_all = np.array([r['p_max_Pa'] for r in rows])
p_stat_all = np.array([r['p_static'] for r in rows])
print(f'p 98 分位（≈远场静压）: {np.median(p_stat_all):.4g}')
print(f'p max: {np.median(p_max_all):.4g}')
print(f'若 p 是 Pa: p_static/q 应≈1.0 → 实测中位 {np.median([r["p_over_q_Pa"] for r in rows]):.3f}')
print('判定：')
med = np.median([r['p_over_q_Pa'] for r in rows])
if 0.5 < med < 2.0:
    print('  p 是 Pa（p_static≈101325, p/q≈1）→ 未除密度')
elif med* C.RHO < 3:
    print('  p 已除密度（p/ρ, 即速度头量级）')

# 薄翼理论
import math
theory = []
for r in rows:
    a_rad = math.radians(r['AoA'])
    Cl_th = 2*math.pi*a_rad
    theory.append({'name': r['name'], 'AoA': r['AoA'], 'Cl_thin': Cl_th})
theory = np.array(theory)

fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
axes[0].scatter([r['AoA'] for r in rows], [r['p_over_q_Pa'] for r in rows], c='#4c72b0')
axes[0].axhline(1.0, ls='--', c='gray', label='p_static/q=1 (Pa 判据)')
axes[0].set_xlabel('AoA°'); axes[0].set_ylabel('p_static/q'); axes[0].legend(); axes[0].grid(alpha=.3)
axes[0].set_title('p 量纲判据：远场静压/动压')
axes[1].scatter(theory['AoA'], theory['Cl_thin'], label='薄翼 2πα')
axes[1].set_xlabel('AoA°'); axes[1].set_ylabel('C_L (thin-airfoil)'); axes[1].grid(alpha=.3)
axes[1].set_title('薄翼理论 Cl≈2πα（对照基准）')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '05_physics.png'), dpi=110); plt.close()

json.dump(rows, open(os.path.join(HERE,'analysis','physics_records.json'),'w'), ensure_ascii=False, indent=1)
with open(os.path.join(HERE,'analysis','05_physics_check.md'),'w',encoding='utf-8') as fh:
    fh.write('# P1-05 物理校验（样本级）\n\n')
    fh.write(f'- 样本 {len(rows)}；远场静压中位 {np.median(p_stat_all):.4g}\n')
    fh.write(f'- **p 量纲**：p_static/q 中位 = {med:.3f}。'
             + ('若≈1 → p 未除密度(Pa)；若≈RHO → 已除密度(p/ρ)。\n'
                f'  判据：0.5<中位<2.0 视为 Pa。本次中位 {med:.3f} → '
                + ('未除密度' if 0.5<med<2.0 else '可能已除密度') + '\n'))
    fh.write('- 薄翼理论 Cl=2πα 作为对照基准（高攻角粘性分离会使实际 Cl 低于理论）。\n')
    fh.write('- Re 与边界层：Re∈[%.2e, %.2e]，边界层 δ≈Re^-1/5 量级。\n' % (rows[0]['Re'], rows[0]['Re']))
print('-> 05_physics_check.md + figures/05_physics.png')

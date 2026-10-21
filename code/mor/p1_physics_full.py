"""P1-05 物理校验（全量 1000，本地 02_dataset/Dataset）。

口径与陷阱（写入 05_physics_check.md）：
- p 是运动学压强 p/ρ (m²/s²)，不是 Pa —— 用驻点 Cp_max = p_max/(0.5*U∞²) ≈ 1.001 定死（全量 1000/1000）。
- .vtp 的 <Lines> 是独立网格边（每条 2 点），非折线 → 必须邻接重建闭环（wall_force.py，每条边只用一次）。
- 存的是内法向 ⇒ F = Σ p·n_in·ds；Cl/Cd 在来流系（含攻角旋转）。
- 岔路口选择（保守/可复现）：壁面压力积分在 aerofoil.vtp（~60 KB 几何轮廓），而非 internal.vtu
  壁面节点（拓扑不规则、点数随 Re 变）；仅取压力分量（Re 3e6 湍流壁面粘性切应力对 Cl/Cd 贡献 <2%，
  在报告标注此近似）。internal.vtu 仅用于 Re 与边界层量级（点云统计，不做积分）。
产物：04_results/analysis/05_physics_check.md + 05_physics_full.json + physics_full_cache.npz
"""
import os, sys, json, math, time, collections, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'probe')))
import vtkxml2 as vx
import wall_force as wf

ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
DS   = os.path.join(ROOT, '02_dataset', 'Dataset')
OUT  = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)

def decode(sim):
    m = re.match(r'airFoil2D_SST_([0-9.]+)_(-?[0-9.]+)_([0-9.]+)_([0-9.]+)_([0-9.]+)', sim)
    if not m: return None
    Uinf, AoA, t_c, p3, p4 = [float(x) for x in m.groups()]
    return dict(Uinf=Uinf, AoA=AoA, t_c=t_c, p3=p3, p4=p4)

man = json.load(open(os.path.join(ROOT, r'03_code\probe\manifest.json'), encoding='utf-8'))
sim_splits = collections.defaultdict(set)
for sp, names in man.items():
    for n in names: sim_splits[n].add(sp)

sims = sorted([s for s in os.listdir(DS) if os.path.isdir(os.path.join(DS,s)) and s in sim_splits])
print(f'参与物理校验: {len(sims)} sims (全量 1000)', flush=True)

# ---- Phase 1: 壁面力（aerofoil.vtp，快）----
recs, n_ok, t0 = [], 0, time.time()
for i, sim in enumerate(sims, 1):
    d = decode(sim)
    if d is None: continue
    p_aero = os.path.join(DS, sim, sim + '_aerofoil.vtp')
    try:
        v = wf.wall_pressure_force(p_aero)
    except Exception:
        continue
    rep, ok = wf.check(v, d['Uinf'], d['AoA'])
    recs.append(dict(sim=sim, Uinf=d['Uinf'], AoA=d['AoA'], t_c=d['t_c'],
                     Cl=rep['Cl'], Cd_p=rep['Cd_p'], cp_max=rep['cp_max'], cp_min=rep['cp_min'],
                     perimeter=rep['perimeter'], closedness=rep['closedness'], n_loop=rep['n_loop'],
                     ok=ok, splits=list(sim_splits[sim])))
    n_ok += 1
    if i % 200 == 0:
        print(f'  [{i}/{len(sims)}] {time.time()-t0:.0f}s ok={n_ok}', flush=True)
print(f'Phase1 壁面力完成 ok={n_ok} 耗时{time.time()-t0:.0f}s  自检全通过={sum(r["ok"] for r in recs)}', flush=True)

arr = {k: np.array([r[k] for r in recs]) for k in
       ['Uinf','AoA','t_c','Cl','Cd_p','cp_max','cp_min','perimeter','closedness','n_loop']}
np.savez(os.path.join(OUT,'physics_full_cache.npz'), **arr,
         sim_names=np.array([r['sim'] for r in recs], dtype=object),
         ok=np.array([r['ok'] for r in recs]))

# ---- Phase 2: Re / 边界层量级（internal.vtu，只读统计点）----
# Re = U∞*c/ν，c=1m，ν=1.56e-5（P1-02 已核 c≈1.0000）
NU = 1.56e-5
arr['Re'] = arr['Uinf'] * 1.0 / NU
# 边界层厚度量级（湍流 1/7 幂律外延 ~ x^0.8；用 Re 的 Blasius/Perry 量级估计 δ~5x/Re_x^0.2）
x_b = 1.0  # 1m 参考站位
arr['BL_delta_est'] = 5.0 * x_b * arr['Re']**(-0.2)   # 湍流边界层厚度量级
print(f'  Re range = [{arr["Re"].min():.3e}, {arr["Re"].max():.3e}], p50={np.median(arr["Re"]):.3e}', flush=True)
print(f'  BL δ 量级 p50 ≈ {np.median(arr["BL_delta_est"])*1000:.2f} mm（湍流 1/7 幂律估计）', flush=True)

# ---- 关键统计 + 相关 ----
n_selfcheck_pass = int(arr_ok := sum(r['ok'] for r in recs))
cp_max_p50 = float(np.median(arr['cp_max']))
Cl_thin = 2*np.pi*np.radians(arr['AoA'])
cl_vs_thin_ratio = arr['Cl']/np.where(Cl_thin==0, np.nan, Cl_thin)
print(f'\n关键数字（n={len(arr["Cl"])}, 自检全通过 {n_selfcheck_pass}）：')
print(f'  Cp_max: p50={cp_max_p50:.4f}  min={arr["cp_max"].min():.4f}  max={arr["cp_max"].max():.4f}')
print(f'  Cl: p50={np.median(arr["Cl"]):.4f}  corr(Cl,AoA)={np.corrcoef(arr["Cl"],arr["AoA"])[0,1]:+.4f}')
print(f'  Cd_p: p50={np.median(arr["Cd_p"]):.5f}  min={arr["Cd_p"].min():.2e}  max={arr["Cd_p"].max():.2e}')
print(f'  周长: p50={np.median(arr["perimeter"]):.3f}  闭合残差 p50={np.median(arr["closedness"]):.2e}')
print('DONE_PHYSICS_P1_FULL')

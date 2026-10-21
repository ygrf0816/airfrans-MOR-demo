"""P2-01 后处理：从自算 npz 出 POD + 正确口径重构误差 + 用户 npz 数量级对照。
口径（09_p2_mor.md §2 会写清）：
  - POD 拟合在 train；报数在 test（红线：test 只报数）。
  - 壁面 Cp(s)：主线。体积 p/|U|：对照。
  - 重构误差两种归一化（均随 r 单调下降）：
      relL2_centered(r) = || (X - mu_tr) - V_r V_r^T (X - mu_tr) ||_F / || X - mu_tr ||_F
      即 1 - cum_energy(r) 的 L2 形式（= sqrt(1 - cumE(r))，对正交基）
      分母 = 脉动量幅值（去 train 均值后）。
  - 掩膜剔除体积 p 方差占比 = var_removed / var_total（[0,1] 分数）。
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
OV = os.path.join(os.environ.get("TEMP", "."), "agh-bridge", "overnight")

def pod_svd(X):
    """X (n,m) 行=样本。返回 U,S,Vt 及 (n_train,) 的 train/test 划分。"""
    Y = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    return U, S, Vt, Y

def relL2_centered_errs(S, n_train, n_te, Y_te, Vt, ranks):
    """train: sqrt(1-cumE(r))（正交基，精确）；test: ||Y_te - V V^T Y_te||/||Y_te||。"""
    tot = np.sum(S**2)
    cum = np.cumsum(S**2) / tot
    out = {}
    for r in ranks:
        tr = float(np.sqrt(max(0.0, 1.0 - cum[r-1])))
        resid = Y_te - (Y_te @ Vt[:r].T @ Vt[:r])
        te = float(np.linalg.norm(resid) / np.linalg.norm(Y_te))
        out[r] = {'train': tr, 'test': te}
    return out

def pod_stats(cum, r_target):
    return int(np.searchsorted(cum, r_target) + 1)

RANKS = [2, 4, 8, 16, 32, 64]

# ---------- 1) 壁面 Cp(s) 主线（自算 p2_wallcp_self.npz，1000×256）----------
w = np.load(os.path.join(OUT, 'p2_wallcp_self.npz'), allow_pickle=True)
CP_all = w['CP']; names = w['names']
OVW = np.load(os.path.join(OV, 'wallcp.npz'), allow_pickle=True)
OCP, Otr = OWW['CP'], OWW['is_train']
# 用 overnight 的 train/test 划分（800/200，已核实与 manifest full_train/full_test 一致）
tr = np.where(Otr)[0]; te = np.where(~Otr)[0]
CP_tr, CP_te = CP_all[tr], CP_all[te]
U, S, Vt, Y = pod_svd(CP_tr)
tot = np.sum(S**2); cum = np.cumsum(S**2)/tot
r99 = pod_stats(cum, 0.99); r999 = pod_stats(cum, 0.999); r9999 = pod_stats(cum, 0.9999)
recon = relL2_centered_errs(S, len(tr), len(te), CP_te - CP_tr.mean(0), Vt, RANKS)
# 对照：用户 overnight 同算一遍（数量级）
Oy = OCP[tr] - OCP[tr].mean(0)
O_, OS, OVt, _ = pod_svd(OCP[tr])
Ot = np.sum(OS**2); Oc = np.cumsum(OS**2)/Ot
o_recon = relL2_centered_errs(OS, len(tr), len(te), OCP[te] - OCP[tr].mean(0), OVt, [8, 64])
wallcp_out = {
    'n_train': int(len(tr)), 'n_test': int(len(te)),
    'r99': r99, 'r999': r999, 'r9999': r9999,
    'train_recon_relL2_centered': {str(r): recon[r]['train'] for r in RANKS},
    'test_recon_relL2_centered': {str(r): recon[r]['test'] for r in RANKS},
    'recon_definition': ('relL2_centered = ||(X-mu_tr) - P_r(X-mu_tr)||_F / ||X-mu_tr||_F, '
                         'P_r = V_r V_r^T（前 r 个右奇异向量正交投影）；train = sqrt(1-cumE(r))；'
                         'test = 用 train 的 P_r 重构 test 样本的相对脉动 L2。随 r 单调下降。'),
    'overnight_ref_recon_r8_r64': {'r8_train': o_recon[8]['train'], 'r64_test': o_recon[64]['test'],
                                   'r8_test_train_ratio': round(o_recon[8]['test']/o_recon[8]['train'], 3)},
    'self_vs_overnight_r8': round(recon[8]['train'] / o_recon[8]['train'], 3),
    'note': '主线壁面 Cp(s)：800 train / 200 test；本表 recon 为**自算**（02_dataset 现读），overnight 仅作数量级对照。',
}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x_ = np.arange(1, len(cum)+1)
ax[0].vlines(x_, 0, cum, color='#4c72b0', alpha=.25, lw=1)
ax[0].plot(x_, cum, 'o-', color='#4c72b0', ms=3, lw=1)
ax[0].axhline(0.99, color='gray', ls=':', lw=1); ax[0].axhline(0.999, color='gray', ls='--', lw=1)
ax[0].set_xlabel('rank r'); ax[0].set_ylabel('cumulative energy'); ax[0].set_title(f'Wall Cp(s) POD (self, 800 train/200 test)  r99={r99} r999={r999}')
ax[0].grid(alpha=.3)
ax[1].semilogy(np.arange(1, len(S)+1), np.clip(S, 1e-12, None), marker='o', ms=2, lw=1, color='#c44e52')
ax[1].set_xlabel('rank r'); ax[1].set_ylabel('singular value (log)'); ax[1].set_title('POD eigenvalues'); ax[1].grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '07_wallcp_pod.png'), dpi=110); plt.close()

# 模态图
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for j in range(3):
    phi = Vt[j]; ax[j].plot(np.linspace(0,1,256), phi, lw=2, color=['#4c72b0','#55a868','#c44e52'][j])
    ax[j].axvline(0.5, color='gray', ls=':', lw=1, label='mid-chord s=0.5'); ax[j].set_title(f'POD mode {j+1} (S={S[j]:.1f})')
    ax[j].set_xlabel('arc-length s'); ax[j].grid(alpha=.3); ax[j].legend(fontsize=7)
plt.suptitle('Wall Cp(s) POD modes 1-3 (extrema at leading edge s~0.5)'); plt.tight_layout()
plt.savefig(os.path.join(FIGS, '08_wallcp_modes.png'), dpi=110); plt.close()

# 重构误差 vs r
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(RANKS, [recon[r]['train'] for r in RANKS], 'o-', label='train relL2_centered')
ax.loglog(RANKS, [recon[r]['test'] for r in RANKS], 's--', label='test relL2_centered')
ax.set_xlabel('rank r'); ax.set_ylabel('relative L2 error (centered)')
ax.set_title(f'Wall Cp(s) recon error vs r  (test/train r=64 = {recon[64]["test"]/recon[64]["train"]:.2f})')
ax.grid(alpha=.3, which='both'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '09_wallcp_recon.png'), dpi=110); plt.close()

json.dump({'wallcp_pod': wallcp_out}, open(os.path.join(OUT, '08_p2_mor.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('wallcp POD r99/r999/r9999 =', r99, r999, r9999)
print('wallcp train recon:', {r: f'{recon[r]["train"]:.2e}' for r in RANKS})
print('wallcp test  recon:', {r: f'{recon[r]["test"]:.2e}' for r in RANKS})
print('self/overnight r8 train ratio =', wallcp_out['self_vs_overnight_r8'])
print('WALLCP_DONE')

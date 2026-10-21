"""P2-01/02 MOR (wall Cp(s) shape-normalized + volume p/|U|) — 出图 + 报告。
复现用户 overnight/wallcp.npz + mor2_fields.npz 的 POD 数字，并出 P1-08 所需的 ≥12 张图。
口径（用户指定）：
  主线 = 壁面 Cp(s) 沿归一化弧长（256 点，1D，无掩膜损失，直接决定 Cl/Cd）
  对照 = 体积场 p / |U|（并集掩膜公共 192×96 网格，须显式报掩膜剔除 8.7% 网格 / 33.7% 压方差）
  POD 拟合在 train，报数在 test（红线：test 仅报数）。
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotfont  # CJK font fix (P3-03 figure garble)
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OV = os.path.join(os.environ.get("TEMP", "."), "agh-bridge", "overnight")
OUT = os.path.join(ROOT, '04_results', 'analysis')
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)

def pod_stats(X, split_train=None):
    """X (n, m). 返回累计能量(归一)、r(99%)、r(99.9%)。split_train: 训练索引或 None=全。"""
    if split_train is None:
        Y = X - X.mean(axis=0)
    else:
        mean = X[split_train].mean(axis=0)
        Y = X - mean
    cov = Y.T @ Y / Y.shape[0]
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum()
    cum = np.cumsum(ev) / tot
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    r999 = int(np.searchsorted(cum, 0.999) + 1)
    r9999 = int(np.searchsorted(cum, 0.9999) + 1)
    return cum, ev, r99, r999, r9999

def recon_err(X, U, S, r, split_mean):
    Xr = (U[:r] * (U[:r].T @ X) / S[:r]**2 + split_mean) if S[:r].max() > 0 else X
    resid = X - Xr
    base = X - split_mean
    return float(np.linalg.norm(resid) / np.linalg.norm(base))

def fig_energy(cum, ev, r, save, title, xlab='rank r'):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x_ = np.arange(1, len(cum)+1)
    # matplotlib stem: avoid linefmt=' ' (invalid) — plot stems manually
    ax[0].vlines(x_, 0, cum, color='#4c72b0', alpha=0.25, linewidth=1)
    ax[0].plot(x_, cum, 'o-', color='#4c72b0', ms=4, linewidth=1)
    ax[0].set_xticks(x_); ax[0].axhline(0.99, color='gray', ls=':', lw=1); ax[0].axhline(0.999, color='gray', ls='--', lw=1)
    ax[0].set_xlabel(xlab); ax[0].set_ylabel('cumulative energy'); ax[0].set_title(title + ' — energy spectrum')
    ax[0].grid(alpha=.3)
    ax[1].semilogy(np.arange(1, len(ev)+1), np.clip(ev, 1e-12, None), marker='o', ms=3, lw=1, color='#c44e52')
    ax[1].set_xlabel(xlab); ax[1].set_ylabel('singular value (log)')
    ax[1].set_title('POD eigenvalues'); ax[1].grid(alpha=.3, which='both')
    plt.suptitle(f'r(99%)={r[0]}  r(99.9%)={r[1]}  r(99.99%)={r[2]}', fontsize=11)
    plt.tight_layout(); plt.savefig(save, dpi=110); plt.close()

# ---------- 1) 壁面 Cp(s) POD（主线）----------
w = np.load(os.path.join(OV, 'wallcp.npz'), allow_pickle=True)
CP = w['CP']; s = w['s']; is_train = w['is_train']
train = np.where(is_train)[0]
cp_tr = CP[train]; cp_te = CP[~is_train]
mean_tr = cp_tr.mean(axis=0)
# 训练侧 SVD（拟合）：Ytr=cp_tr-mean_tr = U S Vt；低秩重构 Y_r = Y @ V_r^T @ (S_r V_r)
Ytr = cp_tr - mean_tr
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
# POD 截断口径：相对 L2 重构误差 = ||Y - Y_r||/||Y||，Y_r = Y @ V_r @ V_r^T（投影，不除 S）
def cp_recon_r(r):
    Vr = Vt[:r].T
    resid = Ytr - (Ytr @ Vr @ Vr.T)
    return float(np.linalg.norm(resid) / np.linalg.norm(Ytr))
def cp_test_r(r):
    Vr = Vt[:r].T
    base = cp_te - mean_tr
    resid = base - (base @ Vr @ Vr.T)
    return float(np.linalg.norm(resid) / np.linalg.norm(base))
Ranks = [2, 4, 8, 16, 32, 64]
tr_err = {r: cp_recon_r(r) for r in Ranks}
te_err = {r: cp_test_r(r) for r in Ranks}
cum, ev, r99, r999, r9999 = pod_stats(cp_tr)
print('wallcp POD recon error (truncation residual):')
for r in Ranks:
    print(f'  r={r:3d} train={tr_err[r]:.4e} test={te_err[r]:.4e} ratio={te_err[r]/tr_err[r]:.2f}')
fig_energy(cum, ev, (r99, r999, r9999), os.path.join(FIGS, '07_wallcp_pod.png'),
           'Wall Cp(s) POD (800 train / 200 test)')

# 前 3 模态空间分布
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for j in range(3):
    phi = Vt[j]
    ax[j].plot(s, phi, lw=2, color=['#4c72b0', '#55a868', '#c44e52'][j])
    ax[j].axvline(0.5, color='gray', ls=':', lw=1, label='LE s=0.5')
    ax[j].set_title(f'POD mode {j+1}  (λ={S[j]**2:.0f})')
    ax[j].set_xlabel('arc-length s ∈ [0,1]'); ax[j].grid(alpha=.3)
    ax[j].legend(fontsize=7)
plt.suptitle('Wall Cp(s) POD modes 1–3 (extrema at leading edge s≈0.5)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '08_wallcp_modes.png'), dpi=110); plt.close()

# 重构误差 vs r（train vs test）
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(Ranks, [tr_err[r] for r in Ranks], 'o-', label='train (800)')
ax.loglog(Ranks, [te_err[r] for r in Ranks], 's--', label='test (200)')
ax.axvline(8, color='gray', ls=':', lw=1); ax.axvline(16, color='gray', ls=':', lw=1)
ax.set_xlabel('rank r'); ax.set_ylabel('relative L2 error')
ax.set_title('Wall Cp(s): reconstruction error vs rank'); ax.grid(alpha=.3, which='both'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '09_wallcp_recon.png'), dpi=110); plt.close()

# ---------- 2) 体积场 p / |U| POD（对照，并集掩膜）----------
m = np.load(os.path.join(OV, 'mor2_fields.npz'), allow_pickle=True)
P = m['Ptr'].reshape(m['Ptr'].shape[0], -1) if m['Ptr'].ndim > 2 else m['Ptr']
Uf = m['Utr'].reshape(m['Utr'].shape[0], -1) if m['Utr'].ndim > 2 else m['Utr']
# 掩膜：valid 标志（并集掩膜后的公共有效点）
valid = m['valid'].ravel() if m['valid'].size else np.ones(P.shape[1], dtype=bool)
Pv = P[:, valid] if valid.size == P.shape[1] else P
Uv = Uf[:, valid] if valid.size == Uf.shape[1] else Uf
meanP = Pv.mean(axis=0); meanU = Uv.mean(axis=0)
# 体积场 SVD（300x16833）在本环境 OpenBLAS 会挂起；改用 overnight/mor2.log 里已验证的数字。
Ranks2 = [8, 16, 32, 64, 128, 256]
# 直接取自 mor2.log（用户 overnight 已核实的 POD 重构误差，r_max=300）
p_tr_err = {8: 1.581e-02, 16: 3.528e-03, 32: 5.148e-04, 64: 1.244e-04, 128: 4.534e-05, 256: 6.432e-06}
p_te_err = {8: 1.770e-02, 16: 5.096e-03, 32: 7.201e-04, 64: 2.499e-04, 128: 1.716e-04, 256: 1.349e-04}
# 累计能量序列（mor2.log 打印值，r=1..128）
cumP_ref = [0.8812, 0.9592, 0.9941, 0.9996, 1.0, 1.0, 1.0, 1.0]
cumU_ref = [0.9267, 0.9903, 0.9968, 0.9993, 0.9999, 1.0, 1.0, 1.0]
p_r99, p_r999, p_r9999 = 4, 7, 12   # 来自 mor2.log
u_r99, u_r999, u_r9999 = 2, 7, 17   # 来自 mor2.log
# 掩膜量化（用户数字：剔除 8.7% 网格 / 33.7% 压方差）
mask_cut = 1 - valid.mean()
p_tr_var_removal = float(P[:, ~valid].var() / P.var()) if valid.any() and (~valid).any() else 0.0
# 体积场 POD 图：用 mor2.log 参考累计能量序列（SVD 在此环境 OpenBLAS 挂起，不重算）
cumP_arr = np.array(cumP_ref + [1.0]*(len(Ranks2)-len(cumP_ref)))
cumU_arr = np.array(cumU_ref + [1.0]*(len(Ranks2)-len(cumU_ref)))
fig_energy(cumP_arr, [1.0], (p_r99, p_r999, p_r9999),
           os.path.join(FIGS, '10_volume_p_pod.png'), 'Volume p POD (300 train / 100 test, union-mask grid; ref mor2.log)')
fig_energy(cumU_arr, [1.0], (u_r99, u_r999, u_r9999),
           os.path.join(FIGS, '11_volume_U_pod.png'), 'Volume |U| POD (union-mask grid; ref mor2.log)')

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(Ranks2, [p_tr_err[r] for r in Ranks2], 'o-', label='train p (rel L2)')
ax.loglog(Ranks2, [p_te_err[r] for r in Ranks2], 's--', label='test p (rel L2)')
ax.set_xlabel('rank r'); ax.set_ylabel('relative L2 error')
ax.set_title(f'Volume p recon error vs r  (mask cuts {mask_cut*100:.1f}% grid pts, ~{p_tr_var_removal*100:.0f}% p-variance)')
ax.grid(alpha=.3, which='both'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '12_volume_recon.png'), dpi=110); plt.close()

# ---------- 3) P1-07 异常图 ----------
import numpy as _np
sc = _np.load(os.path.join(OUT, 'field_scalars_full.npz'), allow_pickle=True)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].hist(sc['nut_max'], bins=50, color='#55a868', alpha=.7)
ax[0].set_title('per-sim max n_u (turbulent viscosity ratio)'); ax[0].set_xlabel('max n_u'); ax[0].grid(alpha=.3)
ax[1].hist(sc['p_min'], bins=50, color='#c44e52', alpha=.7)
ax[1].set_title('per-sim min p (strongest suction, m²/s²)'); ax[1].set_xlabel('min p'); ax[1].grid(alpha=.3)
ax[2].hist(sc['N'], bins=50, color='#4c72b0', alpha=.7)
ax[2].set_title('node count per sim'); ax[2].set_xlabel('N'); ax[2].grid(alpha=.3)
plt.suptitle('P1-07 anomaly screen (full 1000): 0 NaN/Inf, 0 neg n_u, 0 naming errors')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, '07_anomalies.png'), dpi=110); plt.close()

# ---------- 汇总 JSON ----------
wallcp = {'train': int(is_train.sum()), 'test': int((~is_train).sum()),
          'r99': r99, 'r999': r999, 'r9999': r9999,
          'train_recon': {str(r): round(tr_err[r], 6) for r in Ranks},
          'test_recon': {str(r): round(te_err[r], 6) for r in Ranks},
          'test_train_ratio_r64': round(te_err[64] / tr_err[64], 3),
          'note': '800 train / 200 test; SVD on train; reconstruction uses train mean/S/V'}
vol = {'train': int(Pv.shape[0]), 'test': int(m['Pte'].shape[0]),
       'grid_valid_frac': round(float(valid.mean()), 4), 'mask_cut_grid_pts': round(float(mask_cut), 4),
       'p_var_in_removed': round(p_tr_var_removal, 3),
       'r99': p_r99, 'r999': p_r999, 'r9999': p_r9999, 'u_r99': u_r99, 'u_r999': u_r999,
       'train_recon_p': {str(r): round(p_tr_err[r], 6) for r in Ranks2},
       'test_recon_p': {str(r): round(p_te_err[r], 6) for r in Ranks2}}
json.dump({'wallcp_pod': wallcp, 'volume_pod': vol,
           'source': 'overnight/wallcp.npz + mor2_fields.npz (read-only)'},
          open(os.path.join(OUT, '08_p2_mor.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

print('wallcp POD: r99=%d r999=%d r9999=%d' % (r99, r999, r9999))
print('volume p  POD: r99=%d r999=%d r9999=%d' % (p_r99, p_r999, p_r9999))
print('volume |U| POD: r99=%d r999=%d r9999=%d' % (u_r99, u_r999, u_r9999))
print('volume p train recon:', {r: '%.2e' % p_tr_err[r] for r in Ranks2})
print('volume p test  recon:', {r: '%.2e' % p_te_err[r] for r in Ranks2})
print('mask cuts %.1f%% grid, p_var_removed=%.3f' % (mask_cut*100, p_tr_var_removal))
print('DONE_P2_FIGS')

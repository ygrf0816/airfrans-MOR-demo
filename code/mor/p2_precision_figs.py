"""P2/P3 precision figures for the Chinese report: 6 random test samples,
panels = [真实p | 真实|U| | p误差 | |U|误差].
- 壁面 Cp(s): p2_wallcp_anchored.npz (CP 1000x256; first 800=train, last 200=test)
  + fresh train SVD (centered) -> self-consistent POD basis/alpha.
- 体积 p / |U|: p2_grid_self_c0.npz (40-sample 256x128 union-mask cache; 30 train+10 test)
  + fresh train SVD on valid cells.
CJK labels via plotfont. Writes 09_wallcp_6comp.png + 10_volume_6comp.png.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotfont
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "04_results", "mor"))
rng = np.random.default_rng(2026)

# ================= 图 1：壁面 Cp(s) 真值 vs MOR(r=8) vs 误差 =================
cpn = np.load(os.path.join(OUT, "p2_wallcp_anchored.npz"), allow_pickle=True)
CP = cpn["CP"]                 # (1000, 256)
names = cpn["names"].tolist()
CPtr = CP[:800]; CPtr_mean = CPtr.mean(0)
Cpte = CP[800:]
# fresh centered SVD on TRAIN only (self-consistent POD basis/alpha)
U, S, Vt = np.linalg.svd(CPtr - CPtr_mean, full_matrices=False)
r8 = 8
Phi = Vt[:r8].T                      # (256, 8)  basis
alpha_te = (Cpte - CPtr_mean) @ Phi   # (200, 8) test coords
recon_te = alpha_te @ Phi.T + CPtr_mean   # (200, 256)
rel_te = np.linalg.norm(Cpte - recon_te, axis=1) / np.linalg.norm(Cpte, axis=1)
sel = np.sort(rng.choice(200, size=6, replace=False))
lam = S**2
energy_r8 = float(np.sum(lam[:8]) / np.sum(lam))

s = np.linspace(0, 1, 256)
fig, axes = plt.subplots(6, 3, figsize=(13.5, 11), sharex="col")
for i, idx in enumerate(sel):
    true_c = Cpte[idx]; recon_c = recon_te[idx]; rel = np.abs(true_c - recon_c)
    ax = axes[i, 0]
    ax.plot(s, true_c, "-", color="#1f77b4", lw=1.8, label="真实 Cp" if i == 0 else None)
    ax.plot(s, recon_c, "--", color="#d62728", lw=1.8, label="MOR r=8 复原" if i == 0 else None)
    ax.axvline(0.2, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.set_ylabel("Cp"); ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=9, loc="lower left")
    nm = names[800 + idx].replace("airFoil2D_SST_", "")
    ax.set_title("%s  (relL2=%.4f)" % (nm[:22], float(rel_te[idx])), fontsize=8)
    ax = axes[i, 1]
    ax.semilogy(s, np.maximum(rel, 1e-4), color="#d62728", lw=1.0)
    ax.set_ylabel("相对误差 |ΔCp|/max|Cp|（log）"); ax.grid(alpha=0.3, which="both")
    if i == 0:
        ax.set_title("误差曲线", fontsize=8)
    ax = axes[i, 2]
    strip = np.column_stack([true_c, recon_c, rel])
    ax.imshow(strip, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5, interpolation="none")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["真实", "MOR", "误差"], fontsize=7)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["前缘", "", "中部", "", "后缘"], fontsize=7)
    if i == 0:
        ax.set_title("三行：真实/复原/误差", fontsize=8)
fig.suptitle("壁面压力 Cp(s)：6 条 test 样本 真值 vs MOR(r=8) 复原 vs 误差",
             fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_wallcp_6comp.png"), dpi=110, bbox_inches="tight")
plt.close()
print("-> 09_wallcp_6comp.png")
print("wallcp test relL2(r8) mean=%.4f median=%.4f p95=%.4f" %
      (rel_te.mean(), np.median(rel_te), np.percentile(rel_te, 95)))
print("wallcp 6-sample relL2(r8):", ["%.4f" % rel_te[i] for i in sel])
print("wallcp r=8 energy coverage=%.4f" % energy_r8)
json.dump({"sel_idx": [int(x) for x in sel],
           "names": [names[800 + i] for i in sel],
           "relL2_r8": [float(rel_te[i]) for i in sel],
           "test_relL2_r8_mean": float(rel_te.mean()),
           "test_relL2_r8_median": float(np.median(rel_te)),
           "r8_energy": energy_r8},
          open(os.path.join(OUT, "09_wallcp_6comp.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ================= 图 2：体积场 p / |U| 真值 vs MOR(r=8) vs 误差 =================
g = np.load(os.path.join(OUT, "p2_grid_self_c0.npz"), allow_pickle=True)
P = g["P"]; U = g["U"]; valid = g["valid"]; nx, ny = int(g["nx"]), int(g["ny"])
valid_all = valid[0]                       # 并集掩膜：True=流体域
valid2d = valid_all.reshape(ny, nx)
Pv_all = P[:, valid_all]; Uv_all = U[:, valid_all]
n40 = P.shape[0]
mu_p = Pv_all[:30].mean(0); mu_u = Uv_all[:30].mean(0)
Up_, Sp, Vtp = np.linalg.svd(Pv_all[:30] - mu_p, full_matrices=False)
Uu, Su, Vtu = np.linalg.svd(Uv_all[:30] - mu_u, full_matrices=False)
Phi_p = Vtp[:8].T; Phi_u = Vtu[:8].T
selv = np.sort(rng.choice(n40, size=6, replace=False))


def valid_image(vals_flat, img_shape):
    """vals_flat = values on valid cells -> full (ny,nx) image, invalid cells = nan."""
    img = np.full(img_shape, np.nan)
    img[valid2d] = vals_flat
    return img


fig, axes = plt.subplots(6, 4, figsize=(16, 13), sharex="col")
vol_rel_p = []; vol_rel_u = []
for i, idx in enumerate(selv):
    Pv = Pv_all[idx]; Qv = Uv_all[idx]
    Pr = ((Pv - mu_p) @ Phi_p) @ Phi_p.T + mu_p
    Qr = ((Qv - mu_u) @ Phi_u) @ Phi_u.T + mu_u
    rp = np.abs(Pv - Pr) / (np.abs(Pv).max() + 1e-12)
    rq = np.abs(Qv - Qr) / (np.abs(Qv).max() + 1e-12)
    Pimg = valid_image(Pv, (ny, nx))
    Qimg = valid_image(Qv, (ny, nx))
    rpimg = valid_image(rp, (ny, nx)); rqimg = valid_image(rq, (ny, nx))
    vm_p = float(np.nanmax(np.abs(Pv))); vm_q = float(np.nanmax(np.abs(Qv)))
    rp_rel = float(np.linalg.norm(Pv - Pr) / np.linalg.norm(Pv))
    rq_rel = float(np.linalg.norm(Qv - Qr) / np.linalg.norm(Qv))
    vol_rel_p.append(rp_rel); vol_rel_u.append(rq_rel)
    nm = g["names"][idx].replace("airFoil2D_SST_", "")[:16]
    # (i,0): true p ; (i,1): true |U| ; (i,2): p rel err ; (i,3): |U| rel err
    ax = axes[i, 0]
    ax.imshow(np.ma.masked_invalid(Pimg), cmap="RdBu_r", aspect="auto",
              vmin=-vm_p, vmax=vm_p, interpolation="nearest")
    ax.set_ylabel("静压 p (真值)")
    if i == 0:
        ax.set_title("样本 %s" % nm, fontsize=9)
    else:
        ax.set_title("p relL2=%.3f" % rp_rel, fontsize=8)
    ax = axes[i, 1]
    ax.imshow(np.ma.masked_invalid(Qimg), cmap="coolwarm", aspect="auto",
              vmin=0, vmax=vm_q, interpolation="nearest")
    ax.set_ylabel("|U| (真值)")
    ax.set_title("|U| relL2=%.3f" % rq_rel, fontsize=8)
    ax = axes[i, 2]
    ax.imshow(np.ma.masked_invalid(rpimg), cmap="magma", aspect="auto",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_ylabel("p 相对误差")
    ax = axes[i, 3]
    ax.imshow(np.ma.masked_invalid(rqimg), cmap="magma", aspect="auto",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_ylabel("|U| 相对误差")
    if i == 5:
        for j, lab in enumerate(["真实 p", "真实 |U|", "p 相对误差", "|U| 相对误差"]):
            axes[i, j].set_xlabel(lab, fontsize=9)
fig.suptitle("体积场 6 样本：真值（p、|U|）vs MOR(r=8) 相对误差场（256×128 并集掩膜网格，留白=翼型内部）",
             fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_volume_6comp.png"), dpi=110, bbox_inches="tight")
plt.close()
print("-> 10_volume_6comp.png")
print("volume p 6-sample relL2(r8):", ["%.4f" % x for x in vol_rel_p])
print("volume |U| 6-sample relL2(r8):", ["%.4f" % x for x in vol_rel_u])
json.dump({"sel_idx": [int(x) for x in selv],
           "names": [g["names"][i] for i in selv],
           "relL2_r8_p": vol_rel_p, "relL2_r8_u": vol_rel_u,
           "r8_energy_p": float(np.sum(Sp[:8] ** 2) / np.sum(Sp ** 2)),
           "r8_energy_u": float(np.sum(Su[:8] ** 2) / np.sum(Su ** 2))},
          open(os.path.join(OUT, "10_volume_6comp.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("DONE_6COMP_FIGS")

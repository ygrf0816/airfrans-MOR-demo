"""Volume-field 6-sample figure, CORRECTED layout per user feedback:
per-field triad  [真值 | MOR重构 | 相对误差] for p and for |U| separately.
So 6 rows x 6 cols: (p真值, p重构, p误差, |U|真值, |U|重构, |U|误差).
Invalid (airfoil-interior) cells left blank (masked NaN).
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

g = np.load(os.path.join(OUT, "p2_grid_self_c0.npz"), allow_pickle=True)
P = g["P"]; U = g["U"]; valid = g["valid"]; nx, ny = int(g["nx"]), int(g["ny"])
valid_all = valid[0]
valid2d = valid_all.reshape(ny, nx)
Pv_all = P[:, valid_all]; Uv_all = U[:, valid_all]
mu_p = Pv_all[:30].mean(0); mu_u = Uv_all[:30].mean(0)
Up_, Sp, Vtp = np.linalg.svd(Pv_all[:30] - mu_p, full_matrices=False)
Uu, Su, Vtu = np.linalg.svd(Uv_all[:30] - mu_u, full_matrices=False)
Phi_p = Vtp[:8].T; Phi_u = Vtu[:8].T
selv = np.sort(rng.choice(P.shape[0], size=6, replace=False))


def valid_image(vals_flat, img_shape):
    img = np.full(img_shape, np.nan)
    img[valid2d] = vals_flat
    return img


fig, axes = plt.subplots(6, 6, figsize=(18, 13), sharex="col")
vol_rel_p = []; vol_rel_u = []
for i, idx in enumerate(selv):
    Pv = Pv_all[idx]; Qv = Uv_all[idx]
    Pr = ((Pv - mu_p) @ Phi_p) @ Phi_p.T + mu_p
    Qr = ((Qv - mu_u) @ Phi_u) @ Phi_u.T + mu_u
    rp = np.abs(Pv - Pr) / (np.abs(Pv).max() + 1e-12)
    rq = np.abs(Qv - Qr) / (np.abs(Qv).max() + 1e-12)
    Pimg = valid_image(Pv, (ny, nx)); Primg = valid_image(Pr, (ny, nx))
    Qimg = valid_image(Qv, (ny, nx)); Qrimg = valid_image(Qr, (ny, nx))
    rpimg = valid_image(rp, (ny, nx)); rqimg = valid_image(rq, (ny, nx))
    vm_p = float(np.nanmax(np.abs(Pv))); vm_q = float(np.nanmax(np.abs(Qv)))
    rp_rel = float(np.linalg.norm(Pv - Pr) / np.linalg.norm(Pv))
    rq_rel = float(np.linalg.norm(Qv - Qr) / np.linalg.norm(Qv))
    vol_rel_p.append(rp_rel); vol_rel_u.append(rq_rel)
    nm = g["names"][idx].replace("airFoil2D_SST_", "")[:16]
    # 6 columns: p真值 / p重构 / p误差 / |U|真值 / |U|重构 / |U|误差
    specs = [
        (Pimg, "RdBu_r", -vm_p, vm_p, "真实 p"),
        (Primg, "RdBu_r", -vm_p, vm_p, "MOR 重构 p (r=8)"),
        (rpimg, "magma", 0.0, 1.0, "p 相对误差"),
        (Qimg, "coolwarm", 0.0, vm_q, "真实 |U|"),
        (Qrimg, "coolwarm", 0.0, vm_q, "MOR 重构 |U| (r=8)"),
        (rqimg, "magma", 0.0, 1.0, "|U| 相对误差"),
    ]
    for j, (img, cmap, lo, hi, lab) in enumerate(specs):
        ax = axes[i, j]
        ax.imshow(np.ma.masked_invalid(img), cmap=cmap, aspect="auto",
                  vmin=lo, vmax=hi, interpolation="nearest")
        if i == 0:
            ax.set_title("%s  [%s]" % (nm, lab), fontsize=8)
        elif i == 5:
            ax.set_xlabel(lab, fontsize=9)
        else:
            ax.set_title("%s  (p %.3f / |U| %.3f)" %
                         (lab, rp_rel, rq_rel), fontsize=8)
        if j == 0:
            ax.set_ylabel("样本 %s" % nm, fontsize=8)
fig.suptitle("体积场 6 样本：真实值 → MOR(r=8) 重构 → 相对误差（p 与 |U| 各三列；256×128 并集掩膜，留白=翼型内部）",
             fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_volume_6comp.png"), dpi=110, bbox_inches="tight")
plt.close()
print("-> 10_volume_6comp.png (CORRECTED: 真值/重构/误差 per field)")
print("volume p 6-sample relL2(r8):", ["%.4f" % x for x in vol_rel_p])
print("volume |U| 6-sample relL2(r8):", ["%.4f" % x for x in vol_rel_u])
json.dump({"sel_idx": [int(x) for x in selv],
           "names": [g["names"][i] for i in selv],
           "relL2_r8_p": vol_rel_p, "relL2_r8_u": vol_rel_u,
           "r8_energy_p": float(np.sum(Sp[:8] ** 2) / np.sum(Sp ** 2)),
           "r8_energy_u": float(np.sum(Su[:8] ** 2) / np.sum(Su ** 2))},
          open(os.path.join(OUT, "10_volume_6comp.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("DONE_6COMP_FIGS_V2")

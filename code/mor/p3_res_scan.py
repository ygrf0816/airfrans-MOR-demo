"""P3-02 resolution scan: re-interpolate the SAME cached 40-sample fields
(P2-01 p2_grid_self_c0.npz, 256x128 valid-cell crop) onto 192x96 and 384x192
grids (same crop box x[-0.5,1.5] y[-0.5,0.5]), and report r(99%)/r(99.9%) +
mode-1 energy for p and |U|. Same data, same 30-train/10-test split, only
grid density changes -> isolates the resolution effect on POD rank.
train-only SVD. No .vtu re-read (fast, reproducible).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "04_results", "mor")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

d = np.load(os.path.join(OUT, "p2_grid_self_c0.npz"), allow_pickle=True)
P40 = d["P"]; U40 = d["U"]; names = d["names"].tolist()
mask256 = d["valid"][0]  # valid mask on the 256x128 grid
nx0, ny0 = d["nx"], d["ny"]  # 256,128
xs0 = d["xs"]; ys0 = d["ys"]
X0, X1, Y0, Y1 = float(xs0[0]), float(xs0[-1]), float(ys0[0]), float(ys0[-1])
print("crop box x[%.2f,%.2f] y[%.2f,%.2f]" % (X0, X1, Y0, Y1))

# original 256x128 grid coords for re-interp source
GX0, GY0 = np.meshgrid(xs0, ys0, indexing="xy")
SRC = np.column_stack([GX0.ravel(), GY0.ravel()])
# valid source points = mask256 True (drop wall-interior points)
src_pts = SRC[mask256]
src_p = P40[:, mask256]   # (40, n_valid256)
src_u = U40[:, mask256]
print("source valid pts on 256x128: %d" % src_pts.shape[0])

from scipy.spatial import cKDTree

tree = cKDTree(src_pts)  # one build on the 256x128 valid points
results = {}
resolutions = [(192, 96), (256, 128), (384, 192)]
n_train, n_test = 30, 10
for (nx, ny) in resolutions:
    xs = np.linspace(X0, X1, nx); ys = np.linspace(Y0, Y1, ny)
    GX, GY = np.meshgrid(xs, ys, indexing="xy")
    DST = np.column_stack([GX.ravel(), GY.ravel()])
    # vectorized nearest-neighbor: each target cell -> nearest source valid cell
    _, nn = tree.query(DST, k=1)
    P_new = src_p[:, nn]   # (40, nx*ny)  nearest-neighbor re-grid of p
    U_new = src_u[:, nn]   # (40, nx*ny)  nearest-neighbor re-grid of u
    valid = np.ones(nx * ny, dtype=bool)
    P_v = P_new[:, valid]; U_v = U_new[:, valid]
    # train/test split
    Pp_tr = P_v[:n_train]; Pu_tr = U_v[:n_train]
    Pp_te = P_v[n_train:n_train + n_test]; Pu_te = U_v[n_train:n_train + n_test]
    mup = Pp_tr.mean(0); muu = Pu_tr.mean(0)
    Yp = Pp_tr - mup; Yu = Pu_tr - muu
    _, Sp, _ = np.linalg.svd(Yp, full_matrices=False)
    _, Su, _ = np.linalg.svd(Yu, full_matrices=False)
    en_p = np.cumsum(Sp**2) / (Sp**2).sum()
    en_u = np.cumsum(Su**2) / (Su**2).sum()
    r99_p = int(np.searchsorted(en_p, 0.99, side="right"))
    r99_u = int(np.searchsorted(en_u, 0.99, side="right"))
    r999_p = int(np.searchsorted(en_p, 0.999, side="right"))
    r999_u = int(np.searchsorted(en_u, 0.999, side="right"))
    e1_p = float((Sp[0]**2) / (Sp**2).sum())
    e1_u = float((Su[0]**2) / (Su**2).sum())
    results["%dx%d" % (nx, ny)] = {
        "n_valid_cells": int(valid.sum()), "n_grid": int(valid.size),
        "p_r99": r99_p, "p_r999": r999_p, "u_r99": r99_u, "u_r999": r999_u,
        "p_mode1_energy": e1_p, "u_mode1_energy": e1_u,
        "p_S0": float(Sp[0]), "u_S0": float(Su[0]),
    }
    print("%dx%d: valid=%d  p r99=%d r999=%d (mode1=%.3f)  u r99=%d r999=%d (mode1=%.3f)"
          % (nx, ny, int(valid.sum()), r99_p, r999_p, e1_p, r99_u, r999_u, e1_u))

json.dump({"resolutions": results,
           "note": "P3-02 resolution scan; SAME cached 40-sample fields (P2-01), "
                   "only grid density changes; nearest re-interp; train-only 30/10 split",
           "independent_192x96_ref": {"p_r99": 4, "u_r99": 2, "p_mode1_energy": 0.881,
                                       "u_mode1_energy": 0.927,
                                       "src": "independent_check/p2_mode_physics.json (external, not independently reproduced)"}
           }, open(os.path.join(OUT, "07_resolution_scan.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("-> 07_resolution_scan.json")

# plot
fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
dens = [(int(a), int(b)) for (a, b) in [(192, 96), (256, 128), (384, 192)]]
pc = [results["%dx%d" % t]["p_r99"] for t in dens]
uc = [results["%dx%d" % t]["u_r99"] for t in dens]
xs_p = [192 * 96, 256 * 128, 384 * 192]
ax.plot(xs_p, pc, "o-", color="#1f77b4", label="p r(99%)")
ax.plot(xs_p, uc, "s-", color="#d62728", label="|U| r(99%)")
# overlay independent 192x96 u r99=2
ax.plot(xs_p[0], 2, "s", color="#d62728", mfc="none", ms=12, label="|U| r99=2 (indep. 192x96)")
ax.set_xscale("log"); ax.set_xlabel("total grid cells (resolution)")
ax.set_ylabel("r(99%)")
ax.set_title("POD rank vs grid resolution (same data, 40-sample checkpoint)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_resolution_scan.png"), bbox_inches="tight")
plt.close()
print("-> 07_resolution_scan.png")
print("DONE_P3_P2_RES_SCAN")

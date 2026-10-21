"""P3-01: Volume-field POD on P2-01's 256x128 union-mask grid.
Fills in the gap: P2-01 built the public grid + wall mask + per-sample interp
cache; P3-01 now runs train-only POD on p and |U| over the valid (masked) cells.
Reports: energy decay, r(99%), r(99.9%), test relL2(r) for r in {8,16,32,64,128,256},
AND absolute RMSE + field amplitude (the "amplitude trap" per P2_REFERENCE §2).
40-sample checkpoint (30 train + 10 test); full-400 deferred.
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "04_results", "mor")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1common import PREFIX


def Uinf_of(nm):
    return float(nm[len(PREFIX):].split("_")[0])


# Load the 40-sample cache
d = np.load(os.path.join(OUT, "p2_grid_self_c0.npz"), allow_pickle=True)
NG = d["total_grid"]
mask = d["valid"][0]  # union mask (fixed crop box)
n_valid = int(mask.sum())
P40 = d["P"]; U40 = d["U"]
names40 = d["names"].tolist()
print("grid %dx%d=%d pts, valid=%d (%.2f%%)" % (d["nx"], d["ny"], NG, n_valid, mask.mean() * 100))
print("40-sample P/U cached: P %s U %s" % (P40.shape, U40.shape))

# Extract valid cells only
Pv = P40[:, mask]  # (40, n_valid)
Uv = U40[:, mask]
print("P valid: %s, U valid: %s, finite frac P=%.3f U=%.3f"
      % (Pv.shape, Uv.shape, np.isfinite(Pv).mean(), np.isfinite(Uv).mean()))

# --- Split: 30 train + 10 test ---
n_train, n_test = 30, 10
tr = slice(0, n_train); te = slice(n_train, n_train + n_test)
Ptr = Pv[tr]; Utr = Uv[tr]
Pte = Pv[te]; Ute = Uv[te]
print("train %d, test %d" % (n_train, n_test))

# --- SVD POD on p (valid cells, train-only) ---
mu_p = Ptr.mean(0); mu_u = Utr.mean(0)
Yp = Ptr - mu_p; Yu = Utr - mu_u
print("\n=== POD p (valid cells, n=%d x %d) ===" % Yp.shape, flush=True)
tp0 = time.time()
Up_, Sp, Vtp = np.linalg.svd(Yp, full_matrices=False)
t_svd_p = time.time() - tp0
lam_p = Sp**2; en_p = np.cumsum(lam_p) / lam_p.sum()
r99_p = int(np.searchsorted(en_p, 0.99, side="right"))
r999_p = int(np.searchsorted(en_p, 0.999, side="right"))
print("SVD p took %.1fs; S[0:6]=%s" % (t_svd_p, np.round(Sp[:6], 1).tolist()))
print("r99_p=%d  r999_p=%d" % (r99_p, r999_p))

print("\n=== POD |U| (valid cells) ===", flush=True)
tu0 = time.time()
Uu, Su, Vtu = np.linalg.svd(Yu, full_matrices=False)
t_svd_u = time.time() - tu0
lam_u = Su**2; en_u = np.cumsum(lam_u) / lam_u.sum()
r99_u = int(np.searchsorted(en_u, 0.99, side="right"))
r999_u = int(np.searchsorted(en_u, 0.999, side="right"))
print("SVD |U| took %.1fs; S[0:6]=%s" % (t_svd_u, np.round(Su[:6], 1).tolist()))
print("r99_u=%d  r999_u=%d" % (r99_u, r999_u))

# --- Test relL2 ---
# Use full rank (30 modes max for 30 samples)
Pp = Vtp.T  # (n_valid, 30)
Pu = Vtu.T
r_list = [8, 16, 32, 64, 128, 256]
test_relL2_p = {}; test_relL2_u = {}
for r in r_list:
    if r > min(n_train, n_valid):
        test_relL2_p[r] = float("nan")
        test_relL2_u[r] = float("nan")
        continue
    Yte_p = Pte - mu_p
    Yte_u = Ute - mu_u
    Pr_p = Pp[:, :r]
    Pr_u = Pu[:, :r]
    R_p = Yte_p @ Pr_p @ Pr_p.T
    R_u = Yte_u @ Pr_u @ Pr_u.T
    test_relL2_p[r] = float(np.linalg.norm(Yte_p - R_p) / np.linalg.norm(Yte_p))
    test_relL2_u[r] = float(np.linalg.norm(Yte_u - R_u) / np.linalg.norm(Yte_u))
print("\ntest relL2 p (30-train/10-test):", {str(k): ("%.4e" % v if not np.isnan(v) else "n/a") for k, v in test_relL2_p.items()})
print("test relL2 u (30-train/10-test):", {str(k): ("%.4e" % v if not np.isnan(v) else "n/a") for k, v in test_relL2_u.items()})

# --- Absolute RMSE + field amplitude (amplitude trap) ---
# Cp = p / (0.5 * Uinf^2); U in m/s
Uinf_tr = np.array([Uinf_of(nm) for nm in names40[:n_train]])
Uinf_te = np.array([Uinf_of(nm) for nm in names40[n_train:n_train + n_test]])
q_te = 0.5 * Uinf_te**2
cp_true = Pte / q_te[:, None]  # (10, n_valid) in Cp units
u_amp = float(np.abs(Ute).mean())  # mean |U| magnitude in m/s
cp_amp = float(np.abs(cp_true).mean())
print("\n|U| amp mean = %.1f m/s, Cp amp mean = %.3f" % (u_amp, cp_amp))

cp_rmse_list = []; u_rmse_list = []
for r in [8, 16, 32, 64, 128]:
    if r > min(n_train, n_valid):
        cp_rmse_list.append(float("nan")); u_rmse_list.append(float("nan"))
        continue
    Pr_p = Pp[:, :r]; Pr_u = Pu[:, :r]
    Yte_p = Pte - mu_p; Yte_u = Ute - mu_u
    recon_p = Yte_p @ Pr_p @ Pr_p.T + mu_p
    recon_u = Yte_u @ Pr_u @ Pr_u.T + mu_u
    cp_recon = recon_p / q_te[:, None]
    rmse_cp = float(np.sqrt(((cp_recon - cp_true)**2).mean(axis=1)).mean())
    rmse_ms = float(np.sqrt(((recon_u - Ute)**2).mean(axis=1)).mean())
    cp_rmse_list.append(rmse_cp)
    u_rmse_list.append(rmse_ms)
rs = [8, 16, 32, 64, 128]
print("abs RMSE + amplitude (test 10 samples):")
for i, r in enumerate(rs):
    print("  r=%3d  Cp_RMSE=%.4f  |U|_RMSE=%.4f m/s  (Cp_amp=%.3f, |U|_amp=%.1f m/s)"
          % (r, cp_rmse_list[i], u_rmse_list[i], cp_amp, u_amp))

# --- save ---
res = {
    "n_train": n_train, "n_test": n_test, "n_valid_cells": n_valid,
    "grid": "%dx%d (P2-01 union mask, crop x[-0.5,1.5] y[-0.5,0.5])" % (d["nx"], d["ny"]),
    "p_pod": {
        "r99": r99_p, "r999": r999_p,
        "S_first6": Sp[:6].tolist(),
        "test_relL2": {str(k): v for k, v in test_relL2_p.items()},
        "cp_RMSE": {str(r): cp_rmse_list[i] for i, r in enumerate(rs)},
        "cp_amp_mean": cp_amp, "p_amp_mean": float(np.abs(Pte).mean()),
    },
    "u_pod": {
        "r99": r99_u, "r999": r999_u,
        "S_first6": Su[:6].tolist(),
        "test_relL2": {str(k): v for k, v in test_relL2_u.items()},
        "u_RMSE_ms": {str(r): u_rmse_list[i] for i, r in enumerate(rs)},
        "u_amp_mean_ms": u_amp,
    },
    "svd_time_s": {"p": t_svd_p, "u": t_svd_u},
    "source": "P3-01 self-built; P2-01 256x128 union-mask grid; 40-sample checkpoint "
              "(30 train + 10 test); full-400 deferred",
    "amplitude_trap_note": "P2_REFERENCE §2: reporting only relative error without "
                           "absolute RMSE + field amplitude gives wrong compression "
                           "conclusion. This P3-01 reports both (cp_RMSE + u_RMSE + amps).",
}
json.dump(res, open(os.path.join(OUT, "06_volume_pod.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n-> 06_volume_pod.json")
print("KEY p: r99=%d r999=%d  u: r99=%d r999=%d" % (r99_p, r999_p, r99_u, r999_u))
print("test relL2 p(r=8)=%.4e  u(r=8)=%.4e" % (test_relL2_p[8], test_relL2_u[8]))
print("cp_RMSE(r=8)=%.4f  u_RMSE(r=8)=%.4f m/s" % (cp_rmse_list[0], u_rmse_list[0]))

# plot
# plot (only use r up to min(n_train, N) since test_relL2 nan beyond that)
r_plot = [r for r in r_list if not np.isnan(test_relL2_p[r])]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
N_plot = min(len(en_p), n_train)
ax[0].semilogy(range(1, N_plot + 1), en_p[:N_plot], "o-", color="#1f77b4", label="p (valid cells)")
ax[0].semilogy(range(1, N_plot + 1), en_u[:N_plot], "s-", color="#d62728", label="|U| (valid cells)")
ax[0].axhline(0.99, color="k", ls="--", lw=0.8, label="99%")
ax[0].axhline(0.999, color="k", ls=":", lw=0.8, label="99.9%")
ax[0].set_xlabel("mode index"); ax[0].set_ylabel("cumulative energy fraction")
ax[0].set_title("Volume-field POD cumulative energy (valid cells, 40-sample)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].semilogy(r_plot, [test_relL2_p[r] for r in r_plot], "o-", label="p test relL2")
ax[1].semilogy(r_plot, [test_relL2_u[r] for r in r_plot], "s-", label="|U| test relL2")
ax[1].set_xlabel("r (POD modes)"); ax[1].set_ylabel("relative L2 reconstruction error (test)")
ax[1].set_title("Test relL2 vs r (30-train/10-test, 40-sample checkpoint)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_volume_pod.png"), bbox_inches="tight")
plt.close()
print("-> 06_volume_pod.png")
print("DONE_P3_P1_VOLUME_POD_40SAMPLE")

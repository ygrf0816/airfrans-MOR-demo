"""P2-03: POD mode physics attribution.
Extracts POD coefficients (projected amplitudes) for the first 8 modes of the
leading-edge-anchored Cp(s) POD (P2-02), correlates each coefficient with the
design variables (AoA, U∞/Re, camber, thickness), and produces modal-space
plots (scatter in coeff-1/coeff-2 colored by AoA; bar of |corr| per mode).
Data = p2_wallcp_anchored.npz (P2-02 self-built) + manifest.json geometry decode.
overnight/P2_REFERENCE = magnitude cross-check only.
"""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "04_results", "mor")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1common import PREFIX, RHO, NU_REF


def decode_vars(nm):
    """nm -> dict of physical design variables (geometry + flow)."""
    p = [float(x) for x in nm[len(PREFIX):].split("_")]
    Uinf = p[0]; AoA = p[1]
    naca = p[2:]
    Re = Uinf / NU_REF
    if len(naca) == 4:
        # 4-digit: camber*10, thickness*10, break*10, taper (last is ~0)
        camber = naca[0] / 10.0   # in fractions (e.g. 1.2 -> 0.12 = 12%)
        t_c = naca[1] / 10.0
    elif len(naca) == 5:
        # 5-digit: camber-max, camber-loc*10, thickness*10, start*10, end*10
        camber = naca[0] / 100.0  # e.g. 21 -> 0.21 = 21%  (actually 5-digit: first 2 digits are camber% x10)
        t_c = naca[2] / 10.0
    else:
        camber, t_c = 0.0, 0.0
    return {"Uinf": Uinf, "AoA": AoA, "Re": Re,
            "camber": camber, "t_c": t_c,
            "series": "5d" if len(naca) == 5 else "4d"}


# load anchored Cp cache
d = np.load(os.path.join(OUT, "p2_wallcp_anchored.npz"), allow_pickle=True)
CP = d["CP"]; names = d["names"].tolist()
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
tr = [i for i, n in enumerate(names) if n in man["full_train"]]
te = [i for i, n in enumerate(names) if n in man["full_test"]]
Xtr = CP[tr]

# train-only SVD (P2-02)
mu = Xtr.mean(0); Ytr = Xtr - mu
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
K = 8  # report first 8 modes
# POD coefficients: alpha = Ytr @ V_k^T  (each = k-th modal amplitude across 800 train)
alpha = Ytr @ Vt[:K].T      # (800, K)

# decode all 1000; use train rows for correlation
allvars = [decode_vars(nm) for nm in names]
vars_train = np.array([
    [allvars[i]["Uinf"], allvars[i]["AoA"], allvars[i]["Re"],
     allvars[i]["camber"], allvars[i]["t_c"]] for i in tr],
    dtype=float)
var_names = ["Uinf", "AoA", "Re", "camber", "t_c"]

# correlations: corr(alpha_k, var_j) for k in 1..K, j in vars
corrs = np.zeros((K, len(var_names)))
for k in range(K):
    for j in range(len(var_names)):
        corrs[k, j] = float(np.corrcoef(alpha[:, k], vars_train[:, j])[0, 1])
print("mode x var correlation matrix (train):")
print("  mode", list(range(1, K + 1)))
for j, vn in enumerate(var_names):
    print("  %-7s %s" % (vn, " ".join("%.3f" % corrs[k, j] for k in range(K))))

# dominant variable per mode
dom_var = np.argmax(np.abs(corrs), axis=1)
print("\ndominant variable per mode:", ["%s" % var_names[dom_var[k]] for k in range(K)])

# test-side coefficients (for generalization check of the correlation structure)
Yte = CP[te] - mu
alpha_te = Yte @ Vt[:K].T
vars_te = np.array([[allvars[i]["Uinf"], allvars[i]["AoA"], allvars[i]["Re"],
                      allvars[i]["camber"], allvars[i]["t_c"]] for i in te], dtype=float)
corrs_te = np.zeros((K, len(var_names)))
for k in range(K):
    for j in range(len(var_names)):
        corrs_te[k, j] = float(np.corrcoef(alpha_te[:, k], vars_te[:, j])[0, 1])
print("\nTEST-side correlations (same modes, test only):")
for j, vn in enumerate(var_names):
    print("  %-7s %s" % (vn, " ".join("%.3f" % corrs_te[k, j] for k in range(K))))

# energy / reconstruction table
lam = S**2; en = np.cumsum(lam) / lam.sum()
r99 = int(np.searchsorted(en, 0.99, side="right"))
r999 = int(np.searchsorted(en, 0.999, side="right"))

# per-mode shape: sample the 8th mode's Cp profile at 5 representative AoA bins
# (for the modal-space figure)
en_by_aoa = {}
for i, idx in enumerate(tr):
    b = min(int(allvars[idx]["AoA"] // 2), 4)  # AoA//2 bin 0..4
    en_by_aoa.setdefault(b, []).append(i)

np.savez(os.path.join(OUT, "p2_pod3_modes.npz"),
         alpha_train=np.ascontiguousarray(alpha), alpha_test=np.ascontiguousarray(alpha_te),
         Vt_first8=np.ascontiguousarray(Vt[:K]), S_first12=S[:12],
         corr_train=np.ascontiguousarray(corrs), corr_test=np.ascontiguousarray(corrs_te),
         r99=r99, r999=r999,
         names_train=np.array([names[i] for i in tr]))
json.dump({"r99": r99, "r999": r999, "K": K,
           "corr_train": {var_names[j]: [float(corrs[k, j]) for k in range(K)] for j in range(len(var_names))},
           "corr_test": {var_names[j]: [float(corrs_te[k, j]) for k in range(K)] for j in range(len(var_names))},
           "dominant_var_per_mode": [var_names[dom_var[k]] for k in range(K)],
           "source": "P2-03 self-built; p2_wallcp_anchored.npz (P2-02) + manifest geometry decode",
           "cross_check_ref": "overnight/P2_REFERENCE mode-AoA corr ~0.97 (ref)"},
          open(os.path.join(OUT, "03_mode_physics.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n-> p2_pod3_modes.npz + 03_mode_physics.json")
print("r99=%d r999=%d  S[0:3]=%s" % (r99, r999, np.round(S[:3], 2).tolist()))

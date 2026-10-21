import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "04_results", "mor")
d = np.load(os.path.join(OUT, "p2_wallcp_anchored.npz"), allow_pickle=True)
print("anchored CP shape:", d["CP"].shape)
# sim0 range
print("sim0 CP p10/p50/p90:", np.round(np.percentile(d["CP"][0], [10, 50, 90]), 4))
# per-sample std p05/p50/p95
print("per-sample Cp std p05/p50/p95:", np.round(np.percentile(d["CP"].std(1), [5, 50, 95]), 4))
# train-only SVD to get the exact numbers
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
names = d["names"].tolist()
tr = [i for i, n in enumerate(names) if n in man["full_train"]]
te = [i for i, n in enumerate(names) if n in man["full_test"]]
Xtr = d["CP"][tr]; Xte = d["CP"][te]
mu = Xtr.mean(0); Ytr = Xtr - mu
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
en = np.cumsum(S**2) / (S**2).sum()
r99 = int(np.searchsorted(en, 0.99, side="right"))
r999 = int(np.searchsorted(en, 0.999, side="right"))
r9999 = int(np.searchsorted(en, 0.9999, side="right"))
print("r99=%d r999=%d r9999=%d" % (r99, r999, r9999))
P = Vt.T; Yte = Xte - mu
for r in [2, 4, 8, 16, 32, 64]:
    Rr = (Yte @ P[:, :r]) @ P[:, :r].T
    trr = float(np.sqrt(max(0.0, 1.0 - min(en[r-1], 1.0))))
    ter = float(np.linalg.norm(Yte - Rr) / np.linalg.norm(Yte))
    print("r=%2d  train_relL2=%.4e  test_relL2=%.4e" % (r, trr, ter))
# mode physics
Uinf = np.array([float(n.split("_")[2]) for n in names])
AoA = np.array([float(n.split("_")[3]) for n in names])
coeffs = Ytr @ Vt[:8].T
for k in range(8):
    c_aoa = float(np.corrcoef(coeffs[:, k], AoA[tr])[0, 1])
    c_uinf = float(np.corrcoef(coeffs[:, k], Uinf[tr])[0, 1])
    if abs(c_aoa) > 0.1 or abs(c_uinf) > 0.1:
        print("mode %d: corr(AoA)=%.3f corr(Uinf)=%.3f" % (k+1, c_aoa, c_uinf))

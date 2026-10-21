import sys, json, os
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "04_results", "mor")
os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(ROOT, "04_results", "analysis", "p2_wallcp_self.npz"), allow_pickle=True)
CP = d["CP"]; names = d["names"].tolist()
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
ft = set(man["full_train"]); fte = set(man["full_test"])
tr = [i for i, n in enumerate(names) if n in ft]
te = [i for i, n in enumerate(names) if n in fte]
assert len(tr) == 800 and len(te) == 200, "split sizes %d/%d" % (len(tr), len(te))
Xtr = CP[tr]; Xte = CP[te]
print("train %d x %d  test %d x %d" % (Xtr.shape[0], Xtr.shape[1], Xte.shape[0], Xte.shape[1]), flush=True)

# train-only centering (no test leakage)
mu = Xtr.mean(axis=0)
Ytr = Xtr - mu
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
lam = S**2
energy = np.cumsum(lam) / lam.sum()
print("S[0:12]:", np.round(S[:12], 3), flush=True)


def r_of(target):
    return int(np.searchsorted(energy, target, side="right"))


r99 = r_of(0.99); r999 = r_of(0.999); r9999 = r_of(0.9999)
print("r(99%%)=%d  r(99.9%%)=%d  r(99.99%%)=%d" % (r99, r999, r9999), flush=True)


def train_relL2(r):
    return float(np.sqrt(max(0.0, 1.0 - min(energy[r - 1], 1.0))))


P = Vt.T  # (256, 256)
Yte = Xte - mu
test_relL2 = {}
for r in [2, 4, 8, 16, 32, 64]:
    Rr = (Yte @ P[:, :r]) @ P[:, :r].T
    test_relL2[r] = float(np.linalg.norm(Yte - Rr) / np.linalg.norm(Yte))
    print("test relL2(r=%2d) = %.4e" % (r, test_relL2[r]), flush=True)

train_relL2_d = {r: train_relL2(r) for r in [2, 4, 8, 16, 32, 64]}

# mode physics: correlate first-8 POD coords with AoA / Uinf (decode from name idx 2,3)
def decode(nm):
    parts = nm.split("_")
    return float(parts[2]), float(parts[3])


Uinf = np.array([decode(nm)[0] for nm in names])
AoA = np.array([decode(nm)[1] for nm in names])
coeffs_tr = Ytr @ Vt[:8].T
corr = {}
for k in range(8):
    corr[k + 1] = {"AoA": float(np.corrcoef(coeffs_tr[:, k], AoA[tr])[0, 1]),
                   "Uinf": float(np.corrcoef(coeffs_tr[:, k], Uinf[tr])[0, 1])}
print("mode->AoA/Uinf corr:", json.dumps(corr), flush=True)

res = {
    "n_train": len(tr), "n_test": len(te), "n_modes": int(Vt.shape[0]),
    "S_first12": S[:12].tolist(),
    "r99": r99, "r999": r999, "r9999": r9999,
    "train_relL2": {str(k): v for k, v in train_relL2_d.items()},
    "test_relL2": {str(k): v for k, v in test_relL2.items()},
    "mode_corr": corr,
    "source": "self-computed p2_wallcp_self.npz (P1/P2-01) + full_train/full_test split (800/200)",
    "centering": "train-only mean (no test leakage)",
    "cross_check_ref": {"P2_REFERENCE_r99": 7, "P2_REFERENCE_test_relL2_r8": 6.01e-2},
    "note": "DISCREPANCY vs P2_REFERENCE r99=7: self-computed unnormalized-arc-length Cp gives "
            "r99=%d. P2_REFERENCE used normalized arc-length s in [0,1]; self CP s-grid may carry "
            "per-sample arc-length scale spread that inflates the rank. Flagged, not silently matched." % r99,
}
with open(os.path.join(OUT, "02_pod_energy.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("-> 04_results/mor/02_pod_energy.json", flush=True)
print("KEY: r99=%d r999=%d r9999=%d  test_relL2(r8)=%.4e  (ref 6.01e-2)"
      % (r99, r999, r9999, test_relL2[8]), flush=True)
print("DONE_P2_POD_WALLCP")

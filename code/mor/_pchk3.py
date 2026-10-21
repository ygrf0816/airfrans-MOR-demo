import sys, os, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

# overnight wallcp.npz (cross-check ONLY - magnitude) vs my self-computed
ov = os.path.join("C:/Users/Administrator/AppData/Local/Temp/agh-bridge/overnight", "wallcp.npz")
o = np.load(ov, allow_pickle=True)
CP_o = o["CP"]; is_train = o["is_train"].astype(bool)
print("overnight wallcp: CP %s  train=%d test=%d" % (CP_o.shape, is_train.sum(), (~is_train).sum()))
Xtr = CP_o[is_train]; Xte = CP_o[~is_train]
mu = Xtr.mean(0); Ytr = Xtr - mu
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
lam = S**2; en = np.cumsum(lam) / lam.sum()
r99 = int(np.searchsorted(en, 0.99, side="right"))
r999 = int(np.searchsorted(en, 0.999, side="right"))
print("overnight S[0:6]:", np.round(S[:6], 3))
print("overnight r99=%d r999=%d  (P2_REFERENCE says r99=7)" % (r99, r999))
# test relL2 r=8
P = Vt.T; Yte = Xte - mu
for r in [4, 8, 16, 32]:
    R = (Yte @ P[:r]) @ P[:r].T
    print("overnight test relL2(r=%d)=%.4e" % (r, np.linalg.norm(Yte - R) / np.linalg.norm(Yte)))

# now my self CP with the SAME split
d = np.load("04_results/analysis/p2_wallcp_self.npz", allow_pickle=True)
CPs = d["CP"]; names = d["names"].tolist()
man = json.load(open("02_dataset/Dataset/manifest.json", encoding="utf-8"))
ft = set(man["full_train"]); fte = set(man["full_test"])
tr = [i for i, n in enumerate(names) if n in ft]; te = [i for i, n in enumerate(names) if n in fte]
print("\nself CP: train rows=%d test rows=%d" % (len(tr), len(te)))
# compare a few train Cp vectors between self and overnight to see the magnitude gap
# align by name
o_names = o["names"].tolist()
o_idx = {n: i for i, n in enumerate(o_names)}
common = [n for n in names if n in o_idx][:5]
for n in common:
    i = names.index(n); j = o_idx[n]
    diff = np.max(np.abs(CPs[i] - CP_o[j]))
    print("  %s  self Cp p50=%.3f ov p50=%.3f  maxdiff=%.3f"
          % (n, np.median(CPs[i]), np.median(CP_o[j]), diff))

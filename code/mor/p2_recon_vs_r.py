"""P2-04 (CORRECTED): reconstruction error vs r (8/16/32/64/128/256), by split
and by LOCAL-Mach bin (<0.3 / 0.3~0.7 / >0.7). The >0.7 bin is the physical
transonic/sonic-overshoot zone (P1-07b uses LOCAL Mach = Umag_max/a, NOT the
free-stream Mach = Uinf/a). AoA edge is kept as a secondary bin dimension.
Data = p2_wallcp_anchored.npz (P2-02 self-built). Train-only SVD.
Mach counts from field_scalars_full.npz (_U_mag_max), cross-checked vs P1-07b.
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
A = 340.3  # m/s, ~298K (same convention as P1-07b local Mach = Umag_max/a)


def decode(nm):
    p = [float(x) for x in nm[len("airFoil2D_SST_"):].split("_")]
    return p[0], p[1]  # Uinf, AoA


d = np.load(os.path.join(OUT, "p2_wallcp_anchored.npz"), allow_pickle=True)
CP = d["CP"]; names = d["names"].tolist()
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
tr = [i for i, n in enumerate(names) if n in man["full_train"]]
te = [i for i, n in enumerate(names) if n in man["full_test"]]
Xtr = CP[tr]; Xte = CP[te]
mu = Xtr.mean(0); Ytr = Xtr - mu; Yte = Xte - mu
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
lam = S**2; en = np.cumsum(lam) / lam.sum()
r99 = int(np.searchsorted(en, 0.99, side="right"))
r999 = int(np.searchsorted(en, 0.999, side="right"))
print("r99=%d r999=%d  S[0:3]=%s" % (r99, r999, np.round(S[:3], 2).tolist()))

# ---- local Mach per sample from field_scalars_full.npz ----
sc = np.load(os.path.join(ROOT, "04_results", "analysis", "field_scalars_full_stats.npz"),
             allow_pickle=True)
Umag_max = sc["_U_mag_max"]
# align to `names` order: sc._sim_names
sc_names = [str(x) for x in sc["_sim_names"]]
idx = {n: i for i, n in enumerate(sc_names)}
# our cache names must match; build local Mach aligned to cache order
mach_local_cache = np.empty(len(names))
for i, n in enumerate(names):
    j = idx.get(n)
    if j is None:
        mach_local_cache[i] = np.nan
    else:
        mach_local_cache[i] = float(Umag_max[j]) / A
print("local Mach (cache order): max=%.3f  min=%.3f" % (np.nanmax(mach_local_cache), np.nanmin(mach_local_cache)))
mach_te = mach_local_cache[te]
print("full_test local Mach>0.7 n=%d / %d  max=%.3f"
      % (int((mach_te > 0.7).sum()), len(te), np.nanmax(mach_te)))

# AoA edge (secondary dimension)
AoA_te = np.array([decode(nm)[1] for nm in names])[te]
AoA_tr = np.array([decode(nm)[1] for nm in names])[tr]
p5, p95 = float(np.percentile(AoA_tr, 5)), float(np.percentile(AoA_tr, 95))
AoA_hi = AoA_te > p95
AoA_lo = AoA_te < p5
print("train AoA p5=%.2f p95=%.2f; test AoA>p95 n=%d <p5 n=%d"
      % (p5, p95, AoA_hi.sum(), AoA_lo.sum()))

# ---- Local-Mach bins for test: <0.3 / 0.3~0.7 / >0.7 ----
mb_lt03 = mach_te < 0.3
mb_0307 = (mach_te >= 0.3) & (mach_te <= 0.7)
mb_gt07 = mach_te > 0.7
print("test local-Mach bins: <0.3 n=%d, 0.3-0.7 n=%d, >0.7 n=%d"
      % (mb_lt03.sum(), mb_0307.sum(), mb_gt07.sum()))

r_list = [8, 16, 32, 64, 128, 256]
P_full = Vt.T


def _relL2(Yz, r, Pr):
    Rr = Yz @ Pr @ Pr.T
    return float(np.linalg.norm(Yz - Rr) / np.linalg.norm(Yz))


res = {"r99": r99, "r999": r999, "r_list": r_list, "n_train": len(tr), "n_test": len(te),
       "maxMach_test_local": float(np.nanmax(mach_te)),
       "n_mach_gt07_test": int(mb_gt07.sum()),
       "aoa_p5_train": p5, "aoa_p95_train": p95,
       "train_relL2": {}, "test_relL2": {},
       "mach_lt03_relL2": {}, "mach_03_07_relL2": {}, "mach_gt07_relL2": {},
       "ext_aoa_hi_relL2": {}, "ext_aoa_lo_relL2": {},
       "source": "P2-04 self-built (CORRECTED); p2_wallcp_anchored.npz + train-only SVD; "
                 "local Mach from field_scalars_full.npz _U_mag_max",
       "cross_check_ref": {"P2_REFERENCE_r64_test": 3.29e-4,
                           "P1_07b_localMach_gt07_full_test": 7},
       "note": "CORRECTION: P1-07b uses LOCAL Mach (Umag_max/a) up to 1.3; free-stream Mach "
               "(Uinf/a) is only ~0.274. An earlier P2-04 draft mixed these up and wrongly "
               "claimed 'test all subsonic, no Mach>0.7 zone' -- that was WRONG. Full_test has "
               "7 local-Mach>0.7 samples; reynolds_test/aoa_test have 24/21. This version "
               "restores the local-Mach binning."}
for r in r_list:
    Pr = P_full[:, :r]
    tr_err = _relL2(Ytr, r, Pr)
    te_err = _relL2(Yte, r, Pr)
    res["train_relL2"][str(r)] = tr_err
    res["test_relL2"][str(r)] = te_err
    # local-Mach bins on test
    res["mach_lt03_relL2"][str(r)] = _relL2(Yte[mb_lt03], r, Pr) if mb_lt03.sum() else float("nan")
    res["mach_03_07_relL2"][str(r)] = _relL2(Yte[mb_0307], r, Pr) if mb_0307.sum() else float("nan")
    res["mach_gt07_relL2"][str(r)] = _relL2(Yte[mb_gt07], r, Pr) if mb_gt07.sum() else float("nan")
    res["ext_aoa_hi_relL2"][str(r)] = _relL2(Yte[AoA_hi], r, Pr) if AoA_hi.sum() else float("nan")
    res["ext_aoa_lo_relL2"][str(r)] = _relL2(Yte[AoA_lo], r, Pr) if AoA_lo.sum() else float("nan")
    print("r=%3d  train=%.4e  test=%.4e  M<0.3=%.4e M03-07=%.4e M>0.7=%.4e  AoAhi=%.4e AoAlo=%.4e"
          % (r, tr_err, te_err,
             res["mach_lt03_relL2"][str(r)], res["mach_03_07_relL2"][str(r)],
             res["mach_gt07_relL2"][str(r)], res["ext_aoa_hi_relL2"][str(r)],
             res["ext_aoa_lo_relL2"][str(r)]))

json.dump(res, open(os.path.join(OUT, "04_recon_vs_r.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("-> 04_recon_vs_r.json")

# plot
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
rs = [r for r in r_list if r <= 256]
ax[0].semilogy(rs, [res["train_relL2"][str(r)] for r in rs], "o-", label="train", color="#1f77b4")
ax[0].semilogy(rs, [res["test_relL2"][str(r)] for r in rs], "s-", label="test(all)", color="#d62728")
ax[0].semilogy(rs, [res["mach_gt07_relL2"][str(r)] for r in rs], "^-", label="Mach>0.7 (n=7)", color="#9467bd")
ax[0].semilogy(rs, [res["mach_03_07_relL2"][str(r)] for r in rs], "v-", label="Mach 0.3-0.7", color="#2ca02c")
ax[0].set_xlabel("r (POD modes)"); ax[0].set_ylabel("relative L2 reconstruction error")
ax[0].set_title("Recon error vs r, by split + LOCAL-Mach bin (train-only SVD)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(range(1, 257), en, color="#1f77b4", lw=1.2)
ax[1].axhline(0.99, color="k", ls="--", lw=0.8)
ax[1].axhline(0.999, color="k", ls=":", lw=0.8)
ax[1].axvline(r99, color="r", ls="--", lw=0.8)
ax[1].set_xlabel("mode index"); ax[1].set_ylabel("cumulative energy fraction")
ax[1].set_title("POD cumulative energy (anchored Cp, train)")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_recon_vs_r.png"), bbox_inches="tight")
plt.close()
print("-> 04_recon_vs_r.png")
print("DONE_P2_P4_CORRECTED")

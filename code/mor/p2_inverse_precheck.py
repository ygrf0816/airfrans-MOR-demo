"""P2-05 inverse-design pre-check (fixed SVD/solve): sensitivity + monotonicity +
well-posedness + multi-solution of (Cl, Cd) wrt [AoA, camber, t_c].
Data = physics_full_cache.npz (self-computed Cl/Cd) + manifest decode. Train-only.
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
from p1common import PREFIX, NU_REF


def decode_geom(nm):
    p = [float(x) for x in nm[len(PREFIX):].split("_")]
    Uinf, AoA = p[0], p[1]
    Re = Uinf / NU_REF
    naca = p[2:]
    if len(naca) == 4:
        camber, t_c, series = naca[0] / 10.0, naca[1] / 10.0, "4d"
    elif len(naca) == 5:
        camber, t_c, series = naca[0] / 100.0, naca[2] / 10.0, "5d"
    else:
        camber, t_c, series = 0.0, 0.0, "?"
    return AoA, Uinf, Re, camber, t_c, series


d = np.load(os.path.join(ROOT, "04_results", "analysis", "physics_full_cache.npz"), allow_pickle=True)
Cl_all = d["Cl"]; Cd_all = d["Cd_p"]; AoA = d["AoA"]; Uinf = d["Uinf"]; tc = d["t_c"]
names_all = [str(x) for x in d["sim_names"]]
Re = Uinf / NU_REF
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
ft = set(man["full_train"]); fte = set(man["full_test"])
train_m = np.array([n in ft for n in names_all]); test_m = ~train_m
cam = np.array([decode_geom(n)[3] for n in names_all])
tr = train_m
print("n=%d  train=%d  test=%d  finite Cl=%.4f"
      % (len(names_all), train_m.sum(), test_m.sum(), np.isfinite(Cl_all).mean()))


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


C = {}
C["Cl_AoA"] = corr(AoA[tr], Cl_all[tr])
C["Cd_AoA"] = corr(AoA[tr], Cd_all[tr])
C["Cl_camber"] = corr(cam[tr], Cl_all[tr])
C["Cd_camber"] = corr(cam[tr], Cd_all[tr])
C["Cl_tc"] = corr(tc[tr], Cl_all[tr])
C["Cd_tc"] = corr(tc[tr], Cd_all[tr])
C["Cd_Re"] = corr(Re[tr], Cd_all[tr])
C["Cl_Re"] = corr(Re[tr], Cl_all[tr])
print("\ncorrelations (train %d):" % tr.sum())
for k, v in C.items():
    print("  %s = %.3f" % (k, v))


def partial_cl_camber(band=(0, 4)):
    m = (AoA[tr] >= band[0]) & (AoA[tr] < band[1])
    return corr(cam[tr][m], Cl_all[tr][m]) if m.sum() >= 10 else float("nan")


def partial_cd_tc(band=(0, 4)):
    m = (AoA[tr] >= band[0]) & (AoA[tr] < band[1])
    return corr(tc[tr][m], Cd_all[tr][m]) if m.sum() >= 10 else float("nan")


C["Cl_camber|AoA[0,4)"] = partial_cl_camber((0, 4))
C["Cd_tc|AoA[0,4)"] = partial_cd_tc((0, 4))

aoa_bins = [(-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 16)]
Cl_by_aoa = []
for lo, hi in aoa_bins:
    m = (AoA[tr] >= lo) & (AoA[tr] < hi)
    Cl_by_aoa.append((lo, hi, int(m.sum()),
                      float(Cl_all[tr][m].mean()) if m.sum() >= 5 else float("nan")))
print("\nCl mean by AoA bin (train):")
for lo, hi, n, mv in Cl_by_aoa:
    print("  [%d,%d) n=%d Cl=%.3f" % (lo, hi, n, mv))
finite = [x[3] for x in Cl_by_aoa if not np.isnan(x[3])]
mono_inc = all(finite[i] <= finite[i + 1] + 0.05 for i in range(len(finite) - 1))
print("Cl monotonic-increasing vs AoA? %s" % mono_inc)

target_Cl = np.quantile(Cl_all[tr], [0.1, 0.25, 0.5, 0.75, 0.9])
tol = 0.1
multi_sol = []
for t in target_Cl:
    m = np.abs(Cl_all - t) < tol
    multi_sol.append({"target_Cl": float(t), "n_samples": int(m.sum()),
                      "distinct_AoA": int(len(np.unique(np.round(AoA[m], 1)))),
                      "distinct_camber": int(len(np.unique(np.round(cam[m], 2)))) if m.sum() else 0})
print("\nmulti-solution probe (|Cl - target|<0.1):")
for ds in multi_sol:
    print("  Cl=%.2f  n=%d  distinct_AoA=%d  distinct_camber=%d"
          % (ds["target_Cl"], ds["n_samples"], ds["distinct_AoA"], ds["distinct_camber"]))

# Jacobian condition (per-target ridge, clean)
X = np.column_stack([AoA[tr], cam[tr], tc[tr]])
Xn = (X - X.mean(axis=0)) / X.std(axis=0)
lam_ridge = 1.0
Y = np.column_stack([Cl_all[tr], Cd_all[tr] * 100])
Yc = Y - Y.mean(axis=0)
ws = []
for j in range(2):
    b = Xn.T @ Yc[:, j]
    wj = np.linalg.solve(Xn.T @ Xn + lam_ridge * np.eye(3), b)
    ws.append(wj)
J = np.column_stack(ws)
_, sJ, _ = np.linalg.svd(J, full_matrices=False)
sJ = np.atleast_1d(sJ)
cond_J = float(sJ[0]) / float(sJ[-1])
print("\nJacobian cond number = %.3f (ridge lam=1, normalized vars)" % cond_J)
print("J:\n", J)
pred = Xn @ ws[0]
r2_cl = float(1.0 - ((Yc[:, 0] - pred) ** 2).sum() / (Yc[:, 0] ** 2).sum())
pred2 = Xn @ ws[1]
r2_cd = float(1.0 - ((Yc[:, 1] - pred2) ** 2).sum() / (Yc[:, 1] ** 2).sum())
print("linear R^2 = Cl %.3f, Cd*100 %.3f" % (r2_cl, r2_cd))

np.savez(os.path.join(OUT, "p2_inverse_precheck.npz"),
         Cl=Cl_all, Cd=Cd_all, AoA=AoA, Re=Re, camber=cam, t_c=tc, Uinf=Uinf,
         names=np.array(names_all), train_mask=train_m,
         corr=np.array([C["Cl_AoA"], C["Cd_AoA"], C["Cl_camber"], C["Cd_camber"],
                        C["Cl_tc"], C["Cd_tc"], C["Cd_Re"]]),
         cond_J=cond_J, r2=np.array([r2_cl, r2_cd]), J=np.ascontiguousarray(J))
json.dump({
    "n": len(names_all), "n_train": int(train_m.sum()), "n_test": int(test_m.sum()),
    "corr": C, "J_cond": cond_J, "linear_r2": {"Cl": r2_cl, "Cd_x100": r2_cd},
    "J": J.tolist(),
    "Cl_by_aoa_bin": [{"bin": [b[0], b[1]], "n": b[2], "Cl_mean": b[3]} for b in Cl_by_aoa],
    "Cl_monotonic_vs_AoA": bool(mono_inc),
    "multi_solution": multi_sol,
    "source": "P2-05 self-built; physics_full_cache.npz + manifest decode; train-only fit",
    "cross_check_ref": {"P1_05_Cl_AoA": 0.9419, "P1_05_Cd_AoA": 0.8347},
}, open(os.path.join(OUT, "05_inverse_precheck.json"), "w", encoding="utf-8"),
    ensure_ascii=False, indent=1)
print("\n-> p2_inverse_precheck.npz + 05_inverse_precheck.json")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
sc = ax.scatter(AoA, Cl_all, c=cam, cmap="viridis", s=16, alpha=0.7)
fig.colorbar(sc, ax=ax, label="camber (frac)")
ax.set_xlabel("AoA (deg)"); ax.set_ylabel("Cl")
ax.set_title("Cl vs AoA, colored by camber  (n=%d)" % len(names_all))
ax.grid(alpha=0.3)

ax = axes[1]
sc2 = ax.scatter(AoA, Cd_all * 100, c=tc, cmap="plasma", s=16, alpha=0.7)
fig.colorbar(sc2, ax=ax, label="t/c (frac)")
ax.set_xlabel("AoA (deg)"); ax.set_ylabel("Cd x 100")
ax.set_title("Cd vs AoA, colored by t/c")
ax.grid(alpha=0.3)

ax = axes[2]
ax.scatter(Re / 1e6, Cd_all * 100, c="k", s=12, alpha=0.4)
ax.set_xscale("log"); ax.set_xlabel("Re / 1e6 (log)")
ax.set_ylabel("Cd x 100")
ax.set_title("Cd vs Re (weakest coupling)")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_inverse_precheck.png"), bbox_inches="tight")
plt.close()
print("-> 05_inverse_precheck.png")
print("DONE_P2_P5_PRECHECK")

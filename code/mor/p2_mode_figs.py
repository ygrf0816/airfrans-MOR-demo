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


def decode_vars(nm):
    p = [float(x) for x in nm[len(PREFIX):].split("_")]
    Uinf, AoA = p[0], p[1]
    naca = p[2:]
    if len(naca) == 4:
        camber, t_c = naca[0] / 10.0, naca[1] / 10.0
    elif len(naca) == 5:
        camber, t_c = naca[0] / 100.0, naca[2] / 10.0
    else:
        camber, t_c = 0.0, 0.0
    return AoA, camber, t_c


d = np.load(os.path.join(OUT, "p2_wallcp_anchored.npz"), allow_pickle=True)
CP = d["CP"]; names = d["names"].tolist()
m = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
tr = [i for i, n in enumerate(names) if n in m["full_train"]]
Xtr = CP[tr]; mu = Xtr.mean(0); Ytr = Xtr - mu
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
alpha = Ytr @ Vt[:8].T

AoA_all = np.array([decode_vars(nm)[0] for nm in names])
cam_all = np.array([decode_vars(nm)[1] for nm in names])
tc_all = np.array([decode_vars(nm)[2] for nm in names])
AoA_tr = AoA_all[tr]; cam_tr = cam_all[tr]; tc_tr = tc_all[tr]

plt.rcParams["figure.dpi"] = 110
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) modal space: coeff1 vs coeff2 colored by AoA
ax = axes[0, 0]
sc = ax.scatter(alpha[:, 0], alpha[:, 1], c=AoA_tr, cmap="coolwarm", s=14, alpha=0.7)
cb = fig.colorbar(sc, ax=ax, label="AoA (deg)")
ax.set_title("Modal space: POD coeff (mode1, mode2)  [train, n=800]")
ax.set_xlabel("coeff mode 1"); ax.set_ylabel("coeff mode 2")

# (b) same plane colored by camber
ax = axes[0, 1]
sc2 = ax.scatter(alpha[:, 0], alpha[:, 1], c=cam_tr, cmap="viridis", s=14, alpha=0.7)
fig.colorbar(sc2, ax=ax, label="camber (frac)")
ax.set_title("Modal space: colored by camber")
ax.set_xlabel("coeff mode 1"); ax.set_ylabel("coeff mode 2")

# (c) coeff vs AoA for first 3 modes
ax = axes[1, 0]
for k in range(3):
    ax.plot(AoA_tr, alpha[:, k], marker="o", ms=3, lw=0.6, label="mode %d" % (k + 1), alpha=0.8)
ax.set_xlabel("AoA (deg)"); ax.set_ylabel("POD coefficient")
ax.set_title("Modal coefficients vs AoA (modes 1-3)")
ax.legend(); ax.grid(alpha=0.3)

# (d) mode profiles (Cp shape) for first 8 modes
ax = axes[1, 1]
s_grid = np.linspace(0, 1, 256)
for k in range(8):
    ax.plot(s_grid, Vt[k], lw=1.4, label="mode %d" % (k + 1))
ax.set_xlabel("s (normalized arc-length, LE->TE->LE top-first)"); ax.set_ylabel("POD mode (Cp shape)")
ax.set_title("First 8 POD mode shapes (Cp profile)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

plt.tight_layout()
fp = os.path.join(OUT, "03_mode_physics.png")
plt.savefig(fp, bbox_inches="tight")
plt.close()
print("-> %s" % fp)

# also print a compact "mode->physics" table for the md
dom = {}
for k in range(8):
    corrs = {
        "AoA": float(np.corrcoef(alpha[:, k], AoA_tr)[0, 1]),
        "camber": float(np.corrcoef(alpha[:, k], cam_tr)[0, 1]),
        "t_c": float(np.corrcoef(alpha[:, k], tc_tr)[0, 1]),
    }
    dom[k + 1] = {"corr": corrs, "dominant": max(corrs, key=lambda x: abs(corrs[x]))}
print("\nmode->physics:")
for k in range(1, 9):
    c = dom[k]["corr"]
    print("  mode %d: AoA=%.3f camber=%.3f t_c=%.3f  dominant=%s"
          % (k, c["AoA"], c["camber"], c["t_c"], dom[k]["dominant"]))

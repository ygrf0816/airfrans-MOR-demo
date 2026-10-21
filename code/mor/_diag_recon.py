import sys, os, json
import numpy as np, pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from matplotlib.path import Path as MplPath
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATASET = os.path.join(ROOT, "02_dataset", "Dataset")
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
sel = [sorted(man["full_train"])[0]]
NX, NY = 256, 128
xs = np.linspace(-0.5, 1.5, NX); ys = np.linspace(-0.5, 0.5, NY)
GX, GY = np.meshgrid(xs, ys, indexing="xy"); GRID = np.column_stack([GX.ravel(), GY.ravel()])
sim = sel[0]
m = pv.read(os.path.join(DATASET, sim, sim + "_internal.vtu")); P = m.points
p = m.point_data["p"]; Umag = np.hypot(m.point_data["U"][:, 0], m.point_data["U"][:, 1])
ip_p = LinearNDInterpolator((P[:, 0], P[:, 1]), p, fill_value=np.nan)
ip_u = LinearNDInterpolator((P[:, 0], P[:, 1]), Umag, fill_value=np.nan)
pp = ip_p(GRID[:, 0], GRID[:, 1]); pu = ip_u(GRID[:, 0], GRID[:, 1])
print("p native: min=%.1f max=%.1f mean=%.1f rms=%.1f" % (p.min(), p.max(), p.mean(), np.sqrt(np.mean(p**2))))
print("p grid:   finite=%d nan=%d min=%.1f max=%.1f" % (np.isfinite(pp).sum(), np.isnan(pp).sum(), np.nanmin(pp), np.nanmax(pp)))
print("U native: min=%.2f max=%.2f mean=%.2f rms=%.2f" % (Umag.min(), Umag.max(), Umag.mean(), np.sqrt(np.mean(Umag**2))))
tree = cKDTree(GRID)
idx, _ = tree.query(P[:, :2], k=1); idx = idx.astype(np.int64)
ok = np.isfinite(pp[idx])
print("back-project: %d/%d native pts have finite grid value" % (ok.sum(), P.shape[0]))
pr = pp[idx][ok] - p[ok]; ur = pu[idx][ok] - Umag[ok]
print("p resid rms=%.1f ; RMSE/fieldrms = %.4f" % (np.sqrt(np.mean(pr**2)), np.sqrt(np.mean(pr**2))/np.sqrt(np.mean(p**2))))
print("u resid rms=%.3f ; RMSE/fieldrms = %.4f" % (np.sqrt(np.mean(ur**2)), np.sqrt(np.mean(ur**2))/np.sqrt(np.mean(Umag**2))))
# where is the error largest? native |resid| percentiles
print("p resid abs percentiles p50=%.3f p95=%.3f p99=%.3f max=%.3f" % tuple(np.percentile(np.abs(pr), [50, 95, 99, 100])))
print("u resid abs percentiles p50=%.4f p95=%.4f p99=%.4f max=%.4f" % tuple(np.percentile(np.abs(ur), [50, 95, 99, 100])))

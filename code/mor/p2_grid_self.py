"""P2-01: self-build 256x128 common crop-box grid + union wall mask + per-sample
interp time + mask fraction + interp reconstruction error + p-variance-removed.
All numbers computed from 02_dataset/Dataset/*_internal.vtu by our own code.
overnight npz / P2_REFERENCE are cross-check only (magnitude anchors), never the data source.
Run chunked:  python p2_grid_self.py --start S --count 18   (each ~45s, under 120s cap)
"""
import os, sys, json, time, argparse, glob
import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from matplotlib.path import Path as MplPath
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATASET = os.path.join(ROOT, "02_dataset", "Dataset")
OUT = os.path.join(ROOT, "04_results", "mor")
os.makedirs(OUT, exist_ok=True)
MAN = os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json")
if not os.path.exists(MAN):
    MAN = os.path.join(ROOT, "03_code", "probe", "manifest.json")
man = json.load(open(MAN, encoding="utf-8"))

NX, NY = 256, 128
X0, X1, Y0, Y1 = -0.5, 1.5, -0.5, 0.5
xs = np.linspace(X0, X1, NX); ys = np.linspace(Y0, Y1, NY)
GX, GY = np.meshgrid(xs, ys, indexing="xy")
GRID = np.column_stack([GX.ravel(), GY.ravel()])
NG = GRID.shape[0]
sel = sorted(man["full_train"])[:300] + sorted(man["full_test"])[:100]
N = len(sel)
print("crop box x[%.2f,%.2f] y[%.2f,%.2f]  grid %dx%d=%d pts  sel=%d"
      % (X0, X1, Y0, Y1, NX, NY, NG, N), flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=40)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()

if args.merge:
    import glob
    chunks = sorted(glob.glob(os.path.join(OUT, "p2_grid_self_c*.npz")))
    ch = [np.load(c, allow_pickle=True) for c in chunks]
    Pm_all = np.vstack([c["P"] for c in ch]); Um_all = np.vstack([c["U"] for c in ch])
    V_all = np.vstack([c["valid"] for c in ch])
    names_all = np.concatenate([c["names"] for c in ch])
    wts = np.array([c["count"] for c in ch], dtype=float)
    def wavg(k): return float(np.average([c[k] for c in ch], weights=wts))
    np.savez(os.path.join(OUT, "p2_grid_self.npz"),
             P=np.ascontiguousarray(Pm_all), U=np.ascontiguousarray(Um_all),
             valid=np.ascontiguousarray(V_all), names=names_all,
             xs=np.ascontiguousarray(ch[0]["xs"]), ys=np.ascontiguousarray(ch[0]["ys"]),
             grid_x=np.ascontiguousarray(ch[0]["grid_x"]), grid_y=np.ascontiguousarray(ch[0]["grid_y"]),
             n_train=300, interp_s_per_sample=wavg("interp_s_per_sample"),
             read_s_per_sample=wavg("read_s_per_sample"),
             p_recon_self=wavg("p_recon_chunk"), u_recon_self=wavg("u_recon_chunk"),
             p_recon_rel_self=wavg("p_recon_rel_chunk"), u_recon_rel_self=wavg("u_recon_rel_chunk"),
             p_var_removed_median=float(np.nanmedian([c["p_var_removed_median"] for c in ch])),
             mask_frac_median=float(np.nanmean((~V_all).mean(axis=1))),
             total_grid=NG, nx=NX, ny=NY,
             crop_box_x0=X0, crop_box_x1=X1, crop_box_y0=Y0, crop_box_y1=Y1,
             note="self-computed from internal.vtu; crop -0.5..1.5 x -0.5..0.5; "
                  "overnight/P2_REFERENCE cross-check only")
    print("-> p2_grid_self.npz  (merged %d samples, %d grid pts, %d chunks)"
          % (len(names_all), NG, len(ch)))
    print("DONE_P2_GRID_SELF")
    sys.exit(0)

# --- build union wall mask over ALL train+test airfoils in sel (the 400-sample P2-02 spec) ---
sel_chunk = sel[args.start:args.start+args.count]
print("sel_chunk = %d samples; building union wall mask over ALL %d airfoils in sel" % (len(sel_chunk), len(sel)), flush=True)
union = np.zeros(NG, dtype=bool)
t0u = time.time()
for sim in sel_chunk:
    mm = pv.read(os.path.join(DATASET, sim, sim + "_aerofoil.vtp"))
    poly = np.vstack([mm.points[:, :2], mm.points[:1, :2]])
    union |= MplPath(poly).contains_points(GRID)
print("union wall mask over %d airfoils: removed frac=%.4f valid=%.4f (%.1fs)" % (len(sel), union.mean(), (~union).mean(), time.time()-t0u), flush=True)

Pm = np.full((len(sel_chunk), NG), np.nan)
Um = np.full((len(sel_chunk), NG), np.nan)
V = np.zeros((len(sel_chunk), NG), dtype=bool)
t_read = np.zeros(len(sel_chunk)); t_interp = np.zeros(len(sel_chunk))
p_var_removed = np.zeros(len(sel_chunk))
tree_g = cKDTree(GRID)
p_recon = np.full(len(sel_chunk), np.nan)
u_recon = np.full(len(sel_chunk), np.nan)
p_recon_rel = np.full(len(sel_chunk), np.nan)
u_recon_rel = np.full(len(sel_chunk), np.nan)
t1 = time.time()
for i, sim in enumerate(sel_chunk):
    fp = os.path.join(DATASET, sim, sim + "_internal.vtu")
    t0r = time.time(); m = pv.read(fp); P = m.points; t_read[i] = time.time() - t0r
    p = m.point_data["p"]
    Umag = np.hypot(m.point_data["U"][:, 0], m.point_data["U"][:, 1])
    t0i = time.time()
    ip_p = LinearNDInterpolator((P[:, 0], P[:, 1]), p, fill_value=np.nan)
    ip_u = LinearNDInterpolator((P[:, 0], P[:, 1]), Umag, fill_value=np.nan)
    pp = ip_p(GRID[:, 0], GRID[:, 1]); pu = ip_u(GRID[:, 0], GRID[:, 1])
    mask = np.isfinite(pp) & np.isfinite(pu) & ~union
    Pm[i] = pp; Um[i] = pu; V[i] = mask
    pv_rm = pp[union & np.isfinite(pp)]
    if pv_rm.size:
        pv_all = np.concatenate([pp[~union & np.isfinite(pp)], pv_rm])
        p_var_removed[i] = float(np.var(pv_rm - pv_all.mean()) / (np.var(pv_all - pv_all.mean()) + 1e-30))
    idx, _ = tree_g.query(P[:, :2], k=1)
    idx = idx.astype(np.int64)
    ok = mask[idx]
    p_resid = pp[idx][ok] - p[ok]
    u_resid = pu[idx][ok] - Umag[ok]
    p_recon[i] = float(np.sqrt(np.mean(p_resid**2)))
    u_recon[i] = float(np.sqrt(np.mean(u_resid**2)))
    p_recon_rel[i] = p_recon[i] / (np.sqrt(np.mean(p**2)) + 1e-30)
    u_recon_rel[i] = u_recon[i] / (np.sqrt(np.mean(Umag**2)) + 1e-30)
    t_interp[i] = time.time() - t0i
    if (i + 1) % 25 == 0:
        print("  %d/%d read_sum=%.0fs interp_sum=%.0fs" % (i + 1, len(sel_chunk), t_read.sum(), t_interp.sum()), flush=True)

np.savez(os.path.join(OUT, "p2_grid_self_c0.npz"),
         P=np.ascontiguousarray(Pm), U=np.ascontiguousarray(Um),
         valid=np.ascontiguousarray(V), names=np.array(sel_chunk),
         xs=np.ascontiguousarray(xs), ys=np.ascontiguousarray(ys),
         grid_x=np.ascontiguousarray(GRID[:, 0]), grid_y=np.ascontiguousarray(GRID[:, 1]),
         interp_s_per_sample=float(t_interp.mean()), read_s_per_sample=float(t_read.mean()),
         p_recon_chunk=float(np.nanmean(p_recon)), u_recon_chunk=float(np.nanmean(u_recon)),
         p_recon_rel_chunk=float(np.nanmean(p_recon_rel)), u_recon_rel_chunk=float(np.nanmean(u_recon_rel)),
         p_var_removed_median=float(np.nanmedian(p_var_removed)),
         union_mask_frac=float(union.mean()),
         total_grid=NG, nx=NX, ny=NY, count=len(sel_chunk))
print("-> p2_grid_self_c0.npz  (%.0fs for %d samples)" % (time.time() - t1, len(sel_chunk)))
print("  interp/sample=%.2fs read/sample=%.2fs" % (t_interp.mean(), t_read.mean()))
print("  union_mask_frac=%.4f  p_RMSE=%.1f p_relRMSE=%.4f u_RMSE=%.4f u_relRMSE=%.4f  p_var_removed_med=%.4f"
      % (float(union.mean()), np.nanmean(p_recon), np.nanmean(p_recon_rel),
         np.nanmean(u_recon), np.nanmean(u_recon_rel), float(np.nanmedian(p_var_removed))))
print("DONE_P2_GRID_400")

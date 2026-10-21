"""Builder: writes 03_code/p1/p2_grid_self.py (crop-box chunked P2-01) and verifies AST.
Run: python 03_code/p1/_write_p2_grid_self.py
"""
import os, ast, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = r'''import os, sys, json, time, argparse, glob
import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
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
sel = sorted(man["full_train"])[:200] + sorted(man["full_test"])[:100]
N = len(sel)
print("crop box x[%.2f,%.2f] y[%.2f,%.2f]  grid %dx%d=%d pts  sel=%d" % (X0, X1, Y0, Y1, NX, NY, NG, N), flush=True)
ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=40)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
if args.merge:
    chunks = sorted(glob.glob(os.path.join(OUT, "p2_grid_self_c*.npz")))
    ch = [np.load(c, allow_pickle=True) for c in chunks]
    Pm_all = np.vstack([c["P"] for c in ch]); Um_all = np.vstack([c["U"] for c in ch])
    V_all = np.vstack([c["valid"] for c in ch]); names_all = np.concatenate([c["names"] for c in ch])
    wts = np.array([c["count"] for c in ch], dtype=float)
    def wavg(key): return float(np.average([c[key] for c in ch], weights=wts))
    np.savez(os.path.join(OUT, "p2_grid_self.npz"),
         P=np.ascontiguousarray(Pm_all), U=np.ascontiguousarray(Um_all),
         valid=np.ascontiguousarray(V_all), names=names_all,
         xs=np.ascontiguousarray(ch[0]["xs"]), ys=np.ascontiguousarray(ch[0]["ys"]),
         grid_x=np.ascontiguousarray(ch[0]["grid_x"]), grid_y=np.ascontiguousarray(ch[0]["grid_y"]),
         n_train=200, interp_s_per_sample=wavg("interp_s_per_sample"),
         read_s_per_sample=wavg("read_s_per_sample"),
         p_recon_self=wavg("p_recon_chunk"), u_recon_self=wavg("u_recon_chunk"),
         p_var_removed_median=float(np.median([c["p_var_removed_median"] for c in ch])),
         mask_frac_median=float(np.nanmean((~V_all).mean(axis=1))),
         total_grid=NG, nx=NX, ny=NY, crop_box_x0=X0, crop_box_x1=X1, crop_box_y0=Y0, crop_box_y1=Y1,
         note="self-computed from internal.vtu; crop -0.5..1.5 x -0.5..0.5; overnight/P2_REFERENCE cross-check only.")
    print("-> p2_grid_self.npz (merged %d samples, %d grid pts, %d chunks)" % (len(names_all), NG, len(ch)))
    print("DONE_P2_GRID_SELF")
    sys.exit(0)
sel_chunk = sel[args.start:args.start + args.count]
print("chunk [%d,%d) = %d samples" % (args.start, args.start + len(sel_chunk), len(sel_chunk)), flush=True)
Pm = np.full((len(sel_chunk), NG), np.nan)
Um = np.full((len(sel_chunk), NG), np.nan)
V = np.zeros((len(sel_chunk), NG), dtype=bool)
t_read = np.zeros(len(sel_chunk)); t_interp = np.zeros(len(sel_chunk))
p_var_removed = np.zeros(len(sel_chunk))
tree_g = cKDTree(GRID)
p_recon = np.full(len(sel_chunk), np.nan); u_recon = np.full(len(sel_chunk), np.nan)
t1 = time.time()
for i, sim in enumerate(sel_chunk):
    fp = os.path.join(DATASET, sim, sim + "_internal.vtu")
    t0r = time.time(); m = pv.read(fp); P = m.points; t_read[i] = time.time() - t0r
    p = m.point_data["p"]; Umag = np.hypot(m.point_data["U"][:, 0], m.point_data["U"][:, 1])
    sd = m.point_data["implicit_distance"]
    t0i = time.time()
    ip_p = LinearNDInterpolator((P[:, 0], P[:, 1]), p, fill_value=np.nan)
    ip_u = LinearNDInterpolator((P[:, 0], P[:, 1]), Umag, fill_value=np.nan)
    ip_s = LinearNDInterpolator((P[:, 0], P[:, 1]), sd, fill_value=np.nan)
    pp = ip_p(GRID[:, 0], GRID[:, 1]); pu = ip_u(GRID[:, 0], GRID[:, 1]); ps = ip_s(GRID[:, 0], GRID[:, 1])
    inside = (ps > 0) & np.isfinite(ps)
    mask = np.isfinite(pp) & np.isfinite(pu) & ~inside
    Pm[i] = pp; Um[i] = pu; V[i] = mask
    pv_rm = pp[~mask]
    if pv_rm.size:
        pv_all = np.concatenate([pp[mask], pv_rm])
        p_var_removed[i] = float(np.var(pv_rm - pv_all.mean()) / (np.var(pv_all - pv_all.mean()) + 1e-30))
    idx, _ = tree_g.query(P[:, :2], k=1)
    ok = mask[idx]
    p_recon[i] = float(np.linalg.norm(pp[idx][ok] - p[ok]) / (np.linalg.norm(p - p.mean()) + 1e-30))
    u_recon[i] = float(np.linalg.norm(pu[idx][ok] - Umag[ok]) / (np.linalg.norm(Umag - Umag.mean()) + 1e-30))
    t_interp[i] = time.time() - t0i
    if (i + 1) % 10 == 0:
        print("  %d/%d read_sum=%.0fs interp_sum=%.0fs mask~%.1f%%" % (i + 1, len(sel_chunk), t_read.sum(), t_interp.sum(), mask.mean() * 100), flush=True)
np.savez(os.path.join(OUT, "p2_grid_self_c%d.npz" % args.start),
     P=np.ascontiguousarray(Pm), U=np.ascontiguousarray(Um),
     valid=np.ascontiguousarray(V), names=np.array(sel_chunk),
     xs=np.ascontiguousarray(xs), ys=np.ascontiguousarray(ys),
     grid_x=np.ascontiguousarray(GRID[:, 0]), grid_y=np.ascontiguousarray(GRID[:, 1]),
     interp_s_per_sample=float(t_interp.mean()), read_s_per_sample=float(t_read.mean()),
     p_recon_chunk=float(np.nanmean(p_recon)), u_recon_chunk=float(np.nanmean(u_recon)),
     p_var_removed_median=float(np.nanmedian(p_var_removed)),
     total_grid=NG, nx=NX, ny=NY, start=args.start, count=len(sel_chunk))
print("-> p2_grid_self_c%d.npz  (%.0fs for %d)" % (args.start, time.time() - t1, len(sel_chunk)))
print("  interp/sample=%.2fs read/sample=%.2fs" % (t_interp.mean(), t_read.mean()))
print("  mask_frac(removed)=%.4f  p_recon=%.4f u_recon=%.4f  p_var_removed_med=%.4f"
      % (float(np.nanmean((~V).mean(axis=1))), np.nanmean(p_recon), np.nanmean(u_recon), float(np.nanmedian(p_var_removed))))
print("DONE_CHUNK_START%d" % args.start)
'''

ast.parse(SRC)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p2_grid_self.py')
open(out, 'w', encoding='utf-8').write(SRC)
print('wrote', out, len(SRC), 'chars, AST OK')

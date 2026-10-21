import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
path = "03_code/p1/p2_grid_self.py"
s = open(path, encoding="utf-8").read()

# 1) import MplPath
anchor = "from scipy.spatial import cKDTree"
add = "\nfrom matplotlib.path import Path as MplPath\n"
if "MplPath" not in s:
    s = s.replace(anchor, anchor + add, 1)

# 2) union wall mask builder + call (insert before sel_chunk line)
union_block = '''
def _build_union_wall_mask(sims, grid, dataset):
    union = np.zeros(grid.shape[0], dtype=bool)
    for sim in sims:
        mm = pv.read(os.path.join(dataset, sim, sim + "_aerofoil.vtp"))
        poly = np.vstack([mm.points[:, :2], mm.points[:1, :2]])
        union |= MplPath(poly).contains_points(grid)
    return union
UNION_MASK = _build_union_wall_mask(sel, GRID, DATASET)
print("union wall mask over %d airfoils: removed grid frac = %.4f (%d/%d)  valid = %.4f"
      % (len(sel), UNION_MASK.mean(), int(UNION_MASK.sum()), NG, (~UNION_MASK).mean()), flush=True)
'''
anchor2 = "sel_chunk = sel[args.start:args.start + args.count]"
if "UNION_MASK" not in s:
    s = s.replace(anchor2, union_block + "\n" + anchor2, 1)

# 3) fix the wrong implicit_distance-based mask
old_mask = """    pp = ip_p(GRID[:, 0], GRID[:, 1]); pu = ip_u(GRID[:, 0], GRID[:, 1]); ps = ip_s(GRID[:, 0], GRID[:, 1])
    inside = (ps > 0) & np.isfinite(ps)
    mask = np.isfinite(pp) & np.isfinite(pu) & ~inside"""
new_mask = """    pp = ip_p(GRID[:, 0], GRID[:, 1]); pu = ip_u(GRID[:, 0], GRID[:, 1])
    mask = np.isfinite(pp) & np.isfinite(pu) & ~UNION_MASK"""
if old_mask in s:
    s = s.replace(old_mask, new_mask, 1)

# 4) also drop now-unused implicit_distance interpolator line (keep for timing completeness? remove to save time)
old_sd = "    ip_s = LinearNDInterpolator((P[:, 0], P[:, 1]), sd, fill_value=np.nan)\n"
if old_sd in s:
    s = s.replace(old_sd, "", 1)
old_sd_var = "    sd = m.point_data[\"implicit_distance\"]\n"
if old_sd_var in s:
    s = s.replace(old_sd_var, "", 1)

open(path, "w", encoding="utf-8").write(s)
import py_compile
py_compile.compile(path, doraise=True)
print("patched p2_grid_self.py (union wall mask + MplPath + no implicit_distance); compile OK")
print("UNION_MASK block present:", "UNION_MASK" in s)
print("mask line present:", "mask = np.isfinite(pp) & np.isfinite(pu) & ~UNION_MASK" in s)

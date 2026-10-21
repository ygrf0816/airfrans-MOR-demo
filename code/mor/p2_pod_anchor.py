"""P2-02: wall-Cp POD with LEADING-EDGE-ANCHORED arc-length (self-computed, not node-min).
P2_REFERENCE §5.7.2: s must anchor at the leading edge (argmin x) + front half over top surface,
else the same (s,n) cell points to a different physical location across samples -> rank inflation
(observed r99 24->7 when fixed). Rebuilds p2_wallcp_self_anched.npz + runs train-only POD.
Data source = 02_dataset/Dataset/*_aerofoil.vtp (self-read); overnight wallcp.npz = magnitude cross-check only.
"""
import os, sys, json, time
import numpy as np
import xml.etree.ElementTree as ET
import zlib, struct
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'probe')))
import vtkxml2 as vx
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATASET = os.path.join(ROOT, "02_dataset", "Dataset")
OUT = os.path.join(ROOT, "04_results", "mor")
os.makedirs(OUT, exist_ok=True)
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
ALL = sorted(set().union(*[set(v) for v in man.values()]))
assert len(ALL) == 1000


def Uinf(sim):
    return float(sim.split("_")[2])


def _lines(vtp):
    root = ET.parse(vtp).getroot()
    piece = next(e for e in root.iter() if e.tag == 'Piece')
    conn = off = None
    for da in piece.find('Lines').findall('DataArray'):
        raw = vx.b64_stream(da.text or ''); nb = struct.unpack_from('<I', raw, 0)[0]
        pos, o = 12, bytearray()
        for _ in range(max(1, nb)):
            cl = struct.unpack_from('<I', raw, pos)[0]; pos += 4
            o += zlib.decompress(raw[pos:pos + cl]); pos += cl
        arr = np.frombuffer(bytes(o), dtype=np.dtype(vx._DT[da.get('type')]))
        if da.get('Name') == 'connectivity':
            conn = arr
        elif da.get('Name') == 'offsets':
            off = arr
    return conn, off


def wall_cp_anchor(sim, n=256):
    """Leading-edge-anchored, top-surface-first Cp(s) on normalized arc-length s in [0,1]."""
    from wall_force import _cycle_from_edges
    fp = os.path.join(DATASET, sim, sim + "_aerofoil.vtp")
    pts, pdata, *_ = vx.read_vtkxml(fp)
    conn, off = _lines(fp)
    loop = _cycle_from_edges(conn, off, pts[:, :2])
    N = len(loop)
    X = np.vstack([pts[loop, :2], pts[loop[0], :2]])   # N+1 pts (closed)
    p_wall = pdata['p'][loop]  # N values
    # leading edge = argmin x
    le = int(np.argmin(X[:, 0]))
    # rotate so LE is index 0, and front half goes over the TOP surface
    Xr = np.roll(X, -le, axis=0)  # still N+1, but index 0 = LE
    # top-first: if Xr[N//2].y < Xr[N//2-1].y the walk is bottom-first -> reverse
    Nr = len(Xr)
    reverse = Xr[Nr // 2, 1] < Xr[Nr // 2 - 1, 1]
    if reverse:
        Xr = Xr[::-1]
    pr = np.roll(p_wall, -le)  # N values
    if reverse:
        pr = pr[::-1]
    # Now Xr has N+1 entries (last=first=LE), pr has N entries.
    # Build s on the first N points only (drop the closing duplicate):
    Xr_N = Xr[:N]  # N points
    seg = np.linalg.norm(np.diff(np.vstack([Xr_N, Xr_N[:1]]), axis=0), axis=1)  # N segments (closed)
    s = np.concatenate([[0], np.cumsum(seg)])
    s = s[:-1]  # N values, matching pr
    s = s / s[-1]
    q = 0.5 * Uinf(sim) ** 2
    cp = pr / q
    return np.interp(np.linspace(0, 1, n), s, cp)


# build cache
N = 1000
CP = np.zeros((N, 256))
t0 = time.time()
for i, sim in enumerate(ALL):
    CP[i] = wall_cp_anchor(sim)
    if (i + 1) % 100 == 0:
        print("  anchor wallcp %d/1000 %.0fs" % (i + 1, time.time() - t0), flush=True)
np.savez(os.path.join(OUT, "p2_wallcp_anchored.npz"), CP=CP, names=np.array(ALL))
print("-> %s/p2_wallcp_anchored.npz  (%.0fs)" % (OUT, time.time() - t0), flush=True)

# ---- POD: train-only fit, test report ----
tr = [i for i, n in enumerate(ALL) if n in man["full_train"]]
te = [i for i, n in enumerate(ALL) if n in man["full_test"]]
Xtr = CP[tr]; Xte = CP[te]
mu = Xtr.mean(0); Ytr = Xtr - mu
U, S, Vt = np.linalg.svd(Ytr, full_matrices=False)
en = np.cumsum(S**2) / (S**2).sum()
r99 = int(np.searchsorted(en, 0.99, side="right"))
r999 = int(np.searchsorted(en, 0.999, side="right"))
P = Vt.T; Yte = Xte - mu
res = {"n_train": len(tr), "n_test": len(te), "S_first12": S[:12].tolist(),
       "r99": r99, "r999": r999, "train_relL2": {}, "test_relL2": {},
       "source": "self-computed leading-edge-anchored Cp (P2-02), full_train/full_test split",
       "cross_check_ref": {"overnight_r99": 6, "P2_REFERENCE_r99": 7}}
for r in [2, 4, 8, 16, 32, 64]:
    res["train_relL2"][str(r)] = float(np.sqrt(max(0.0, 1.0 - min(en[r - 1], 1.0))))
    Rr = (Yte @ P[:, :r]) @ P[:, :r].T
    res["test_relL2"][str(r)] = float(np.linalg.norm(Yte - Rr) / np.linalg.norm(Yte))
json.dump(res, open(os.path.join(OUT, "02_pod_energy.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("S[0:6]:", np.round(S[:6], 3))
print("r99=%d r999=%d" % (r99, r999))
for r in [2, 4, 8, 16, 32, 64]:
    print("  r=%2d train_relL2=%.4e test_relL2=%.4e" % (r, res["train_relL2"][str(r)], res["test_relL2"][str(r)]))
print("DONE_P2_POD_ANCHORED")

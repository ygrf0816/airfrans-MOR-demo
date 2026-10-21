import sys, os, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
A = 340.3  # speed of sound ~298K, m/s (same convention P1-07b used)
# local Mach = Umag_max / A ; field_scalars_full.npz holds per-sim scalars incl Umag_max
sc = np.load(os.path.join(ROOT, "04_results", "analysis", "field_scalars_full_stats.npz"),
             allow_pickle=True)
print("field_scalars_full keys:", list(sc.files)[:40])
# find the Umag_max / Mach arrays
arrs = {k: sc[k] for k in sc.files}
# locate per-sample arrays of length 1000
cand = {}
for k, a in arrs.items():
    if isinstance(a, np.ndarray) and a.ndim == 1 and a.size == 1000:
        cand[k] = a
print("len-1000 arrays:", list(cand.keys()))
if "names" in cand:
    names = cand["names"]
elif "sim_names" in cand:
    names = cand["sim_names"]
else:
    names = None
# find Umag_max (key is _U_mag_max in this build)
um = None
for k in cand:
    kk = k.lower().replace("_", "")
    if "umagmax" in kk:
        um = cand[k]; print("using Umag_max key:", k); break
if um is None:
    print("no Umag_max key found in len-1000 arrays; keys:", list(cand.keys()))
    # fallback: recompute from dataset? Too slow; instead scan all keys for any 'max'
    maxkeys = [k for k in sc.files if "max" in k.lower()]
    print("all *max* keys:", maxkeys)
    sys.exit(0)
mach_local = um / A
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
nm_set = set()
# align names order
# field_scalars names order may equal sorted(full train+test); build split membership by name
# We need per-sample split. If names present, use it.
# align names order
if "_sim_names" in cand:
    names = cand["_sim_names"]
elif "sim_names" in cand:
    names = cand["sim_names"]
else:
    names = None
names_l = [str(x) for x in names]
full_test = set(man["full_test"]); reynolds_test = set(man["reynolds_test"])
aoa_test = set(man["aoa_test"]); full_train = set(man["full_train"]); aoa_train = set(man["aoa_train"])
for split, s in [("full_test", full_test), ("reynolds_test", reynolds_test),
                 ("aoa_test", aoa_test), ("aoa_train", aoa_train), ("full_train", full_train)]:
    m = np.array([n in s for n in names_l])
    if m.sum() == 0:
        print("%s: no samples in this array (n=%d)" % (split, m.sum())); continue
    mv = mach_local[m]
    over = int((mv > 0.7).sum())
    over1 = int((mv > 1.0).sum())
    print("%s n=%d  localMach>0.7=%d (%.1f%%)  >1.0=%d  max_localMach=%.3f"
          % (split, int(m.sum()), over, 100 * over / m.sum(), over1, mv.max()))

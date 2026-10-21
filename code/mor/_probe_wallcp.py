import sys, json, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
d = np.load(os.path.join(ROOT, "04_results", "analysis", "p2_wallcp_self.npz"), allow_pickle=True)
print("wallcp keys:", list(d.files))
print("CP shape:", d["CP"].shape, "names n:", len(d["names"]))
man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))
ft = set(man["full_train"]); fte = set(man["full_test"])
nm = set(d["names"].tolist())
print("full_train=%d full_test=%d overlap=%d" % (len(ft), len(fte), len(ft & fte)))
print("cache n=%d in_train=%d in_test=%d" % (len(nm), len(nm & ft), len(nm & fte)))
print("sim0 CP range %.3f .. %.3f" % (d["CP"][0].min(), d["CP"][0].max()))
# how many train cp vectors are finite
tr_idx = [i for i, n in enumerate(d["names"].tolist()) if n in ft]
te_idx = [i for i, n in enumerate(d["names"].tolist()) if n in fte]
print("train cp rows=%d test cp rows=%d" % (len(tr_idx), len(te_idx)))

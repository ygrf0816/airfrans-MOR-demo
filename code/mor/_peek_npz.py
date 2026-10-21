import sys, os
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def show(path, name):
    print("=== %s (%s) ===" % (name, path))
    d = np.load(path, allow_pickle=True)
    for k in d.files:
        v = d[k]
        try:
            print("  %-24s %-16s %s" % (k, str(v.shape), type(v).__name__))
        except Exception:
            print("  %-24s (object)" % k)
    print()

base = "04_results/mor"
show(os.path.join(base, "p2_wallcp_anchored.npz"), "P2 壁面Cp锚定")
show(os.path.join(base, "p2_pod3_modes.npz"), "P2-03 POD模态")
show(os.path.join(base, "p2_grid_self_c0.npz"), "P3 体积场网格缓存")

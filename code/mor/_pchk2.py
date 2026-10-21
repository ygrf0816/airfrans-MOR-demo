import sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = np.load("04_results/analysis/p2_wallcp_self.npz", allow_pickle=True)
CP = d["CP"]
print("CP shape", CP.shape, "finite frac", np.isfinite(CP).mean())
print("per-sample Cp std: p05=%.3f p50=%.3f p95=%.3f" % tuple(np.percentile(CP.std(1), [5, 50, 95])))
print("sim0 name:", d["names"][0])
print("sim0 CP p10/p50/p90:", np.round(np.percentile(CP[0], [10, 50, 90]), 3))
# global cp magnitude
print("global CP p1/p50/p99:", np.round(np.percentile(CP, [1, 50, 99]), 3))
# how many cp near zero vs large
print("frac |CP|>3:", (np.abs(CP) > 3).mean(), " frac |CP|>0.5:", (np.abs(CP) > 0.5).mean())

import sys, os, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1common import PREFIX

man = json.load(open(os.path.join(ROOT, "02_dataset", "Dataset", "manifest.json"), encoding="utf-8"))


def mach(nm):
    return float(nm[len(PREFIX):].split("_")[0]) / 340.3


for k in ["full_test", "aoa_test", "reynolds_test"]:
    arr = [mach(n) for n in man[k]]
    hi = [n for n in man[k] if mach(n) > 0.7]
    print("%s n=%d  Mach>0.7 n=%d (%.1f%%)  maxMach=%.3f"
          % (k, len(arr), len(hi), 100 * len(hi) / len(arr), max(arr)))

# is aoa_test a subset of full_test?
ft, at = set(man["full_test"]), set(man["aoa_test"])
print("\naoa_test % full_test ? subset=%d  overlap=%d  aoa_test_only=%d"
      % (len(at & ft), len(at & ft), len(at - ft)))
# high-Mach samples in aoa_test: are they in the cache?
d = np.load(os.path.join(ROOT, "04_results", "mor", "p2_wallcp_anchored.npz"), allow_pickle=True)
cache = set(d["names"].tolist())
hi_at = [n for n in man["aoa_test"] if mach(n) > 0.7]
print("aoa_test Mach>0.7 n=%d, in cache=%d" % (len(hi_at), len(set(hi_at) & cache)))

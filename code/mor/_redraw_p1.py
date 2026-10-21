"""Re-draw P1 report figures that had CJK tofu-box labels.
Direct sequential import (no subprocess -> avoids the reader-thread crash).
Each target script now does `import plotfont` (CJK font) -> clean figures.
Run from 03_code/p1 so the scripts' relative paths resolve.
"""
import sys, os, time, importlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

# P1 report-figure scripts carrying CJK title/label/legend.
jobs = [
    "p1_inventory",       # 01_inventory.png
    "p1_params",          # 02_corr_matrix.png
    "p1_mesh_sample",     # 03_mesh_geometry.png
    "p1_field_aggregate", # 04_field_stats.png
    "p1_physics_aggregate",# 05_physics.png
    "p1_splits",          # 06_split_overlap.png
]
for name in jobs:
    f = name + ".py"
    if not os.path.exists(f):
        print("SKIP %s (missing)" % f); continue
    # reload clean each time (avoid cached state between figures)
    if name in sys.modules:
        del sys.modules[name]
    t0 = time.time()
    try:
        mod = importlib.import_module(name)
        print("%-26s OK  %.0fs" % (f, time.time() - t0))
    except Exception as e:
        print("%-26s ERR  %.0fs  %s: %s" % (f, time.time() - t0, type(e).__name__, str(e)[:160]))
print("DONE_REDRAW_P1_DIRECT")

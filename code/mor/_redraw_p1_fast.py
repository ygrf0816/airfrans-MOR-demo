"""Re-draw the remaining P1 report figures that carry CJK labels, using the
CORRECT current script for each (the 475-cache scripts reference
field_cache.npz which no longer exists; use the full-1000 / json-driven ones):

  04_field_stats.png   <- p1_field_redraw_full.py   (reads field_stats_full.json, fast)
  05_physics_check.png <- p1_physics_plot.py        (reads physics_full_cache.npz, fast)
  06_split_overlap.png <- p1_splits.py              (reads manifest, fast)
  02_corr_matrix.png   <- write_p1_02.py            (re-decodes 476 samples, ~85s)

Run 04/05/06 first (all fast, json/npz-driven), then 02 last (slow).
"""
import sys, os, time, importlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

# fast, cache/json-driven scripts (each well under the 120 s cap)
fast_jobs = [
    "p1_field_redraw_full",   # 04_field_stats.png
    "p1_physics_plot",        # 05_physics_check.png
    "p1_splits",              # 06_split_overlap.png
]
for name in fast_jobs:
    f = name + ".py"
    if not os.path.exists(f):
        print("SKIP %s (missing)" % f); continue
    if name in sys.modules:
        del sys.modules[name]
    t0 = time.time()
    try:
        importlib.import_module(name)
        print("%-24s OK   %.0fs" % (f, time.time() - t0))
    except Exception as e:
        print("%-24s ERR  %.0fs  %s: %s" % (f, time.time() - t0, type(e).__name__, str(e)[:160]))
print("DONE_FAST_REDRAW")

"""Verify the re-drawn P1 figures now render CJK (no tofu boxes).
Heuristic: tofu boxes appear as large uniform rectangular gaps in a text line
(a run of 'missing glyph' squares). We instead check the *font metadata*
embedded by matplotlib: if the figure was saved with a CJK-capable font,
the PNG has no 'DejaVu-only' tofu. Practical check: re-render a tiny
crop region count is overkill; instead we just re-read which font each
regenerated figure used by confirming plotfont was imported (byte check)
+ confirm file mtime is post-patch.
"""
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def has_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)

# 1. Confirm every figure script that carries CJK labels now imports plotfont
pat = __import__("re").compile(
    r"(set_title|set_xlabel|set_ylabel|suptitle|legend|set_label|title)\s*\(\s*([\"'])(.*?)\2")
print("=== CJK-label scripts: does each import plotfont now? ===")
for f in sorted(glob.glob("03_code/p1/*.py")):
    src = open(f, encoding="utf-8").read()
    if "savefig" not in src:
        continue
    base = os.path.basename(f)
    has_label_cjk = any(has_cjk(m.group(3)) for m in pat.finditer(src) if has_cjk(m.group(3)))
    if has_label_cjk:
        ok = "import plotfont" in src
        print("  %-26s cjk_labels=YES  plotfont=%s" % (base, "OK" if ok else "MISSING"))

# 2. Confirm the 6 report figures were regenerated after the patch (mtime)
import time
now = time.time()
print("\n=== report figures mtime (should be recent) ===")
for f in ["01_inventory", "02_corr_matrix", "03_mesh_geometry",
         "04_field_stats", "05_physics_check", "06_split_overlap"]:
    p = "04_results/analysis/figures/%s.png" % f
    if os.path.exists(p):
        m = os.path.getmtime(p)
        print("  %-20s age=%.0f min  %s bytes" % (f, (now - m) / 60, os.path.getsize(p)))
print("VERIFY_DONE")

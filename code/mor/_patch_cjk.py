"""Batch-fix Chinese-label tofu boxes: inject a CJK font into every
figure script under 03_code/p1/ that sets matplotlib rcParams, then
re-run the figures that actually produced PNGs with CJK labels.

Strategy (conservative/reproducible):
- Write a small `plotfont.py` helper that sets the CJK font once.
- Insert an `import plotfont` line at the top of every script that
  does `matplotlib.use("Agg")` or `import matplotlib.pyplot` AND has CJK.
- Re-run the scripts that emit PNGs to regenerate clean figures.
"""
import sys, glob, os, re, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))

# 1. create plotfont.py (idempotent)
font_helper = '''"""Enable CJK rendering for matplotlib figures (fixes tofu-box garbling).
Import at the top of any figure script:  import plotfont
"""
import matplotlib
matplotlib.rcParams["font.sans-serif"] = [
    "Source Han Sans SC", "SimHei", "Microsoft YaHei",
    "Noto Sans CJK SC", "STSong", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False
'''
with open(os.path.join(HERE, "plotfont.py"), "w", encoding="utf-8") as f:
    f.write(font_helper)
print("wrote plotfont.py")

# 2. patch each figure script to import plotfont right after the matplotlib import
def has_cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)

patched = []
for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
    base = os.path.basename(f)
    if base in ("_scan_cjk.py", "plotfont.py"):
        continue
    src = open(f, encoding="utf-8").read()
    # only patch scripts that both (a) use pyplot AND (b) contain CJK
    if "import matplotlib" not in src and "matplotlib.pyplot" not in src:
        continue
    if not has_cjk(src):
        continue
    if "import plotfont" in src:
        continue
    # insert after the first line that sets matplotlib backend / pyplot import
    lines = src.split("\n")
    insert_at = None
    for i, ln in enumerate(lines):
        if "matplotlib.use" in ln or "import matplotlib.pyplot" in ln or "from matplotlib import" in ln:
            insert_at = i
            break
    if insert_at is None:
        insert_at = 0
    # place after that line
    lines.insert(insert_at + 1, "import plotfont  # CJK font fix (P3-03 figure garble)")
    new = "\n".join(lines)
    open(f, "w", encoding="utf-8").write(new)
    patched.append(base)

print("patched %d scripts:" % len(patched))
for p in patched:
    print("  ", p)
json.dump({"patched": patched}, open(os.path.join(HERE, "_patch_cjk_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("DONE_PATCH_CJK")

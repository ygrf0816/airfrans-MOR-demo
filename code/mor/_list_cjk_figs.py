import sys, os, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def has_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)

pat = re.compile(
    r"(set_title|set_xlabel|set_ylabel|suptitle|legend|set_label|title)\s*\(\s*([\"'])(.*?)\2"
)
for f in sorted(glob.glob("03_code/p1/*.py")):
    src = open(f, encoding="utf-8").read()
    if "savefig" not in src:
        continue
    base = os.path.basename(f)
    hits = set()
    for m in pat.finditer(src):
        if has_cjk(m.group(3)):
            hits.add(m.group(3))
    if hits:
        print("---", base)
        for h in sorted(hits):
            print("   ", h[:40])
    else:
        # also check if the script imports plotfont already and produces png
        if "savefig" in src and "import plotfont" in src:
            print("[patched, no CJK in label args but has CJK elsewhere -> check]", base)

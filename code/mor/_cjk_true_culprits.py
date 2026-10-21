import sys, os, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def has_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)

pat = re.compile(r"(set_title|set_xlabel|set_ylabel|suptitle|legend|set_label|title)\s*\(\s*([\"'])(.*?)\2")
print("=== 实际把中文写进 title/label/legend 的脚本（方块字真凶）===")
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
        # target dir
        tgt = "04_results/analysis/figures" if "analysis" in src or "figures" in src else ("04_results/mor" if "mor" in src else "?")
        print("\n--- %s  (-> %s)" % (base, tgt))
        for h in sorted(hits):
            print("   ", h[:46])

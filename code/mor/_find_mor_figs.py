import sys, os, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def has_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)

pat = re.compile(r"04_results[/\\]mor|mor[/\\]\S+\.png")
for f in sorted(glob.glob("03_code/p1/*.py")):
    src = open(f, encoding="utf-8").read()
    base = os.path.basename(f)
    if "savefig" not in src:
        continue
    writes_mor = bool(re.search(r"savefig\([^)]*04_results", src)) or ("mor" in src and "savefig" in src)
    cjk = has_cjk(src)
    pngs = sorted(set(re.findall(r"(\d{2}_[a-z_]+\.png|[0-9]{1,2}_[a-zA-Z_]+\.png)", src)))
    if writes_mor or cjk:
        print("%-30s writes_mor=%s cjk=%s pngs=%s" % (base, writes_mor, cjk, pngs[:6]))

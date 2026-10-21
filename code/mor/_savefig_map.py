import sys, os, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for f in sorted(glob.glob("03_code/p1/*.py")):
    src = open(f, encoding="utf-8").read()
    if "savefig" not in src:
        continue
    for m in re.finditer(r"savefig\(\s*([\"'])(.*?)\1", src):
        tgt = m.group(2)
        print(os.path.basename(f), "->", tgt)

"""Scan all P1/P2/P3 figure scripts for CJK in title/label/legend strings
(the source of tofu-box garbling when matplotlib's default DejaVu Sans
font has no CJK glyphs).
"""
import sys, glob, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def has_cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)

callers = re.compile(
    r"(set_title|set_xlabel|set_ylabel|set_title|suptitle|set_label|legend|text)\(\s*([\"'])(.*?)\2"
)
for f in sorted(glob.glob("03_code/p1/*.py")):
    src = open(f, encoding="utf-8").read()
    hits = []
    for m in callers.finditer(src):
        arg = m.group(3)
        if has_cjk(arg):
            hits.append(arg)
    # also catch CJK in f-strings / module docstring titles that feed plt
    if hits:
        print("--- %s" % f)
        for h in dict.fromkeys(hits):
            print("   ", h[:50])
    else:
        # quick: any CJK anywhere in the file that could reach a label
        if has_cjk(src):
            print("--- %s  (has CJK somewhere, check labels)" % f)

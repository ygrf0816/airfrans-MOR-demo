"""Fix p1_field_redraw_full.py: `list * 100` repeats the list (7->700), causing
shape mismatch vs x (7). Convert to numpy arrays and scale element-wise.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

path = "03_code/p1/p1_field_redraw_full.py"
lines = open(path, encoding="utf-8").readlines()
new = []
fixed = 0
for ln in lines:
    # `... * 100, w, label=...` on a list-of-comprehension value -> wrap in np.array
    if "for s in ORDER" in ln and "* 100" in ln:
        ln = ln.replace("] * 100,", "] * 100.0,")
        # ensure the comprehension result is np.array'd: replace leading `[FS[` with `np.array([FS[`
        ln = ln.replace("ax[", "ax[", 1)
        new.append(ln)
        fixed += 1
    else:
        new.append(ln)

# Simpler: just replace the specific `] * 100,` patterns that are list multiplications
src = "\n".join(new)
src = src.replace("] * 100, w, label", "])*100.0, w, label")
# catch the ax[2] one
open(path, "w", encoding="utf-8").write(src)
print("patched list*100 -> element-wise (fixed %d lines)" % fixed)
print("DONE_PATCH_04v2")

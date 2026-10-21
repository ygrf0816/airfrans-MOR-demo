"""Fix p1_field_redraw_full.py: line 48's `list*100` repeats the 7-item list to
700 items, then bar() chokes on shape(700,) vs x shape(7). Replace that one
line with an np.array(...)*100.0 version.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = "03_code/p1/p1_field_redraw_full.py"
lines = open(path, encoding="utf-8").read().split("\n")
patched = 0
for i, ln in enumerate(lines):
    if "frac_neg_mean_of_sim'] for s in ORDER] * 100" in ln:
        # wrap the list in np.array(...)*100.0
        new = ln.replace(
            "[FS[s]['p']['frac_neg_mean_of_sim'] for s in ORDER] * 100",
            "np.array([FS[s]['p']['frac_neg_mean_of_sim'] for s in ORDER])*100.0"
        )
        lines[i] = new
        patched += 1
        print("patched line %d" % (i + 1))

open(path, "w", encoding="utf-8").write("\n".join(lines))
print("DONE_PATCH_04_list100 (patched %d)" % patched)

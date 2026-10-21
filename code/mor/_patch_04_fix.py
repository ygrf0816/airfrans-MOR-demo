"""Fix p1_field_redraw_full.py line 35-37: the per-bar label arg plus the
mismatched 1-pos xticks vs 5 labels crashes matplotlib ("FixedLocator locations
(3)"). Replace with a clean single-locator version.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = "03_code/p1/p1_field_redraw_full.py"
lines = open(path, encoding="utf-8").readlines()

new_lines = []
for i, ln in enumerate(lines):
    if "label=f if j == 0 else None" in ln:
        # drop the label arg
        ln = ln.replace(", label=f if j == 0 else None", "")
    if ln.strip().startswith("ax[0].set_xticks(x + off[1]); ax[0].set_xticklabels([f'{f}"):
        # single-locator: 3 positions (one per field) with field-name labels
        ln = "    ax[0].set_xticks(x); ax[0].set_xticklabels(['p','U_mag','nut'])\n"
    if ln.strip() == "ax[0].set_title('field stats full-1000 (log10 magnitude, sign-flipped)')":
        ln = "    ax[0].set_title('field stats full-1000 (log10|value|, sign-flipped; 3 fields, ticks=field)')\n"
    new_lines.append(ln)

open(path, "w", encoding="utf-8").writelines(new_lines)
print("patched p1_field_redraw_full.py (FixedLocator fix)")
print("DONE_PATCH_04")

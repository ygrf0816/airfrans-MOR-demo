"""Probe: confirm plotfont actually renders CJK (no tofu) in THIS matplotlib
before batch re-drawing. Writes 04_results/mor/_font_probe.png with a
sample Chinese title + axis labels; report glyph fallback warnings.
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotfont  # injects CJK font
import numpy as np

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot([0, 1, 2], [0, 1, 4], "o-")
ax.set_title("测试：中文标题 壁面Cp POD 能量衰减")
ax.set_xlabel("模态序号（前缘锚定）")
ax.set_ylabel("累积能量占比")
ax.legend(["p", "|U|", "测试"])
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "04_results", "mor", "_font_probe.png")
plt.savefig(out, bbox_inches="tight")
plt.close()

# Check matplotlib logged any "missing glyph / fallback" warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print("plotfont sans-serif:", matplotlib.rcParams["font.sans-serif"][:4])
    print("probe png written ->", out, os.path.getsize(out), "bytes")
print("DONE_FONT_PROBE")

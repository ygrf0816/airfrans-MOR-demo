"""plot_style.py — 统一画图风格（第 24 轮）。
中文字体 Microsoft YaHei（回退 SimHei/SimSun），统一字号与保存参数。
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
_font = next((c for c in _CJK_CANDIDATES if c in _installed), None)

FONTSIZE_TITLE = 12
FONTSIZE_AXES = 11
FONTSIZE_TICKS = 10
FONTSIZE_LEGEND = 9

if _font is None:  # 本机没有 CJK 字体时回退，但保证字号统一
    _font = matplotlib.rcParams.get("font.family", "DejaVu Sans")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_font, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": FONTSIZE_AXES,
    "axes.titlesize": FONTSIZE_TITLE,
    "axes.labelsize": FONTSIZE_AXES,
    "xtick.labelsize": FONTSIZE_TICKS,
    "ytick.labelsize": FONTSIZE_TICKS,
    "legend.fontsize": FONTSIZE_LEGEND,
    "figure.dpi": 140,
})


def active_cjk_font():
    """返回实际选中的 CJK 字体名（报告自检用）。"""
    return _font


def save(fig, path):
    """统一保存：dpi=140、bbox_inches='tight'。"""
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

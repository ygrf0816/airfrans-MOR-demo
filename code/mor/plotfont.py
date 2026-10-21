"""Enable CJK rendering for matplotlib figures (fixes tofu-box garbling).

Import at the top of any figure script:  import plotfont

字体列表**跨平台**：matplotlib 会按顺序取第一个「本机存在」的字体。
- Linux（工作站 <host>）：实测可用 AR PL UMing CN / AR PL UKai CN / WenQuanYi Zen Hei；
  ⚠ 那里**没有** SimHei / Microsoft YaHei / Noto Sans CJK SC，所以这几项必须排在后面，
  否则会一路回退到 DejaVu 变成方框（2026-10-05 实测踩过）。
- Windows（本机）：SimHei / Microsoft YaHei / STSong 可用。
"""
import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    # —— Linux（工作站）实测可用，零缺字告警 ——
    "AR PL UMing CN", "AR PL UKai CN", "WenQuanYi Zen Hei",
    # —— Windows（本机）——
    "SimHei", "Microsoft YaHei", "Source Han Sans SC", "STSong",
    # —— 兜底 ——
    "Noto Sans CJK SC", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

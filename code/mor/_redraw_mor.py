import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

jobs = [
    ("p2_mor_figs.py",       "04_results/mor/07_wallcp_pod.png + 08/09/10/11"),
    ("p2_mor_post.py",       "04_results/mor/07_wallcp_pod.png + 08/09"),
    ("p3_volume_pod.py",     "04_results/mor/06_volume_pod.png"),
    ("p2_recon_vs_r.py",     "04_results/mor/04_recon_vs_r.png"),
    ("p3_res_scan.py",       "04_results/mor/07_resolution_scan.png"),
    ("p2_inverse_precheck.py","04_results/mor/05_inverse_precheck.png"),
    ("p2_mode_figs.py",      "04_results/mor/03_mode_physics.png"),
]
import re
for f in sorted(glob("03_code/p1/*.py")):
    pass

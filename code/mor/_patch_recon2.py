import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
path = "03_code/p1/p2_grid_self.py"
s = open(path, encoding="utf-8").read()

old = """    p_resid = pp[idx][ok] - p[ok]; u_resid = pu[idx][ok] - Umag[ok]
    p_recon[i] = float(np.sqrt(np.mean(p_resid**2)))          # p RMSE (p units)
    u_recon[i] = float(np.sqrt(np.mean(u_resid**2)))          # |U| RMSE (m/s)
    p_recon_rel[i] = p_recon[i] / (np.sqrt(np.mean(p**2)) + 1e-30)   # / whole-field RMS
    u_recon_rel[i] = u_recon[i] / (np.sqrt(np.mean(Umag**2)) + 1e-30)"""
new = """    p_resid = pp[idx][ok] - p[ok]; u_resid = pu[idx][ok] - Umag[ok]
    p_recon[i] = float(np.sqrt(np.mean(p_resid**2)))
    u_recon[i] = float(np.sqrt(np.mean(u_resid**2)))
    p_recon_rel[i] = p_recon[i] / (np.sqrt(np.mean(p**2)) + 1e-30)
    u_recon_rel[i] = u_recon[i] / (np.sqrt(np.mean(Umag**2)) + 1e-30)"""
if old in s:
    s = s.replace(old, new, 1)
    print("recon block already normalized, kept")

if "p_recon_rel = np.full" not in s:
    s = s.replace("p_recon = np.full(len(sel_chunk), np.nan); u_recon = np.full(len(sel_chunk), np.nan)",
                  "p_recon = np.full(len(sel_chunk), np.nan); u_recon = np.full(len(sel_chunk), np.nan)\n"
                  "p_recon_rel = np.full(len(sel_chunk), np.nan); u_recon_rel = np.full(len(sel_chunk), np.nan)")
    print("added p_recon_rel arrays")

# add relative-rmse to chunk savez
if "p_recon_rel_chunk" not in s:
    s = s.replace('p_recon_chunk=float(np.nanmean(p_recon)), u_recon_chunk=float(np.nanmean(u_recon)),',
                  'p_recon_chunk=float(np.nanmean(p_recon)), u_recon_chunk=float(np.nanmean(u_recon)),\n'
                  '     p_recon_rel_chunk=float(np.nanmean(p_recon_rel)), u_recon_rel_chunk=float(np.nanmean(u_recon_rel)),')
    s = s.replace('p_recon_self=wavg("p_recon_chunk"), u_recon_self=wavg("u_recon_chunk"),',
                  'p_recon_self=wavg("p_recon_chunk"), u_recon_self=wavg("u_recon_chunk"),\n'
                  '         p_recon_rel_self=wavg("p_recon_rel_chunk"), u_recon_rel_self=wavg("u_recon_rel_chunk"),')
    print("added rel-rmse fields to savez (chunk + merge)")

# update final print line to include relative RMSE
if "p_relRMSE" not in s:
    s = s.replace('print("  mask_frac(removed)=%.4f  p_recon=%.4f u_recon=%.4f  p_var_removed_med=%.4f"\n'
                  '      % (float(np.nanmean((~V).mean(axis=1))), np.nanmean(p_recon), np.nanmean(u_recon), float(np.nanmedian(p_var_removed))))',
                  'print("  mask_frac(removed)=%.4f  p_RMSE=%.1f p_relRMSE=%.4f u_RMSE=%.4f u_relRMSE=%.4f  p_var_removed_med=%.4f"\n'
                  '      % (float(np.nanmean((~V).mean(axis=1))), np.nanmean(p_recon), np.nanmean(p_recon_rel),\n'
                  '         np.nanmean(u_recon), np.nanmean(u_recon_rel), float(np.nanmedian(p_var_removed))))')
    print("updated final print to relative-RMSE")

open(path, "w", encoding="utf-8").write(s)
import py_compile
py_compile.compile(path, doraise=True)
print("patched p2_grid_self.py; compile OK")
print("checks:", "p_recon_rel" in s, "p_recon_rel_chunk" in s, "p_relRMSE" in s)

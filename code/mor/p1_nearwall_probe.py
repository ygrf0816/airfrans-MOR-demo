"""P1-05 sec4 near-wall probe: true first-layer height + nodes inside boundary layer.
Reads implicit_distance from internal.vtu of 3 representative sims (low/mid/high Uinf).
sdf = -implicit_distance; first-layer = min |sdf| over nonzero; BL nodes = |sdf| < delta99.
delta99 (turbulent, Schlichting 1/5 power law) = 0.37 * x / Re_x^0.2, x=1m."""
import os, sys, json
import numpy as np
import pyvista as pv
pv.set_plot_theme('default')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DS = r'E:\workspace\agnes-hackathon\10_airfrans\02_dataset\Dataset'
NU = 1.56e-5
sims = sorted([s for s in os.listdir(DS) if os.path.isdir(os.path.join(DS, s))])
def Uinf_of(s):
    return float(s.split('_')[2])
sims_sorted = sorted(sims, key=Uinf_of)
n = len(sims_sorted)
picks = [sims_sorted[n//6], sims_sorted[n//2], sims_sorted[5*n//6]]
out = {}
for s in picks:
    Uinf = Uinf_of(s)
    Re = Uinf / NU
    m = pv.read(os.path.join(DS, s, s + '_internal.vtu'))
    sdf = -np.array(m.point_data['implicit_distance'])
    N = int(m.n_points)
    nz = np.abs(sdf) > 1e-12
    first_layer = float(np.abs(sdf[nz]).min()) if nz.any() else float('nan')
    delta = 0.37 * 1.0 / Re**0.2
    in_bl = int((np.abs(sdf) < delta).sum())
    out[s] = {'Uinf': Uinf, 'Re': Re, 'N': N,
              'first_layer': first_layer, 'first_layer_mm': first_layer*1000,
              'delta_mm': delta*1000, 'in_bl': in_bl, 'in_bl_frac': in_bl/N,
              'first_layer_over_delta': first_layer/delta}
    print('%s Uinf=%.1f Re=%.3e N=%d first_layer=%.3e ch (%.4f mm) delta99=%.3f mm in_bl=%d (%.1f%%) ratio=%.5f' % (
        s, Uinf, Re, N, first_layer, first_layer*1000, delta*1000, in_bl, 100.0*in_bl/N, first_layer/delta))
json.dump(out, open(os.path.join(DS, '..', '..', '04_results', 'analysis', 'nearwall_probe.json'), 'w'), indent=1, ensure_ascii=False)
print('DONE_NEARWALL_PROBE')

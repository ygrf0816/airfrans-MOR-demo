import sys, os, json
import numpy as np, pyvista as pv
from matplotlib.path import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
man = json.load(open(os.path.join(ROOT, '02_dataset', 'Dataset', 'manifest.json'), encoding='utf-8'))
sel = sorted(man['full_train'])[:200]
xs = np.linspace(-0.5, 1.5, 256); ys = np.linspace(-0.5, 0.5, 128)
GX, GY = np.meshgrid(xs, ys, indexing='xy'); G = np.column_stack([GX.ravel(), GY.ravel()])
union = np.zeros(G.shape[0], dtype=bool)
for sim in sel:
    m = pv.read(os.path.join(ROOT, '02_dataset', 'Dataset', sim, sim + '_aerofoil.vtp'))
    P = m.points; poly = np.vstack([P[:,:2], P[:1,:2]])
    union |= Path(poly).contains_points(G)
print('UNION wall mask over %d train airfoils: removed grid frac = %.4f (%d/%d)' % (len(sel), union.mean(), union.sum(), G.shape[0]))
print('valid (fluid) frac = %.4f' % (~union).mean())
print('overnight ref: ~8.7% removed / valid frac ~0.913')

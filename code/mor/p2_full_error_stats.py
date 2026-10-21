"""MOR full-dataset reconstruction-error statistics for the Chinese report:
- wall-Cp: ALL 1000 samples (train 800 + test 200), relL2 vs r=1..256
  (train POD basis, applied to all 1000 -> full-data-set error statistics)
- output: 11_wallcp_full_error.png (2x2: relL2-vs-r curve, test-side histogram,
  train/test side-by-side, energy cumsum) + 11_wallcp_full_error.json
Also prints the key numbers so the report can cite them.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotfont
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "04_results", "mor"))
cpn = np.load(os.path.join(OUT, "p2_wallcp_anchored.npz"), allow_pickle=True)
CP = cpn["CP"]            # (1000, 256)
names = cpn["names"].tolist()
CPtr = CP[:800]
Cpte = CP[800:]
mu = CPtr.mean(0)
U, S, Vt = np.linalg.svd(CPtr - mu, full_matrices=False)
lam = S**2
E = np.cumsum(lam) / np.sum(lam)
# energy-rank for 99/99.9/99.99
def rank_for(e):
    return int(np.searchsorted(E, 1.0 - e, side="left") + 1)
r99, r999, r9999 = rank_for(0.01), rank_for(0.001), rank_for(0.0001)
print("wall-Cp energy ranks: r99=%d r999=%d r9999=%d" % (r99, r999, r9999))

# relL2 vs r, computed on train and test
def relL2_set(arr, Vt, mu, rmax=256):
    """Returns (N_samples, rmax) array of per-sample relL2 at each r."""
    N = arr.shape[0]
    out = np.zeros((N, rmax))
    for r in range(1, rmax + 1):
        Phi = Vt[:r].T
        alpha = (arr - mu) @ Phi
        recon = alpha @ Phi.T + mu
        out[:, r-1] = np.linalg.norm(arr - recon, axis=1) / np.linalg.norm(arr, axis=1)
    return out

rts = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256]
train_per = relL2_set(CPtr, Vt, mu, rmax=256)   # (800,256)
test_per = relL2_set(Cpte, Vt, mu, rmax=256)     # (200,256)
train_rel = train_per.mean(0)                     # mean over train samples
test_rel = test_per.mean(0)                       # mean over test samples

print("train relL2 r=8: %.4f" % train_rel[7])
print("test  relL2 r=8: %.4f" % test_rel[7])
print("train relL2 r=16: %.4f, test r=16: %.4f" % (train_rel[15], test_rel[15]))
print("test  relL2 r=64: %.4f" % test_rel[63])

# Figure: 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax = axes[0, 0]
ax.semilogy(rts, [train_rel[r-1] for r in rts], "o-", color="#1f77b4", label="train relL2（均值）")
ax.semilogy(rts, [test_rel[r-1] for r in rts], "s-", color="#d62728", label="test relL2（均值）")
ax.axvline(r99, color="gray", ls="--", lw=1)
ax.axvline(r999, color="gray", ls=":", lw=1)
ax.text(r99, 0.5, " r99=%d" % r99, fontsize=8)
ax.text(r999, 0.5, " r999=%d" % r999, fontsize=8)
ax.set_xlabel("模态数 r"); ax.set_ylabel("相对 L2 误差（log）")
ax.set_title("壁面 Cp 重构相对误差 vs 模态数 r（全量 1000 样本）", fontsize=11)
ax.legend(); ax.grid(alpha=0.3, which="both")
ax.set_yscale("log")
ax.set_yticks([1e-2, 1e-3, 1e-4, 1e-5])

ax = axes[0, 1]
# per-sample distribution at r=8 and r=16 for test samples
for r, col in [(8, "#d62728"), (16, "#2ca02c")]:
    ax.hist(test_per[:, r-1], bins=40, color=col, alpha=0.5,
            label="r=%d (mean=%.4f, p95=%.4f)" %
            (r, test_per[:, r-1].mean(), np.percentile(test_per[:, r-1], 95)),
            edgecolor="k", linewidth=0.4)
ax.set_xlabel("单样本相对 L2 误差"); ax.set_ylabel("样本数")
ax.set_title("test 侧 200 样本：单样本 relL2 分布（r=8 / r=16）", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(np.arange(1, 257), E, color="#1f77b4", lw=1.8, label="累计能量")
ax.axhline(0.99, color="gray", ls="--", lw=1); ax.axhline(0.999, color="gray", ls=":", lw=1)
ax.plot([r99], [E[r99-1]], "ro", ms=6)
ax.plot([r999], [E[r999-1]], "go", ms=6)
ax.set_xlabel("模态数 r"); ax.set_ylabel("累计能量占比")
ax.set_title("壁面 Cp：POD 模态累计能量占比", fontsize=11)
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlim(0, 40)

ax = axes[1, 1]
sc = ax.scatter(test_per[:, 7], test_per[:, 15], s=12, alpha=0.6, color="#9467bd")
ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)
ax.set_xlabel("单样本 relL2 (r=8)"); ax.set_ylabel("单样本 relL2 (r=16)")
ax.set_title("test 侧：r=8 与 r=16 单样本误差对比（200 点）", fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(0, float(np.max(test_per[:, 7]))*1.1); ax.set_ylim(0, float(np.max(test_per[:, 15]))*1.1)

fig.suptitle("MOR 全数据集（壁面 Cp）降维误差统计", fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "11_wallcp_full_error.png"), dpi=120, bbox_inches="tight")
plt.close()
print("-> 11_wallcp_full_error.png")
json.dump({
    "r99": r99, "r999": r999, "r9999": r9999,
    "train_relL2_r8": float(train_rel[7]), "test_relL2_r8_mean": float(test_rel[7]),
    "test_relL2_r8_p95": float(np.percentile(test_per[:, 7], 95)),
    "test_relL2_r16_mean": float(test_rel[15]),
    "test_relL2_r16_p95": float(np.percentile(test_per[:, 15], 95)),
    "test_relL2_r32_mean": float(test_rel[31]),
    "test_relL2_r64_mean": float(test_rel[63]),
    "E_cumsum_first32": E[:32].tolist(),
}, open(os.path.join(OUT, "11_wallcp_full_error.json"), "w", encoding="utf-8"),
  ensure_ascii=False, indent=1)
print("DONE_FULL_ERROR_FIGS")

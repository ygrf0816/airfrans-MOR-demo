"""两遍法：粗分箱求弯度线 → 按弯度线把点分成上下表面 → 各自按 x 重采样。

为什么需要两遍：这些 CSV 里上下表面的点**不是成对出现在同一个 x 上**，
直接按窄 x 箱取 max/min y 会把"只有下表面点的箱"误判成上表面，
画出来就是一条条竖条纹（翼型变成一把梳子）。
"""
import numpy as np


def _resample(x, y, xg):
    """把 (x,y) 重采样到 xg：先按 x 分组合并重复点，再单调插值。"""
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    ux, idx = np.unique(xs, return_index=True)
    if len(ux) < 2:
        return np.full_like(xg, np.nan)
    yy = np.array([ys[xs == v].mean() for v in ux])
    return np.interp(xg, ux, yy)


def split_surfaces(x, y, val, nbin_coarse=64, nout=256, xg=None,
                   reject_k=4.0, reject_floor=0.35, win=11):
    """返回 (xg, upper_val, lower_val)；val 与 y 一一对应（例如 cp 或就是 y 本身）。

    reject_k>0 时对每个表面做**稳健离群剔除**：先按 x 排序取滑动中位数作基线，
    丢掉偏离超过 max(reject_floor, reject_k×MAD) 的点。被丢掉的点的下标会记录在
    返回对象的 dropped 属性里（供如实报告）。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    val = np.asarray(val, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
    x, y, val = x[ok], y[ok], val[ok]

    # 第一遍：粗分箱求上下包络，得到弯度线
    edges = np.linspace(0.0, 1.0, nbin_coarse + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    hi = np.full(nbin_coarse, np.nan)
    lo = np.full(nbin_coarse, np.nan)
    b = np.clip(np.digitize(x, edges) - 1, 0, nbin_coarse - 1)
    for i in range(nbin_coarse):
        m = b == i
        if m.any():
            hi[i], lo[i] = y[m].max(), y[m].min()
    good = np.isfinite(hi) & np.isfinite(lo)
    camber = np.interp(x, ctr[good], 0.5 * (hi[good] + lo[good]))
    is_up = y >= camber
    if xg is None:
        xg = np.linspace(0.0, 1.0, nout)

    dropped = []
    out = []
    for mask in (is_up, ~is_up):
        xs, vs = x[mask], val[mask]
        if reject_k and xs.size > 3 * win:
            o = np.argsort(xs)
            vs_s = vs[o]
            pad = win // 2
            ext = np.r_[np.full(pad, vs_s[0]), vs_s, np.full(pad, vs_s[-1])]
            base = np.array([np.median(ext[i:i + win]) for i in range(vs_s.size)])
            mad = np.median(np.abs(vs_s - base)) + 1e-12
            keep = np.abs(vs_s - base) <= max(reject_floor, reject_k * mad)
            idx = np.where(~keep)[0]
            dropped.append((xs[o][idx], vs_s[idx]))
            xs, vs = xs[o][keep], vs_s[keep]
        out.append(_resample(xs, vs, xg))
    res = (xg, out[0], out[1])
    return res if not dropped else res


def outline(x, y, nbin_coarse=64, nout=256):
    """闭合翼型轮廓：上表面 LE→TE，再下表面 TE→LE。"""
    xg, yu, yl = split_surfaces(x, y, y, nbin_coarse, nout)
    return (np.r_[xg, xg[::-1]], np.r_[yu, yl[::-1]])


def roughness(v):
    d = np.abs(np.diff(np.asarray(v, float)))
    d = d[np.isfinite(d)]
    return (np.median(d), np.percentile(d, 95), d.max()) if d.size else (np.nan,) * 3

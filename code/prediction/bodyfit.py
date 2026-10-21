"""贴体 (s,d) 近壁表示的公共工具（缓存构建与评估必须共用同一套口径）。

关键约定
  * 表面折线从最小 x 处起步、最近邻链接成有序闭合折线；
  * 外法向用**到壁有符号距离的实测梯度**定向（沿 ±n 各探一个小步长，sdf 变大的一侧
    是翼型内部，取反即外法向）。这与 b_field_bodyfit_cache.py 用的规则完全一致，
    保证缓存里采样的方向与评估时回读的方向不会反。
  * 站位 s 在弧长上均匀取 NS 个；壁法向 d 在 [d0, d1] 上对数取 ND 个。
"""
from __future__ import annotations
import os
import numpy as np

NS_BD = 512
ND_BD = 64
D0_BD, D1_BD = 1e-5, 0.06


def order_surface(S):
    """翼型表面点 -> 有序闭合折线，并**统一弦向朝向**。

    先最近邻链接（从最小 x 的驻点起步），再用有符号面积把遍历方向固定下来。
    这一步不是可有可无：最近邻链接的初始方向取决于前缘附近点的索引顺序，
    实测 1000 个样本里 490 个走了反方向 —— 同一个 s 索引在一半样本上落到镜像
    位置，近壁 (s,d) 图像逐样本翻转，输入到 POD 坐标的关系根本无法学习
    （模型只能退化到线性回归水平）。

    约定：有符号面积为负，即从**上表面**沿弧长前进（常规翼型几何下）。
    """
    from scipy.spatial import cKDTree
    n = len(S)
    tr = cKDTree(S)
    used = np.zeros(n, bool)
    cur = int(np.argmin(S[:, 0]))
    order = [cur]
    used[cur] = True
    for _ in range(n - 1):
        _, ii = tr.query(S[cur], k=min(32, n))
        nx = None
        for j in np.atleast_1d(ii):
            if not used[j]:
                nx = int(j)
                break
        if nx is None:
            rem = np.where(~used)[0]
            nx = int(rem[np.argmin(((S[rem] - S[cur]) ** 2).sum(1))])
        order.append(nx)
        used[nx] = True
        cur = nx
    Q = S[np.array(order)]
    x, y = Q[:, 0], Q[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))     # 2x 有符号面积
    if area2 > 0:                      # 需要为负（上表面先走）；为正则整体反向
        Q = np.vstack([Q[:1], Q[:0:-1]])
    return Q


def surface_frame(name, task_root, P, sdf_raw, tree_p=None):
    """返回 (s_surf(n,), Q(n,2), nrm(n,2) 外法向, perim)。

    外法向定向：沿 ±n 各挪一小步，看**最近的 CFD 网格节点**的 implicit_distance
    （正 = 翼型内部）；哪一侧更靠内，另一侧就是外法向。用最近节点查表，避免在每个
    测试样本上对 18 万点建一次 Delaunay（那是 10~20 秒/样本的量级）。
    该规则与 b_field_bodyfit_cache.py 用的连续插值规则等价（缓存实测 1000 个样本
    无效格全为 0，说明两个方向的判定完全一致）。
    """
    import pyvista as pv
    from scipy.spatial import cKDTree
    w = pv.read(os.path.join(task_root, name, name + "_aerofoil.vtp"))
    Q = order_surface(np.asarray(w.points, float)[:, :2])
    seg = np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1)
    arcl = np.concatenate([[0.0], np.cumsum(seg)])
    perim = float(arcl[-1])
    s_surf = arcl[:-1] / perim
    Qm = np.roll(Q, 1, 0)
    Qp = np.roll(Q, -1, 0)
    tg = Qp - Qm
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    nrm = np.column_stack([tg[:, 1], -tg[:, 0]])
    if tree_p is None:
        tree_p = cKDTree(P)
    n = len(Q)
    e = 1e-4
    probe = np.vstack([Q + e * nrm, Q - e * nrm])
    _, jn = tree_p.query(probe, k=1)
    plus = sdf_raw[jn[:n]]
    minus = sdf_raw[jn[n:]]
    # 注意：传进来的 sdf_raw 是**到壁的非负距离**（airfrans.load 的第 4 列，0 在壁面、
    # 向外增大），不是有符号距离。所以"往外"的一侧距离更大：
    # 若 +n 侧距离更小，说明 +n 指向翼型内部，需要整体取反。
    if (plus < minus).mean() > 0.5:
        nrm = -nrm
    return s_surf, Q, nrm, perim


def stations(name, task_root, P, sdf_raw, ns=NS_BD, nd=ND_BD, d0=D0_BD, d1=D1_BD,
             tree_p=None):
    """返回射线采样所需的 (Qs(ns,2), Ns(ns,2), dl(nd,))。"""
    s_surf, Q, nrm, _ = surface_frame(name, task_root, P, sdf_raw, tree_p)
    s_st = np.linspace(0, 1, ns, endpoint=False)
    Qs = np.column_stack([np.interp(s_st, s_surf, Q[:, 0]), np.interp(s_st, s_surf, Q[:, 1])])
    Ns = np.column_stack([np.interp(s_st, s_surf, nrm[:, 0]), np.interp(s_st, s_surf, nrm[:, 1])])
    return Qs, Ns, np.geomspace(d0, d1, nd)


def node_map(name, task_root, P, sdf_raw, ns=NS_BD, nd=ND_BD, d0=D0_BD, d1=D1_BD):
    """CFD 节点 -> (s,d) 双线性回读所需的索引与权重。

    sdf_raw 必须是**到壁的非负距离**（airfrans.load 第 4 列的口径：0 在壁面、向外增大）。
    不要传有符号 sdf，也不要在外面取负号 —— 曾经因为取了负号导致贴体区一个节点都没选上、
    结果静默退化成纯笛卡尔表示。

    返回 (i0, i1, wi, j0, j1, wj, sel, d_node)：
      sel 为 True 的节点落在贴体带内（0 < d_node < d1），可用 (s,d) 场回读；
      其余节点应走远场笛卡尔区。i0/i1 是弦向站位（环形，i1=(i0+1)%ns）。
    每个样本只建两棵 KDTree（网格节点 + 表面点），不建 Delaunay。
    """
    from scipy.spatial import cKDTree
    tree_p = cKDTree(P)
    s_surf, Q, _, _ = surface_frame(name, task_root, P, sdf_raw, tree_p)
    dl = np.geomspace(d0, d1, nd)
    d_node = sdf_raw
    _, idx = cKDTree(Q).query(P)
    s_node = s_surf[idx]
    fi = s_node * ns - 0.5
    i0 = np.clip(np.floor(fi).astype(int), 0, ns - 1)
    i1 = (i0 + 1) % ns
    wi = fi - np.floor(fi)
    di = np.interp(d_node, dl, np.arange(nd), left=np.nan, right=np.nan)
    di_safe = np.nan_to_num(di)
    j0 = np.clip(np.floor(di_safe).astype(int), 0, nd - 2)
    j1 = j0 + 1
    wj = di_safe - j0
    sel = np.isfinite(di) & (d_node > 1e-9) & (d_node < d1)
    return i0, i1, wi, j0, j1, wj, sel, d_node


def read_near(flat_vals, i0, i1, wi, j0, j1, wj, ns=NS_BD, nd=ND_BD):
    """(ns*nd,) 的展平场 -> 节点值（双线性）。"""
    F = flat_vals.reshape(ns, nd)
    v0 = F[i0, j0] * (1 - wj) + F[i0, j1] * wj
    v1 = F[i1, j0] * (1 - wj) + F[i1, j1] * wj
    return v0 * (1 - wi) + v1 * wi

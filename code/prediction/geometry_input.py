"""官方口径几何输入生成器：翼型表面 -> 128 维弦向 y/c 曲线特征向量。

官方 AirfRANS 逐点 GNN 用 [x,y,入流U,sdf,法向] 7 维逐点喂（几何靠 sdf+法向隐式进模型）。
本生成器给 B 链路（5 来流标量 + 几何向量 -> POD 坐标）补上"几何真正进模型"：
  从 _aerofoil.vtp 表面采样 64 个弦向站位，每个站位取上下表面 y/c -> 128 维向量。
口径（P2 已核实的翼型表面读取 + P1 弦长归一化）：
  - 表面 994 点（P1 已核实），弦长 c = x_max - x_min（P1 chord_thickness）
  - 64 个弦向站位 x_i/c ∈ [0,1]（余弦加密前缘，官方 naca_generator cosine_spacing）
  - 每个站位：上表面 y_upper/c, 下表面 y_lower/c（到弦线 y=0 的有符号距离，弦长归一化）
  - 拼成 (64*2)=128 维，前 64 上表面、后 64 下表面
  - 不同翼型（不同厚度/弯度）自然给出不同 128 维向量，几何信息进模型

铁律：几何数据全部从 02_dataset/Dataset/<sim>/sim_aerofoil.vtp 读，不碰 P2 缓存。
"""
import os, glob, json
import numpy as np
import pyvista as pv

# 余弦加密（官方 naca_generator cosine_spacing：beta=pi*[1..0], x=(1-cos)/2，前缘密）
def _chord_nodes(n=64):
    beta = np.pi * np.linspace(1, 0, n, endpoint=False)
    x = (1 - np.cos(beta)) / 2.0
    return x  # 64 个弦向站位，前缘密、后缘疏

N_STATION = 64


def decode_sim(name):
    p = [float(x) for x in name[len('airFoil2D_SST_'):].split('_')]
    return {"U_inf": p[0], "AoA_deg": p[1], "naca": p[2:]}


def camber_height_frac(naca, series):
    """官方 camber 高度（弦长比）：4 位 M/100（M 是 hundredths of chord）；
    5 位 c_l = 0.15*L（naca_generator.py camber_line 核实）。"""
    if series == "4digit":
        M = naca[0]
        return M / 100.0
    else:
        L = naca[0]
        return 0.15 * L


def thickness_frac(naca, series):
    T = naca[-1]
    return T / 100.0


# ---- 几何向量缓存 ----
# pod/geomvec_1000.npz 存 names(1000,) + G(1000,128) + n_station。
# 128 维几何向量是 .vtp 的确定性函数，但每次调用都要重读 1000 个 vtp 文件
# （约 1 分钟），是"归因/复算脚本"里最贵且最没必要的一步。
# 缓存缺失、站位不同、或名字不在缓存里 -> 自动回落到逐个读 .vtp，
# 行为与没有缓存时**完全一致**（不改变任何数值口径）。
_GEOM_CACHE_PATH = os.environ.get("AIRFRANS_GEOM_CACHE",
                                  "/home/user/airfrans/pod/geomvec_1000.npz")
_geom_cache = None


def _load_geom_cache():
    global _geom_cache
    if _geom_cache is not None:
        return _geom_cache
    _geom_cache = {}
    if _GEOM_CACHE_PATH and os.path.exists(_GEOM_CACHE_PATH):
        try:
            z = np.load(_GEOM_CACHE_PATH, allow_pickle=True)
            _geom_cache = {
                "index": {n: i for i, n in enumerate(z["names"].tolist())},
                "G": z["G"],
                "n_station": int(z["n_station"]) if "n_station" in z else N_STATION,
            }
        except Exception:
            _geom_cache = {}
    return _geom_cache


def geometric_vector(name, task_root, n_station=N_STATION):
    """单个翼型 -> (n_station*2,)=128 维 y/c 弦向曲线特征。"""
    _c = _load_geom_cache()
    if _c.get("index") and n_station == _c["n_station"] and name in _c["index"]:
        return _c["G"][_c["index"][name]].astype(np.float64)
    fp = os.path.join(task_root, name, name + "_aerofoil.vtp")
    m = pv.read(fp)
    P = m.points[:, :2]
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    c = x1 - x0
    xs = _chord_nodes(n_station) * c + x0          # 64 弦向站位（余弦加密前缘）
    vec = np.zeros(2 * n_station, dtype=np.float64)
    for i, xx in enumerate(xs):
        band = np.abs(P[:, 0] - xx) < (c / n_station) * 0.5
        ys = P[band, 1]
        if ys.size >= 2:
            vec[i] = ys.max() / c                   # 上表面 y/c
            vec[n_station + i] = ys.min() / c        # 下表面 y/c
        elif ys.size == 1:
            vec[i] = vec[n_station + i] = ys[0] / c
    return vec


def camber_4digit(naca):
    return naca[0] / 100.0


def camber_5digit(naca):
    return 0.15 * naca[0]


def input_vector_5scalar(name):
    """官方 camber 口径的 5 来流标量 [U_inf, AoA, camber, t_c, Q]（复用 input_side._camber_t）。"""
    import input_side
    return input_side.input_vector(name)


def load_all_geometric(names, task_root, n_station=N_STATION):
    """batch：N 翼型 -> (N, 2*n_station) 几何向量矩阵。"""
    out = np.zeros((len(names), 2 * n_station), dtype=np.float64)
    for i, n in enumerate(names):
        out[i] = geometric_vector(n, task_root, n_station)
    return out


def standardize(train_5scalar, test_5scalar):
    """5 标量 z-score（train 算 mean/std，test 复用）。"""
    tr = np.asarray(train_5scalar, dtype=np.float64)
    mean, std = tr.mean(axis=0, keepdims=True), tr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    out = {"mean": mean, "std": std,
           "Xtr": ((tr - mean) / std).astype(np.float32)}
    if test_5scalar is not None:
        te = np.asarray(test_5scalar, dtype=np.float64)
        out["Xte"] = ((te - mean) / std).astype(np.float32)
    return out

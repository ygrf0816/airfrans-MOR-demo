# -*- coding: utf-8 -*-
"""生成前端口径核对基准：参数 -> 128 维几何向量（Python 侧真值）。

用途：前端要用 JS 重写 NACA 生成器与 128 维几何抽取，必须和 Python 逐位对上。
本脚本把若干组参数与对应的 128 维向量写进 geom_fixture.json，前端拿它做自检。

口径来源（照抄，不新造）：
  * 几何生成 01_initial/AirfRANS-main/naca_generator.py 的 naca_generator，
    nb_samples=400，cosine_spacing=True，与 03_code/inversion/field_cases.py 的 geom128 一致。
  * 128 维向量 03_code/prediction/geometry_input.py 的 geometric_vector，
    64 个余弦站位，先上表面后下表面，TE 在左、LE 在右（站序是 x/c 从 1 递减到 0）。
  * 五标量 03_code/prediction/input_side.py 的 _camber_t。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "01_initial", "AirfRANS-main"))
import naca_generator as NG  # noqa: E402

N_ST = 64
NB = 400


def geom128(params):
    """(L,P,Q,T) 或 (M,P,T) -> 128 维几何向量。"""
    pos = NG.naca_generator(tuple(params), nb_samples=NB,
                            cosine_spacing=True, verbose=False)
    X, Y = pos[:, 0], pos[:, 1]
    x0, x1 = X.min(), X.max()
    c = x1 - x0
    beta = np.pi * np.linspace(1, 0, N_ST, endpoint=False)
    xs = (1 - np.cos(beta)) / 2.0 * c + x0
    vec = np.zeros(2 * N_ST)
    for i, xx in enumerate(xs):
        band = np.abs(X - xx) < (c / N_ST) * 0.5
        ys = Y[band]
        if ys.size >= 2:
            vec[i], vec[N_ST + i] = ys.max() / c, ys.min() / c
        elif ys.size == 1:
            vec[i] = vec[N_ST + i] = ys[0] / c
    return vec


def five_scalar(params, u_inf, aoa):
    """与 input_side._camber_t 同口径的五标量（未归一化）。"""
    if len(params) == 4:
        L, P, Q, T = params
        return [u_inf, aoa, 0.15 * L, T / 100.0, float(round(Q))]
    M, P, T = params
    return [u_inf, aoa, M / 100.0, T / 100.0, 0.0]


def outline(params, n=NB):
    """生成翼型轮廓点。默认与 geom128 用的是同一份（nb_samples=400，801 个点），
    这样前端可以拿这份轮廓自己重算 geom128 并逐位对照，不会因为采样密度不同对不上。"""
    pos = NG.naca_generator(tuple(params), nb_samples=n,
                            cosine_spacing=True, verbose=False)
    return pos


CASES = [
    ("5d_q0_a", (3.673469640579036, 7.969, 0, 15.08206399216206), "5digit", 62.0, 8.0),
    ("5d_q0_b", (0.919, 6.98, 0, 14.32), "5digit", 31.283, -4.156),
    ("5d_q0_c", (3.993, 3.005, 0, 5.008), "5digit", 62.0, 8.0),
    ("5d_q0_d", (0.004, 7.969, 0, 19.989), "5digit", 62.0, 8.0),
    ("5d_q0_e", (2.3214083105048897, 7.386408088512039, 0, 11.557420377236344),
     "5digit", 62.0, 8.0),
    ("5d_q1_a", (3.311, 7.957, 1, 7.685), "5digit", 62.0, 8.0),
    ("5d_q1_b", (2.248, 5.618, 1, 8.267), "5digit", 62.0, 8.0),
    ("5d_q1_c", (3.9183696086964033, 7.969, 1, 15.542719989753579),
     "5digit", 62.0, 8.0),
    ("4d_a", (2.0, 4.0, 12.0), "4digit", 62.0, 8.0),
    ("4d_b", (5.0, 3.0, 18.0), "4digit", 45.0, 2.0),
    ("4d_c", (0.0, 0.0, 9.0), "4digit", 62.0, 0.0),
    ("4d_d", (8.0, 6.0, 21.0), "4digit", 80.0, 12.0),
]

items = []
for name, params, family, u_inf, aoa in CASES:
    vec = geom128(params)
    ol = outline(params)
    items.append({
        "id": name,
        "family": family,
        "params": list(params),
        "uInf": u_inf,
        "aoa": aoa,
        "fiveScalar": five_scalar(params, u_inf, aoa),
        "geom128": [float(v) for v in vec],
        "outline": [[float(a), float(b)] for a, b in ol],
    })
    print("%-9s 参数 %-46s geom[0:3]=%s 上限 %.4f 下限 %.4f"
          % (name, params, np.round(vec[:3], 5), vec[:N_ST].max(), vec[N_ST:].min()))

out = {
    "nStation": N_ST,
    "nbSamples": NB,
    "note": ("geom128 前 64 个数是上表面 y/c，后 64 个是下表面 y/c，"
             "站序是 x/c 从 1 递减到 0（TE 在左，LE 在右）。"
             "outline 就是生成 geom128 用的同一份轮廓，点数是 2 乘 nbSamples 加 1，"
             "点序是 TE 到 LE 再回 TE，首尾两点都被强制成 (1, 0)。"
             "用 outline 重算 geom128 应当与 geom128 逐位相同。"),
    "geom128Recipe": {
        "step1": "调用 naca_generator，参数用 params，nbSamples=400，cosineSpacing=true",
        "step2": "取轮廓的最小 x 与最大 x，令 c 等于两者之差，x0 等于最小 x",
        "step3": "站位 xs[i] = (1 - cos(pi*(1 - i/64))) / 2 * c + x0，i 从 0 到 63。"
                 "注意 i 从 0 涨到 63 时 xs 是从大到小，也就是从后缘走到前缘，"
                 "所以 vec 的下标 0 在后缘、下标 63 在前缘。写成 "
                 "(1 + cos(pi*i/64)) / 2 * c + x0 也是一样的。",
        "step4": "对每个站位，取轮廓上满足 |x - xs[i]| < c/64*0.5 的所有点的 y",
        "step5": "若点数大于等于 2，上表面 y/c 取这些点的最大 y 除以 c，"
                 "下表面取最小 y 除以 c；若只有 1 个点，上下表面都取该点的 y 除以 c",
        "step6": "vec[i] 放上表面，vec[64+i] 放下表面",
    },
    "items": items,
}
dst = os.path.join(ROOT, "04_results", "demo", "app", "public", "data", "geom_fixture.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
print("-> %s  %.1f KB" % (dst, os.path.getsize(dst) / 1024))

# 自检：同一组参数两次生成必须一致，且上下表面在近前缘处不相交
v = geom128(CASES[0][1])
print("幂等自检 max|diff| =", float(np.abs(v - geom128(CASES[0][1])).max()))
print("上表面最小 y/c = %.5f，下表面最大 y/c = %.5f（应一正一负）"
      % (v[:N_ST].min(), v[N_ST:].max()))


def geom128_from_outline(pos):
    """只用轮廓点重算 128 维向量，验证 fixture 里给的配方是对的。"""
    X, Y = pos[:, 0], pos[:, 1]
    x0, x1 = X.min(), X.max()
    c = x1 - x0
    beta = np.pi * np.linspace(1, 0, N_ST, endpoint=False)
    xs = (1 - np.cos(beta)) / 2.0 * c + x0
    vec = np.zeros(2 * N_ST)
    for i, xx in enumerate(xs):
        band = np.abs(X - xx) < (c / N_ST) * 0.5
        ys = Y[band]
        if ys.size >= 2:
            vec[i], vec[N_ST + i] = ys.max() / c, ys.min() / c
        elif ys.size == 1:
            vec[i] = vec[N_ST + i] = ys[0] / c
    return vec


worst = 0.0
for it in items:
    ol = np.asarray(it["outline"], float)
    worst = max(worst, float(np.abs(geom128_from_outline(ol)
                                    - np.asarray(it["geom128"], float)).max()))
print("用 outline 重算 geom128 的最大绝对差 = %.3e（应为 0）" % worst)
print("轮廓点数 %d，应等于 2*%d+1 = %d"
      % (len(items[0]["outline"]), NB, 2 * NB + 1))

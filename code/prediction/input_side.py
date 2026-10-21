"""预测阶段共用输入侧地基：参数解码 + 族别特征 + 5 标量输入向量构造。

被 B 链路（Transformer → POD 坐标）和 A 链路（GNN 逐点场）共用，
保证两链路输入口径一致。解码复用 03_code/p1/p1common.decode，不另起炉灶。

单位/量纲约定（沿 P1，官方 airfrans.load() 口径）：
- U_inf: m/s
- AoA: 度（数据集文件名编码）
- camber: 无量纲（NACA 弯度参数，4 位 M/100，5 位 L/100）
- t_c: 无量纲（厚弦比 T/100）
- Q: 族别/弯度曲线型。4 位族 = 0（无此自由度），5 位族 = Q 参数（0 正常 / 1 反弯）

5 标量输入向量（供 Transformer 预测 POD 坐标用）：
    x = [U_inf, AoA, camber, t_c, Q]

归一化：训练前对 5 个标量做 z-score（均值/标准差），测试用同套 mean/std，
避免训练/测试分布错位。mean/std 在训练集上算，存 npz，加载时复用。
"""
import os, json
import numpy as np

# 前缀常量（原复用 03_code/p1/p1common.py，但工作站不上传 03_code/p1，内联一份）
# 铁律：这是 P1 已核实的解码口径，不是新造口径
PREFIX = 'airFoil2D_SST_'
NU_REF = 1.56e-5  # m^2/s，P1 已核实


def decode_sim(name: str) -> dict:
    """文件名 -> dict: U_inf, AoA_deg, naca(list), series('4digit'/'5digit')."""
    p = [float(x) for x in name[len(PREFIX):].split("_")]
    naca = p[2:]
    series = "5digit" if len(naca) == 4 else "4digit"
    return {"U_inf": p[0], "AoA_deg": p[1], "naca": naca, "series": series,
            "Re": p[0] / NU_REF}  # 弦长 1m，NU_REF=1.56e-5（P1 已核实）


def _camber_t(naca, series):
    """NACA 参数列表 -> (camber_frac, t_c, Q)。

    官方口径（naca_generator.py camber_line 源码核实）：
      4 位 (M, P, T): M 是 hundredths of chord（最大弯度×100），camber = M/100（弦长比）
        t_c = T/100, Q = 0
      5 位 (L, P, Q, T): L 是弯度系数，camber 高度 = c_l = 0.15*L（官方 camber_line: c_l=3/20*l）
        t_c = T/100, Q = Q（0 正常弯/1 反弯）
    注：之前误用 M/10、L/10 把弯度高估 10 倍，z-score 后 5 维占比全乱。
    """
    if series == "4digit":
        M, P, T = naca
        return M / 100.0, T / 100.0, 0.0
    else:  # 5digit
        L, P, Q, T = naca
        return 0.15 * L, T / 100.0, float(Q)


def input_vector(name: str) -> np.ndarray:
    """文件名 -> 5 标量输入向量 [U_inf, AoA, camber, t_c, Q]。"""
    d = decode_sim(name)
    cam, tc, Q = _camber_t(d["naca"], d["series"])
    return np.array([d["U_inf"], d["AoA_deg"], cam, tc, Q], dtype=np.float64)


def input_vector_batch(names):
    """list 文件名 -> (N,5) float32 输入矩阵（直接喂 torch，避免与 FP32 模型 dtype 冲突）。"""
    arr = np.stack([input_vector(n) for n in names])
    return arr.astype(np.float32)


def standardize(train_names, test_names=None):
    """对 5 标量做 z-score：训练集算 mean/std，返回 (Xtr, Xte, mean, std)。

    test 用同套 mean/std（避免测试分布错位）。
    """
    Xtr = input_vector_batch(train_names)
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # 防止某维常数为 0（如 4 位族 Q 恒 0 但训练混了 5 位）
    Xtr_n = ((Xtr - mean) / std).astype(np.float32)
    out = {"Xtr": Xtr_n, "mean": mean, "std": std}
    if test_names is not None:
        Xte = input_vector_batch(test_names)
        out["Xte"] = ((Xte - mean) / std).astype(np.float32)
    return out


# 字段名（写入 npz / 评估用）
INPUT_FEATURE_NAMES = ["U_inf", "AoA_deg", "camber", "t_c", "Q"]

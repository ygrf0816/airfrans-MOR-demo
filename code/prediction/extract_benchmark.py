"""从官方 AirfRANS 库源码 + PDF 提取：任务定义 / 几何采样范围 / 模型基准精度 / camber 编码口径。
全部数字来自 01_initial/，可回溯，作为后续模型设计的对标基准（P3 铁律：口径以官方为准）。
输出 04_results/prediction/BENCHMARK_official.md + benchmark_official.json。
"""
import os, json

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_MD = os.path.join(ROOT, '04_results', 'prediction', 'BENCHMARK_official.md')
OUT_JSON = os.path.join(ROOT, '04_results', 'prediction', 'benchmark_official.json')
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)

# ---- 1. 官方 4 任务（PDF §4.1 原文） ----
TASKS = {
    "full":     "train 800 / test 200（大数据 regime）",
    "scarce":   "train 200 / test 200（test 同 full，少数据 regime；官方 scarce_test==full_test）",
    "reynolds": "train: Re∈[3e6,5e6]；test: Re∈[2e6,3e6]∪[5e6,6e6]（Re 外推）",
    "aoa":      "train: AoA∈[-2.5°,12.5°]；test: AoA∈[-5°,-2.5°]∪[12.5°,15°]（攻角外推）",
}

# ---- 2. 几何参数采样范围（PDF Table 1） ----
GEOM_RANGE = {
    "4digit": {"M": "[0,7]", "P": "{0}（P∈(0,1.5] 置 0）", "XX(厚弦比%)": "[5,20]"},
    "5digit": {"L": "[0,4]", "P": "[3,8]", "Q": "{0,1}（0 单弯/1 双弯）", "XX(厚弦比%)": "[5,20]"},
    "note": "M=hundredths of chord（最大弯度×100），P=弯度峰位 tenth of chord，XX=最大厚度×100",
}

# ---- 3. 官方模型基准精度（PDF Table 2/3，full 任务，归一化字段 MSE ×10⁻²，C_D/C_L 相对误差） ----
BASELINE_FULL = {
    # Volume 归一化 MSE (×10⁻²): u_x, u_y, p, nu_t
    "volume_MSE_x1e-2": {
        "MLP":         [0.95, 0.98, 0.74, 1.90],
        "GraphSAGE":   [0.83, 0.99, 0.66, 1.60],
        "PointNet":    [3.50, 3.64, 1.15, 2.92],
        "GraphU-Net":  [1.52, 2.03, 0.66, 1.46],
    },
    # Surface p (×10⁻¹)
    "surface_p_MSE_x1e-1": {"MLP": 1.13, "GraphSAGE": 0.66, "PointNet": 0.93, "GraphU-Net": 0.39},
    # 力系数相对误差 (C_D, C_L)
    "force_relerr": {
        "MLP":         {"C_D": 4.289, "C_L": 0.767},
        "GraphSAGE":   {"C_D": 4.050, "C_L": 0.517},
        "PointNet":    {"C_D": 14.637, "C_L": 0.742},
        "GraphU-Net":  {"C_D": 10.385, "C_L": 0.489},
    },
    # Spearman 相关（C_D, C_L），越接近 1 越好
    "spearman": {
        "MLP":         {"C_D": -0.117, "C_L": 0.913},
        "GraphSAGE":   {"C_D": -0.303, "C_L": 0.965},
        "PointNet":    {"C_D": -0.022, "C_L": 0.938},
        "GraphU-Net":  {"C_D": -0.138, "C_L": 0.967},
    },
}

# ---- 4. 官方 GraphSAGE 在 4 任务（PDF Table 5 / _pdf_full.txt:425-442） ----
# ⚠ 单位陷阱：原表 u_x / u_y / p 用 ×10⁻²，而 **νt 与壁面 p_s 用 ×10⁻¹**（原文 L432/L434）。
#   之前把 νt 的 ×10⁻¹ 数值直接填进 ×10⁻² 的数组，会让 νt 误差看起来大 10 倍
#   （例：scarce νt 原文 0.611×10⁻¹ = 6.11×10⁻²，不是 0.611×10⁻²）。
#   这里统一换算到 ×10⁻²，并保留 surface_p 单独用 ×10⁻¹ 的原口径。
GRAPH_SAGE_4TASK = {
    "volume_MSE_x1e-2": {  # u_x, u_y, p, nu_t —— 全部换算到 ×10⁻²
        "u_x": {"full": 0.832, "scarce": 1.457, "reynolds": 7.558, "aoa": 4.435},
        "u_y": {"full": 0.994, "scarce": 1.454, "reynolds": 3.498, "aoa": 9.400},
        "p":   {"full": 0.661, "scarce": 4.696, "reynolds": 3.826, "aoa": 10.908},
        "nu_t": {"full": 1.60, "scarce": 6.11, "reynolds": 16.94, "aoa": 51.78},
    },
    "surface_p_MSE_x1e-1": {  # 原表就是 ×10⁻¹
        "full": 0.662, "scarce": 1.945, "reynolds": 1.797, "aoa": 7.638},

    "C_D_relerr": {"full": 4.050, "scarce": 3.504, "reynolds": 8.971, "aoa": 5.589},
    "C_L_relerr": {"full": 0.517, "scarce": 0.385, "reynolds": 0.616, "aoa": 0.818},
}

# ---- 5. camber 编码（naca_generator.py camber_line，官方源码核实） ----
CAMBER_RULE = {
    "4digit": "m = M/100（M 是 hundredths of chord，如 M=0.919 → camber 高度 = 0.00919 弦）",
    "5digit": "l = L，c_l = 0.15*L（最大弯度系数，如 L=2 → c_l=0.3；再经 Newton 迭代得 camber 曲线）",
    "warning": "我之前 5 标量输入的 camber=M/10（4位）/L/10（5位）把弯度高估 10 倍，z-score 后各维度占比全乱，是模型学不动的一个真实原因",
}

# ---- 6. 官方模型架构要点（params.yaml + 源码核实） ----
ARCH = {
    "input_x (每点 7 维)": "[x,y, 入流Ux,入流Uy, sdf(隐式距离), 法向nx, 法向ny]",
    "target_y (每点 4 维)": "[u, v, p(盘原样), nu_t]",
    "encoder/decoder": "MLP [7,64,64,8] / [8,64,64,4]（每模型前置 encoder 后置 decoder，一起训）",
    "GNN 骨干": "SAGEConv/GCN，3 隐藏层 × 64，BatchNorm，ReLU",
    "图构建": "官方用 radius graph（r=0.05m, max_neighbors=64），不是 quad 单元边！",
    "采样": "每 epoch 随机采 32000 节点（subsampling）",
    "loss": "volume + λ*surface，λ=1（壁面力误差权重）",
    "epochs": 398,
    "lr": 0.001,
    "params": {"MLP": 19988, "GraphSAGE": 29204, "PointNet": 75244, "GraphU-Net": 65820},
    "note": "官方最优基准只有 ~2-3 万参数！我 B 链路 50K 的 Transformer 反而更大但用错了骨干（5 token 自注意力）",
}

# ---- 写 MD ----
lines = ["# 官方 AirfRANS 对标基准（P3 阶段，可回溯 01_initial/）",
         "\n> 全部数字来自官方库源码（`01_initial/AirfRANS-main/`）+ 官方 PDF（`135_airfrans_high_fidelity_computa.pdf`）。",
         "> 用途：P3 预测阶段的模型设计与评估口径对齐官方，后续 3.4 统一评估拿这组数对标。",
         "\n## 1. 官方 4 任务定义（PDF §4.1）", ""]
for t, d in TASKS.items():
    lines.append(f"- **{t}**：{d}")
lines += ["\n## 2. 几何参数采样范围（PDF Table 1）", ""]
for fam in ["4digit", "5digit"]:
    r = GEOM_RANGE[fam]
    lines.append(f"- **{fam}**：" + "，".join(f"{k}={v}" for k, v in r.items()))
lines.append(f"- 说明：{GEOM_RANGE['note']}")

lines += ["\n## 3. 官方模型基准精度（full 任务，PDF Table 2/3）", "",
          "归一化字段 MSE（×10⁻²）+ 力系数相对误差（C_D/C_L）+ Spearman 相关：", ""]
lines.append("| 模型 | u_x | u_y | p | ν_t | C_D 相对误差 | C_L 相对误差 | ρ_C_L |")
lines.append("|---|---|---|---|---|---|---|---|")
v = BASELINE_FULL["volume_MSE_x1e-2"]
for m in ["MLP", "GraphSAGE", "PointNet", "GraphU-Net"]:
    lines.append(f"| {m} | {v[m][0]} | {v[m][1]} | {v[m][2]} | {v[m][3]} | "
                 f"{BASELINE_FULL['force_relerr'][m]['C_D']} | "
                 f"{BASELINE_FULL['force_relerr'][m]['C_L']} | "
                 f"{BASELINE_FULL['spearman'][m]['C_L']} |")
lines.append("\n**关键**：官方最优 C_L 相对误差 = GraphU-Net 0.489，GraphSAGE 0.517（full 任务）。"
             " 我 B 链路当前 cp_relL2≈0.47 是 **Cp 曲线**相对 L2，与官方 C_L 相对误差不同口径，"
             " 需统一到官方指标（C_L 相对误差 + Spearman）再对标。")

lines += ["\n## 4. 官方 GraphSAGE 在 4 任务的力系数（PDF Table 5）", "",
          "| 任务 | C_D 相对误差 | C_L 相对误差 |", "|---|---|---|"]
for t in ["full", "scarce", "reynolds", "aoa"]:
    lines.append(f"| {t} | {GRAPH_SAGE_4TASK['C_D_relerr'][t]} | {GRAPH_SAGE_4TASK['C_L_relerr'][t]} |")

lines += ["\n## 5. camber 编码口径（naca_generator.py 源码核实）", ""]
for k, r in CAMBER_RULE.items():
    lines.append(f"- **{k}**：{r}")

lines += ["\n## 6. 官方模型架构要点（params.yaml + 源码）", ""]
for k, r in ARCH.items():
    lines.append(f"- **{k}**：{r}")

lines += ["\n## 7. 对本 P3 模型设计的修正结论", "",
          "1. **输入要含几何**：官方逐点模型输入是 7 维（位置+入流速度+sdf+法向），不是 5 标量。"
          " 几何信息通过 sdf（隐式距离到翼面）+ 法向 + 位置进入，**不同翼面自然区分**。"
          " B 链路走 POD 坐标若只用 5 标量丢了几何，是'对不同翼型怎么算'答非所问。",
          "2. **camber 口径我之前错 10 倍**（M/10 → 官方 M/100），z-score 全乱。",
          "3. **GNN 图构建**：官方用 radius graph（r=0.05m, max_neighbors=64），"
          " 我 A 链路用 quad 单元边，口径不同需统一。",
          "4. **官方最优才 2-3 万参数**（GraphSAGE 29204），我 50K 的 Transformer 更大反而用错骨干。"
          " 换官方 GraphSAGE 架构（SAGEConv 3 层 × 64 + radius graph）才公平对标。",
          "5. **评估口径**：统一用官方 C_L/C_D 相对误差 + Spearman（不是 cp_relL2），"
          " 3.4 统一评估拿官方表做基准。",
          "",
          "## 8. 下一步（等你定方向）", "",
          "- 把 B 链路输入换成官方 7 维逐点（或几何特征化 128 维 + 来流标量），不是 5 标量。",
          "- 修 camber 口径（M/100, L→0.15L）。",
          "- A 链路 GNN 改官方 radius graph（r=0.05m, max_neighbors=64）+ SAGEConv 3 层 × 64。",
          "- 评估统一 C_L/C_D 相对误差 + Spearman（官方 metrics_NACA.surface_coefficients 口径）。",
          "- 拿官方表（C_L relerr GraphSAGE full=0.517）做对标基准，跑通后比谁更小。"]
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))

# ---- 写 JSON（机器可读对标） ----
json.dump({"tasks": TASKS, "geom_range": GEOM_RANGE,
           "baseline_full": BASELINE_FULL, "graph_sage_4task": GRAPH_SAGE_4TASK,
           "camber_rule": CAMBER_RULE, "arch": ARCH},
          open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written:", OUT_MD, OUT_JSON)
print("benchmark json keys:", list(json.load(open(OUT_JSON)).keys()))

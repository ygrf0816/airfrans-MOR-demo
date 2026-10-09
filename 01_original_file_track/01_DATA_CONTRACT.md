# 演示网页数据与模型契约（唯一口径，改动必须回来改这一份）

这份文件是前端、导出脚本、打包核对三方的共同依据。所有字段名、形状、
取值范围都以下面写死的为准。任何一方要改，必须同时改这一份并通知另一方。

## 0 背景事实（已核实，不许再猜）

这些口径都来自已验收的代码，不是新造：

- 几何生成参数四种：`L`（弯度系数）、`P`（弯度位置）、`Q`（弯型 0 或 1）、`T`（厚度系数）。
  取值范围 `L∈[0.004,3.993]`，`P∈[3.005,7.969]`，`Q∈{0,1}`，`T∈[5.008,19.989]`。
- 模型输入的五标量口径见 `03_code/prediction/input_side.py` 的 `_camber_t`：
  `[U∞, 攻角, 0.15L, T/100, round(Q)]`。注意弯度是 `0.15L`，不是 `L`，也不是 `L/10`。
- 几何向量 128 维口径见 `03_code/prediction/geometry_input.py` 的 `geometric_vector`：
  64 个余弦加密弦向站位，先上表面 `y/c`（下标 0 到 63），后下表面 `y/c`（下标 64 到 127），
  横坐标是 `x/c` 的余弦站位，前缘加密。前缘在 `x/c=0`，后缘在 `x/c=1`。
- 压力曲线 256 点口径见 `03_code/prediction/workstation_pod.py` 的 `wall_cp(anchor_le=True)`：
  **沿翼型轮廓的归一化弧长**，`s = linspace(0,1,256)`，`s=0` 在前缘，前半段走上表面，
  `s≈0.5` 在后缘，后半段走下表面，`s=1` 回到前缘。所以这条曲线是**闭环**，
  不是弦向横轴。横轴坐标就是 `s`。`Cp = p / (0.5 U∞²)`，已经除过动压。
  画成论文里那种"上下表面各一条"的图需要另做换算，换算规则见
  `03_code/prediction/b_cp_to_xchord.py`，前端不必自己做。
- 流场在 256×128 公共网格上，`x` 从 -2.16 到 4.23（256 点），
  `y` 从 -1.65 到 1.65（128 点），掩膜外是翼型内部与远场剔除，有效格 31066 个。
  场的物理量是 `u`、`v`、`p`、`nu_t` 四个，行优先 `(ny, nx) = (128, 256)`。
- 两个模型都是"输入 133 维，输出低维坐标"的 MLP，结构为
  `Linear(133,256) → LayerNorm → GELU → 4×(Linear(256,256) 带中间 LayerNorm/GELU/Dropout) → LayerNorm → Linear(256, 输出维)`。
  压力曲线模型输出 8 维，流场模型输出 128 维（4 场 × 32 模态）。

## 1 文件清单与体积预算

放在 `04_results/demo/app/public/data/` 下，构建时 Vite 会原样拷进 `dist/data/`。

| 文件名 | 类型 | 预估体积 | 必需 |
|---|---|---|---|
| `manifest.json` | JSON 清单 | 3 KB | 是 |
| `b_cp.onnx` | ONNX 模型 | 1.0 MB | 是 |
| `b_field.onnx` | ONNX 模型 | 1.5 MB | 是 |
| `cp_basis.json` | JSON | 60 KB | 是 |
| `field_basis.bin` | float32 二进制 | 15.9 MB | 是 |
| `field_basis.json` | JSON 元信息 | 2 KB | 是 |
| `grid.json` | JSON | 40 KB | 是 |
| `samples.json` | JSON 索引 | 0.6 MB | 是 |
| `samples_geom.bin` | float32 [1000,128] | 0.5 MB | 是 |
| `samples_cp.bin` | float32 [1000,256] | 1.0 MB | 是 |
| `samples_alpha.bin` | float32 [1000,136] | 0.5 MB | 是 |
| `cases_field.bin` | float32 | 6.0 MB | 是 |
| `cases_field.json` | JSON 元信息 | 5 KB | 是 |
| `results.json` | JSON | 4.0 MB | 是 |
| `figs/*.webp` | 图片 | 4.0 MB | 是 |
| `geom_fixture.json` | JSON 基准 | 244 KB | 是（已生成） |

合计约 34 MB。加上前端构建产物 1 MB 以内，交付包远低于 200 MB。

`a_gnn.onnx`（官方 GNN 对照）**不做**。逐点图网络导出 ONNX 需要散点算子，
失败风险高、收益低。官方对照改用静态指标表，放在 `results.json` 的 `official_benchmark` 里。

所有二进制一律小端 float32。解析方式一律用 `Float32Array` 直接映射，
不要写 JSON 数字数组（体积会翻三倍，解析也慢）。

## 2 逐文件字段定义

### 2.1 `manifest.json`

```jsonc
{
  "generatedAt": "2026-10-XX",          // 生成日期
  "version": 1,
  "entries": [
    { "name": "b_cp.onnx", "kind": "model", "sizeMb": 1.0, "required": true,
      "source": "prediction/04_results/prediction/runs/full_B/best_model.pt" }
  ],
  "notes": "资产清单，供前端核对与容量检查"
}
```

`kind` 取值只有这几种：`model`、`basis`、`grid`、`samples`、`cases`、`results`、`image`。

### 2.2 `b_cp.onnx`

- 输入：`x`，float32，形状 `[1,133]`。前 5 维是五标量 z-score 后的值，
  后 128 维是几何向量 z-score 后的值。
- 输出：`alpha`，float32，形状 `[1,8]`，单位是 POD 坐标，不是物理量。
- 还原成压力曲线：`cp[i] = sum_k alpha[k] * cp_basis.phi[i][k] + cp_basis.mu[i]`。
  这一步在 JS 里做，8×256 次乘加，可以忽略不计。
- 导出时必须验证：同一批输入下，ONNX Runtime 输出与 PyTorch 输出逐元素满足
  `|差| <= 1e-4 + 1e-3 * |PyTorch 值|`，这是 PyTorch 与 NumPy 通用的合并容差。
  只看绝对差会把正常结果判成不通过，因为输出是模态坐标，量级能到几十上百，
  float32 一路累加下来的绝对差到 1e-4 量级属于正常舍入。
  同时报最大绝对差、最大相对差、越界元素个数三个数。验证样本至少 8 条，
  覆盖四位族与五位族。自检必须逐条喂入，因为导出的图批次维固定为 1。

### 2.3 `b_field.onnx`

- 输入输出同 `b_cp.onnx`，但输出 `alpha` 形状是 `[1,128]`。
- 还原：第 `k` 个场（`k=0` 是 `u`，1 是 `v`，2 是 `p`，3 是 `nu_t`）用
  `alpha[32k : 32k+32]` 去乘 `field_basis.phi[k]` 再加 `field_basis.mu[k]`，
  得到 31066 个有效格上的值，再按下标顺序填进 128×256 的掩膜里。
- 导出时的验证口径与 `b_cp.onnx` 相同，用合并容差逐条比对。

### 2.4 `cp_basis.json`

归一化统计量也放在这一份里，前端做 z-score 时只能取这里的数。

```jsonc
{
  "podR": 8,
  "nPts": 256,
  "s": [0.0, 0.00392, ...],            // 256 个归一化弧长站位，长度 256
  "mu": [ ... ],                       // 长度 256
  "phi": [ [ ... 8 个数 ... ], ... ],  // 长度 256，每项长度 8，行优先
  "smean": [ ... 5 个数 ... ],          // 压力曲线模型的输入统计量
  "sstd":  [ ... 5 个数 ... ],
  "gmean": [ ... 128 个数 ... ],
  "gstd":  [ ... 128 个数 ... ],
  "fieldSmean": [ ... 5 个数 ... ],     // 流场模型的输入统计量
  "fieldSstd":  [ ... 5 个数 ... ],
  "fieldGmean": [ ... 128 个数 ... ],
  "fieldGstd":  [ ... 128 个数 ... ],
  "note": "前缘锚定弧长序，s=0 前缘，s≈0.5 后缘，闭环"
}
```

z-score 的做法是逐维 `(值 - mean) / std`，减完除完再拼成 133 维。

### 2.5 `field_basis.json` 与 `field_basis.bin`

`field_basis.bin` 是一个连续的 float32 序列，分两段拼接：

1. 第一段是 4 个场的基矩阵，每个场的形状是 `[31066, 32]`，
   按 `u, v, p, nu_t` 顺序紧凑排列（行优先，即先遍历 31066 个格点，再遍历 32 个模态）。
2. 第二段是 4 个场的均值向量，每个场 31066 个数，同样顺序。

`field_basis.json` 给出偏移量，单位是 float32 的个数：

```jsonc
{
  "podR": 32,
  "nGrid": 31066,
  "nFields": 4,
  "order": ["u", "v", "p", "nu_t"],
  "phiOffset": [0, 993112, 1986224, 2979336],   // 每个场基矩阵起点（31066×32=993112）
  "muOffset":  [3972448, 4003514, 4034580, 4065646],  // 每个场均值起点
  "totalFloats": 4096712,
  "meanOut": [42.43694305419922, 9.696487426757812, -452.3697509765625, 0.0007954502943903208],
  "stdOut":  [29.64150619506836, 30.87746810913086, 2963.272705078125, 0.0029404046945273876],
  "unit": ["m/s", "m/s", "m^2/s^2", "m^2/s"],
  "note": "行优先；格点顺序与 grid.json 的掩膜行优先顺序一致"
}
```

**重要**：POD 基与均值是在归一化空间里的。还原出来的值还要再换算回物理单位，
公式是 `物理值 = 归一化值 * stdOut[k] + meanOut[k]`，`k` 是场序号。
这一步不做，画出来的流场数值和 `cases_field.bin` 对不上。
`meanOut` 与 `stdOut` 取自官方对照模型 `runs_full_A_eval_metrics.json` 的 `coef_norm`，
已经核实过它们就是那个仿射换算的系数。

### 2.6 `grid.json`

```jsonc
{
  "nx": 256, "ny": 128,
  "x0": -2.16, "x1": 4.23, "y0": -1.65, "y1": 1.65,
  "x": [ ... 256 个数 ... ],
  "y": [ ... 128 个数 ... ],
  "maskRle": [ [0, 4114], [1, 12], ... ],   // 行优先拉平后，[值, 连续个数] 成对出现
  "nValid": 31066,
  "note": "掩膜为 true 表示该格在流场内；rle 展平顺序是 (ny, nx) 行优先"
}
```

`maskRle` 展开后长度必须正好 32768，且 `true` 的个数必须等于 `nGrid` 即 31066。

### 2.7 `samples.json`、`samples_geom.bin`、`samples_cp.bin`、`samples_alpha.bin`

`samples.json`：

```jsonc
{
  "count": 1000,
  "fields": ["name", "uInf", "aoa", "series", "L", "P", "Q", "T",
             "clOfficial", "cdOfficial", "split"],
  "items": [
    { "i": 0, "name": "airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32",
      "uInf": 31.283, "aoa": -4.156, "series": "5digit",
      "L": 0.919, "P": 6.98, "Q": 0, "T": 14.32,
      "clOfficial": 0.42, "cdOfficial": 0.011, "split": "train" }
  ]
}
```

- 4 位族的样本没有 `L` 与 `Q`，这两个字段写 `null`，
  另给 `M`（弯度百分数）与固定 `Q=0`。前端按 `series` 分支显示。
- `split` 只有 `train`、`test` 两种，用于"模型见过没见过的样本"对照。

三个二进制文件的行序与 `items` 的顺序严格一致：

- `samples_geom.bin`：float32 `[1000,128]`，第 `i` 行是第 `i` 个样本的几何向量。
  这里是**没有归一化的原始 y/c 曲线**，前端可以直接拿去画翼型形状。
  要喂给模型时再用 `cp_basis.json` 里的 `gmean/gstd` 做 z-score。
- `samples_cp.bin`：float32 `[1000,256]`，第 `i` 行是第 `i` 个样本的真实压力曲线（物理值，未归一化）。
- `samples_alpha.bin`：float32 `[1000,136]`，前 8 列是压力曲线模型的 POD 坐标，
  后 128 列是流场模型的 128 个 POD 坐标。归一化空间里的坐标，与模型输出同一空间。

### 2.8 `cases_field.bin` 与 `cases_field.json`

`cases_field.json`：

```jsonc
{
  "cases": [
    { "code": "case1_success", "L": 3.67, "P": 7.969, "Q": 0, "T": 15.08,
      "uInf": 62.0, "aoas": [4, 8, 10], "verdict": "通过",
      "label": "成功案例" }
  ],
  "blocks": [
    { "key": "p@case1_success@8", "field": "p", "case": "case1_success",
      "aoa": 8, "offset": 31066, "count": 31066 }
  ],
  "nGrid": 31066,
  "totalFloats": 4096712,
  "channelOrder": ["u", "v", "p", "nu_t"],
  "note": "每个块是 31066 个 float32，按 grid 的行优先顺序对应有效格；无效格前端不填"
}
```

`cases_field.bin` 就是所有块按 `blocks` 的顺序首尾相连的 float32。
数据来源是已验收的 `04_results/inversion/results/cases_field.npz`，
它已经把场换算到与官方 `field_grid.npz` 同一单位，不要重新换算。

### 2.9 `results.json`

一份文件装下其余全部可视化数据。顶层键：

```jsonc
{
  "dataset": {
    "nTotal": 1000, "nTrain": 800, "nTest": 200,
    "nFourDigit": 489, "nFiveDigit": 511,
    "reRange": [2005000.0, 5999000.0], "aoaRange": [-4.94, 14.93],
    "uInfRange": [31.4, 93.6],
    "aoaHist": { "binEdges": [...], "counts": [...] },
    "reHist":  { "binEdges": [...], "counts": [...] },
    "splitCards": [ { "name": "full", "train": 800, "test": 200 },
                    { "name": "scarce", "train": 200, "test": 200 } ],
    "cpStats": { "mean": [...], "p05": [...], "p95": [...] }   // 沿 256 点的分位带
  },
  "modelMetrics": {
    "bCp":   { "relL2Test": 0.0, "relL2NearWall": 0.0, "spearmanCl": 0.0,
               "podFloorRelL2": 0.0, "nParam": 0 },
    "bField":{ "mseGridMean": 0.0, "mseMeshMean": 0.0, "floorMse": 0.0,
               "truncMseGrid": 0.0, "podTruncRelL2Mean": 0.0,
               "perFieldMse": { "u": 0.0, "v": 0.0, "p": 0.0, "nu_t": 0.0 },
               "relL2GridPerField": { "u": 0.0, "v": 0.0, "p": 0.0, "nu_t": 0.0 },
               "spearmanGridPerField": { "u": 0.0, "v": 0.0, "p": 0.0, "nu_t": 0.0 },
               "truncationFloorPerField": {
                 "u": { "relL2Mean": 0.0088, "relL2Median": 0.0085,
                        "relL2P95": 0.0128, "ampRatio": 1.0 } } },
    "aGnn":  { "testVolumeMse": 0.0, "testSurfaceMse": 0.0, "nParam": 0 }
  },
  "objective": {
    "clTargetAoa": 8.0, "bandAoas": [4.0, 8.0, 10.0], "uInf": 62.0,
    "machLimit": 0.6, "sepLimit": 0.007874, "dbDesignBest": 1.146,
    "bandUpperBound": 1.4225,
    "note": "带内损失 = max(CL8-CL4, CL8-CL10)"
  },
  "fronts": [
    { "seed": 42, "L": 3.9, "P": 7.95, "Q": 0, "T": 14.2,
      "cl4": 0.0, "cl8": 0.0, "cl10": 0.0, "delta": 0.0,
      "machMax": 0.0, "sepMax": 0.0, "feasible": true, "hv": 0.0, "gen": 0 }
  ],
  "history": [
    { "seed": 42, "gen": 0, "bestCl8": 0.0, "hv": 0.0, "nFeasible": 0 }
  ],
  "grid": [
    { "L": 2.0, "P": 5.0, "Q": 0, "T": 10.0,
      "cl4": 0.0, "cl8": 0.0, "cl10": 0.0, "delta": 0.0,
      "machMax": 0.0, "sepMax": 0.0, "feasible": true }
  ],
  "cases": [
    { "code": "case1_success", "label": "成功案例", "verdict": "通过",
      "L": 3.67, "P": 7.969, "Q": 0, "T": 15.08,
      "cl4": 0.0, "cl8": 0.0, "cl10": 0.0, "delta": 0.0,
      "machMax": 0.0, "sepMax": 0.0, "feasible": true,
      "reason": "三项约束全部满足" }
  ],
  "inversions": [
    { "name": "airFoil2D_SST_34.806_8.99_2.295",
      "isHeldOut": true, "group": "heldout",
      "uInf": 34.806, "aoa": 8.99,
      "trueParams": { "L": 2.295, "P": 5.593, "Q": 1, "T": 19.795 },
      "recoveredParams": { "L": 3.202, "P": 3.279, "Q": 0, "T": 19.989 },
      "targetCp": [ ... 256 个数 ... ],
      "recoveredCp": [ ... 256 个数 ... ],
      "cpAtTrue": [ ... 256 个数 ... ],
      "cpRelL2": 0.0661, "modelFloorRelL2": 0.0799,
      "clRecovered": 1.0552, "clOfficial": 1.0255,
      "nCandidates": 60, "nParamsWithin1pct": 37,
      "multiSolution": {
        "nCandidates": 60, "nUniqueCandidates": 25, "bestCpRelL2": 0.0661,
        "nWithinCpTol": { "0.01": 37, "0.02": 39, "0.05": 50, "0.1": 53 },
        "tolDefinition": "候选的 cp_relL2 不超过最优 cp_relL2 的多少倍",
        "nWithinTol1pct": 37, "nUniqueWithinTol1pct": 3,
        "paramSpreadWithinTol1pct": { "L": 0.0, "P": 0.02, "Q": 0, "T": 0.015 },
        "paramSpreadNote": "百分之一容差下近优解的四个参数各自的极差，数值大说明形状差别大"
      } }
  ],
  "anchor": {
    "spearmanOwn": 0.9834, "spearmanFixed": 0.9842,
    "trainSpearman": 0.9810, "heldOutSpearman": 0.9936,
    "deciles": [ { "bin": 1, "predMean": 0.0, "trueMean": 0.0, "n": 0 } ],
    "neighbour": [ { "name": "...", "dist": 0.0, "biasPct": 0.0 } ]
  },
  "officialBenchmark": {
    "rows": [ { "model": "官方 GNN", "metric": "test volume MSE", "value": 0.0 },
              { "model": "本项目 A 链路", "metric": "test volume MSE", "value": 0.0 } ],
    "note": "两个口径的网格数不同，不能直接比数值大小，只能比同口径的排序"
  },
  "caveats": [
    "绝对升力值未经过风洞或官方力系数验证，只用相对排序与相对误差",
    "压力曲线拟合好不等于形状还原唯一，反演存在明显多解"
  ],
  "llmTools": {
    "designVars": [
      { "key": "L", "min": 0.004, "max": 3.993, "step": 0.001,
        "label": "弯度系数", "effect": "越大升力越高，但巡航阻力也越大" }
    ],
    "intents": [
      { "id": "max_cl_band", "label": "8 度附近升力尽量高且 10 度不掉",
        "keywords": ["升力", "高", "8度", "10度", "不要掉"], "preset": { "bandAoas": [8,10] } }
    ],
    "presets": [
      { "id": "preset_high_lift", "label": "高升力方案", "L": 3.9, "P": 7.95, "Q": 0, "T": 14.2 }
    ]
  }
}
```

`grid` 这一项约 1587 行，是已算好的可行性网格。`fronts` 是三个种子的全部前沿成员。
`history` 是收敛曲线。三者都来自 `04_results/inversion/results/` 下已验收的产物，
只做格式转换，不重算。

### 2.10 `figs/`

把论文与阶段报告的 PNG 转成 WebP，宽度上限 1400 像素，质量 82。
文件名 `fig01.webp` 到 `figNN.webp`，另在 `results.json` 之外单给一份
`figs/index.json`：

```jsonc
{ "items": [ { "file": "fig01.webp", "title": "数据集总览", "source": "论文图 1",
               "caption": "一段中文说明" } ] }
```

## 3 数值口径的三条硬规矩

1. **不许改物理量。** 压力曲线是弧长序闭环 Cp，流场是四个物理量场，
   不要把归一化后的值当成物理值传给前端，也不要反过来。
2. **归一化统计量必须随模型一起导出。** `smean/sstd/gmean/gstd` 与
   `Phi/mu` 都来自 checkpoint 的 `norm_stats` 与 POD 基，
   前端只能用这几个数做 z-score，不许自己再算一份。
3. **索引对齐靠名字，不靠下标。** 样本、案例、反演目标一律用名字做键，
   跨文件取数据时先建名字到下标的字典。这条是踩过坑的。

## 4 交接约定

- 导出脚本写在 `04_results/demo/tools/` 下，只写文件到
  `04_results/demo/app/public/data/`，不许改 `app/src/` 下任何文件。
- 前端只改 `app/src/`、`app/package.json`、`app/index.html`、
  `app/public/` 下除 `data/` 以外的内容。
- 两边都不许动对方的目录。契约要改，先改这份文件。

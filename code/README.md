# 核心代码

按阶段分目录。这些脚本由本作品自己编写，其中几何生成、数据加载与力系数口径三处
参照或直接使用了官方 AirfRANS 的代码，逐条清单见上一级的 `06_airfrans_reuse.md`。
通用依赖见 `requirements.txt`。

| 目录 | 内容 | 主要文件 |
|---|---|---|
| `prediction/` | 代理模型训练、评估、官方口径与基线对照 | `train_B_field.py`、`train_B_transformer.py`、`official_metrics.py`、`geometry_input.py`、`input_side.py`、`single_sample.py` |
| `explain/` | 可解释性分析的落盘与作图 | `plot_report_figs.py`、`plot_field_local.py`、`plot_near_local.py` |
| `inversion/` | 反向设计：可达性、多目标搜索、压力曲线反演 | `ga_design.py`、`verify_weight_local.py`、`field_cases.py`、`summarize_seeds.py` |
| `mor/` | 数据集体检、参数解码、降维与重建核对 | `p1_params.py`、`p1_splits.py`、`p2_pod_wallcp.py`、`p3_volume_pod.py`、`wall_force.py` |
| `data_check/` | 单样本解析与字段核对 | `sample_report.py` 等 |
| `station/` | 远端工作站驱动 | `remote.py` |
| `analysis/` | 数据集体检阶段的脚本，产物已并入 `mor/` 与 `03_reports/01_dataset_check/` | — |

## 运行环境

代码里读取数据集的根目录写成相对项目根目录的路径，另有部分脚本为远端工作站所写，
顶部常量形如 `ROOT = "/home/zch/airfrans"`。在本地跑之前先把这一类常量改成自己的路径。

依赖见上一级的 `requirements.txt`。

## 几个容易踩的地方

这些都是排查过程中真实踩过的，写在这里免得复现的人重复踩：

1. **压力曲线是沿翼型轮廓一圈的闭合回路，不是从前缘到后缘的横轴。** 256 个点沿弧长均匀取样，起点锚在前缘，前半段走上表面，约一半处到后缘，后半段回到前缘。索引 0 与 255 都是前缘，约 127 是后缘。把它当成弦向坐标会得到完全错误的结论。
2. **几何向量的站序是后缘在前。** 128 维里前 64 个是上表面，后 64 个是下表面，下标 0 在后缘、下标 63 在前缘。
3. **场模态基在归一化空间。** 还原出的值要先乘标准差再加均值才是物理量，系数在 `field_basis.json` 里。
4. **多解性计数的口径。** 数的是「压力距离落在最优值百分之一以内的候选个数」，不是「参数落在真值百分之一以内」，两者差一个数量级，不要混用。
5. **`.vtp` 的 `<Lines>` 是约 994 条独立网格边，不是单一折线。** 必须按邻接表重建闭环才能算周长与法向和。存的是内法向，所以 `F = Σ p · n_in · ds`。
6. **最大厚度位置是离散量。** 全库只有五个取值，五位族只用其中三个，不能当连续变量扰动或求导。

## 外部接口

反向设计页的对话功能可选调用 OpenAI 兼容的 `chat/completions` 接口，
接口地址、模型名与密钥由使用者在页面上现场填写，只存在当次页面的内存中。
核心训练与搜索流程不调用外部接口。

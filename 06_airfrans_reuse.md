# AirfRANS 代码库复用说明

这份文件把本作品用到的第三方代码逐条列清楚，包括仓库地址、复用方式、用途与许可证，
目的是让评审能一眼看清哪些是官方代码、哪些是本作品自己写的，避免抄袭与来源不清的争议。

## 一、复用的代码库

| 项 | 内容 |
|---|---|
| 仓库名称 | AirfRANS |
| GitHub 地址 | https://github.com/Extrality/AirfRANS |
| 作者与机构 | Florent Bonnet、Jocelyn Ahmed Mazari、Paola Cinnella、Patrick Gallinari（Sorbonne Université / ISIR） |
| 论文 | AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions，NeurIPS 2022 Datasets and Benchmarks Track |
| 论文链接 | https://arxiv.org/abs/2212.07564 |
| 本作品引用的版本 | 随数据集一同发布的 `AirfRANS-main`，工作区内的副本在 `10_airfrans/01_initial/AirfRANS-main/` |
| 许可证 | 仓库 `LICENSE` 文件为 ODC Open Database License（ODbL），主要针对数据集本身；代码与数据的使用与再分发请以该仓库的 LICENSE 与 README 为准 |

## 二、复用方式

复用方式分两种，都可以在源码里逐条核对，调用点都在文件顶部。

**一种是直接 import 官方发布的 `airfrans` Python 包。**
工作站上的安装位置为
`/home/zch/.conda/envs/pytorch/lib/python3.10/site-packages/airfrans/`，
该包包含 `naca_generator`、`dataset`、`sampling`、`simulation` 等模块。

**另一种是把官方仓库里的一个文件原样复制到项目里使用。**
官方仓库 `AirfRANS-main/naca_generator.py` 被复制了一份到工作站的项目根目录
`/home/zch/airfrans/naca_generator.py`，几何生成走的是这一份。
两份文件的 MD5 完全相同（`e2b2d0f4531e08dd2ef5c249ef394e4e`），即逐字节一致，未作修改。
这里如实写明是复制而不是仅 import 包，以免核查时对不上。

需要说明的是，PyPI 上的 `airfrans` 包里的 `naca_generator.py` 与 GitHub 仓库快照
不是同一个修订版，PyPI 版为 134 行、MD5 `23ad3f6a70847b7de3062f2f6f1a9031`，
项目里那份为 93 行。本作品的几何生成没有用 PyPI 版。

## 三、逐条清单

### 3.1 几何生成器 `naca_generator`

| 项 | 内容 |
|---|---|
| 官方文件 | 官方仓库 `AirfRANS-main/naca_generator.py` |
| 调用方式 | 该文件被原样复制到 `/home/zch/airfrans/naca_generator.py`，`02_code/inversion/ga_design.py` 第 51 行 `import naca_generator as NG` 用的是这一份；两份文件 MD5 相同，未作修改 |
| 用途 | 把四个设计参数（弯度系数 L、弯度位置 P、弯型 Q、厚度系数 T）转成真实的翼型轮廓坐标。这是全项目几何环节的基准，反向设计搜出来的解都要经过它才能得到可画、可算的轮廓 |
| 是否改动 | 未改动，逐字节一致 |

同一生成器在 `02_code/inversion/field_cases.py` 与 `plot_report_figs.py` 里
通过 `sys.path` 指向工作区内的官方仓库副本后 import，用于把搜索结果的参数还原成形状。

官方生成器是「参数到形状」这一步的权威实现，自己重写的版本一旦有偏差，
后面几何相关的结论都会跟着错。我们另写过一份浏览器端的 JavaScript 实现，
用官方生成器算出的 12 组样例逐点比对，判据是最大绝对差小于 1e-9，实测通过。

### 3.2 数据加载器 `airfrans.dataset.load`

| 项 | 内容 |
|---|---|
| 官方模块 | `airfrans/dataset.py` |
| 调用方式 | `import airfrans` 后调用 `airfrans.dataset.load(task_root, task, train=...)`，见 `02_code/prediction/train_A_gnn.py` 第 90 至 91 行、`viz_a_vs_b.py` |
| 用途 | 按官方 12 列布局读取样本，用于图神经网络对照实验与真值可视化 |
| 是否改动 | 未改动 |

官方 12 列布局的含义（`x`、`y`、`u_in`、`v_in`、`u_out`、`v_out`、`p_out`、`nu_t`、
`Cp`、`surf`、`U_inf`、`AoA` 等）不是猜出来的，是从官方 `dataset.py` 读出来的。
本作品有相当一部分排查工作是围绕「我们自己解析 `.vtu` / `.vtp` 得到的列是否与官方一致」展开的，
口径对齐的结论记在 `03_reports/06_independent_check/` 下。

### 3.3 力系数口径 `metrics.Compute_coefficients`

| 项 | 内容 |
|---|---|
| 官方模块 | `AirfRANS-main/metrics.py`，函数 `Compute_coefficients`（约第 186 至 234 行） |
| 复用方式 | **口径参照，不是代码调用**。本作品在 `02_code/prediction/official_metrics.py` 里按同一口径自己实现，并在文件头注明出处与行号 |
| 用途 | 算官方升力与阻力系数。官方口径是逐点 U 梯度得到壁面剪应力，再加压力项，乘以真实法向与真实 cell 弧长，按攻角旋转后取 `coef = 2 × force / U∞²`。官方论文也说明这是「从未归一化的回归场做后处理」得到的 |
| 为什么自己实现 | 官方 `Compute_coefficients` 依赖完整的官方数据加载链路；本作品需要在只有 `.vp` / `.vtu` 原文件、且要批量跑一千条样本的场景下算同一口径，因此按同一公式重写，并用官方真值做逐条比对 |

这一步是全项目最容易被质疑的地方，所以本作品的处理方式是：
**口径写死、来源写明、结果与官方真值逐条比对**。
比对结论记在 `03_reports/05_inverse_design/03_weight_verify.md`。

另外 `02_code/mor/wall_force.py` 里的壁面力积分是**本作品自己写的**：
它自己解析 `.vtp` 的 XML 与 zlib 压缩块，自己按邻接表重建闭合轮廓，不依赖官方代码。
该文件头部记录了四条自检（周长约为 2、法向和约为 0、`Cp_max` 约为 1、`Cd_p` 量级合理）。

### 3.4 数据集划分清单

| 项 | 内容 |
|---|---|
| 来源 | 数据集自带的 `Dataset/manifest.json` |
| 用途 | 官方七个划分（`full`、`scarce`、`reynolds`、`aoa` 的训练与测试） |
| 是否改动 | 未改动，直接使用 |

本作品的实验都把四种划分的结果一并报告，正是为了对齐这份官方清单，
避免只在一种划分上好看的嫌疑。

### 3.5 官方基线数字

官方论文正文与补充材料给出的逐点图网络基线指标，
用于与本作品结果并列对照，数值来源与出处逐条记在
`02_code/prediction/extract_benchmark.py` 与
`03_reports/07_surrogate/BENCHMARK_official.md` 中。

需要特别说明的是：官方在自己约十八万个计算网格节点上评估，
本作品在 256 × 128 的均匀公共网格上评估，两者网格不同，
因此并列时只能比同口径的排序与相对误差，不能直接说谁更强。
这一条已经在说明书、论文与演示作品里都写明。

## 四、本作品自己写的部分

除上述复用项之外，以下内容为本作品原创，没有对应的官方实现：

- 压力曲线与体积流场的降维表示，以及模态坐标的存储与还原链路
- 两个小规模代理模型的结构、训练、导出与浏览器端推理
- 四维参数空间上的带约束多目标进化搜索
- 压力曲线反演与多解性分析
- 可解释性分析的实验设计（几何四分量拆解、两条腿敏感性、交互作用、理论对照）
- 浏览器端演示作品的前端代码
- 中文与英文论文

## 五、引用

如果本作品的任何部分被后续工作使用，请同时引用官方数据集论文：

```
@inproceedings{bonnet2022airfrans,
  title={Airf{RANS}: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions},
  author={Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://arxiv.org/abs/2212.07564}
}
```

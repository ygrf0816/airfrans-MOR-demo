# 原始数据集说明与获取方式

原始数据集有 14.3 GB，无法随作品一起提交。这份文件说明它是什么、从哪里取、
取到之后放在哪、怎么核对取对了。

## 一、数据集是什么

| 项 | 内容 |
|---|---|
| 名称 | AirfRANS |
| 内容 | 1000 条二维 NACA 翼型的 RANS 数值解 |
| 几何族别 | 四位 NACA 翼型 489 条，五位 NACA 翼型 511 条 |
| 每条样本包含 | 一个来流速度、一个攻角、一组翼型几何参数，以及对应的表面压力分布、体积流场（速度分量、压力、涡黏系数）与官方算出的升力阻力系数 |
| 来流速度范围 | 31.28 至 93.59 米每秒 |
| 攻角范围 | 负 4.94 至 14.93 度 |
| 雷诺数范围 | 约 200 万至 600 万 |
| 网格 | 每条样本一套非结构网格，约 18 万个计算节点 |
| 官方论文 | AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions，NeurIPS 2022 Datasets and Benchmarks Track，https://arxiv.org/abs/2212.07564 |
| 官方代码库 | https://github.com/Extrality/AirfRANS |
| 许可证 | ODC Open Database License（ODbL），见官方仓库 `LICENSE` 文件 |

## 二、从哪里获取

**官方下载地址，直接来自官方仓库 README 的说明：**

```
https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip
```

这个地址写在官方仓库 `README.md` 里，原文是
「you must have the dataset in folder `Dataset/` at the root of this repository,
you can download the dataset here」。

**下载后的压缩包大小是 9,564.46 MB（约 9.3 GB），解压后约 14.3 GB。**

下载后请核对两个数，确认拿到的与本文所用的是同一份：
压缩包大小 9,564,461,056 字节左右；解压后有 1000 个样本目录。

## 三、解压后的目录结构

```
Dataset/
├── manifest.json                                    官方划分清单
├── airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32/    一条样本一个目录
│   ├── ..._aerofoil.vtp        翼型轮廓、表面压力与法向，约 58 KB
│   ├── ..._freestream.vtp      远场边界，约 75 KB
│   └── ..._internal.vtu        体积网格与流场解，约 14.4 MB
├── airFoil2D_SST_31.382_3.588_1.994_6.206_0.0_13.271/
└── ... 共 1000 个目录
```

目录名的编码方式（本作品在数据集体检阶段逐条核实过，
结论见 `03_reports/01_dataset_check/`）：

```
airFoil2D_SST_{来流速度}_{攻角}_{参数一}_{参数二}_{参数三}
```

四位族用三个几何参数，五位族用四个，因此目录名的字段数不一样，这是官方命名规则本身的特点。

## 四、官方划分清单

`manifest.json` 里是官方的七个划分，本作品的实验都按这份清单做，
没有另设划分：

| 划分名 | 样本数 | 含义 |
|---|---|---|
| `full_train` | 800 | 大数据划分，训练 |
| `full_test` | 200 | 大数据划分，测试 |
| `scarce_train` | 200 | 少数据划分，训练 |
| `reynolds_train` | 504 | 按雷诺数划分，训练 |
| `reynolds_test` | 496 | 按雷诺数划分，测试 |
| `aoa_train` | 804 | 按攻角划分，训练 |
| `aoa_test` | 196 | 按攻角划分，测试 |

## 五、放到哪里

本作品的工作区约定是数据集放在项目根目录的 `02_dataset/` 下：

```
10_airfrans/
├── 02_dataset/
│   ├── Dataset.zip          下载得到的原始压缩包
│   └── Dataset/             解压后的内容
└── 03_code/  04_results/    ...
```

代码里读取数据集的位置都写成相对于项目根目录的路径。
如果放到别的位置，改 `02_code/prediction/official_metrics.py` 与
`02_code/prediction/single_sample.py` 顶部的数据根目录常量即可。

## 六、为什么演示作品不需要它

演示作品里的数据是降维后的表示，不是原始解：

| 内容 | 大小 |
|---|---|
| 原始数据集（解压后） | 14.3 GB |
| 一千条样本降维后的表示 | 16.98 MB |
| 两个代理模型（ONNX） | 2.55 MB |
| 整个演示站点 | 55.85 MB |

压缩比约 885 比 1。数据量从 14 GB 压到 17 MB，靠的是把压力曲线与体积流场
先做降维、再让代理模型直接预测低维坐标。演示作品自带降维后的表示与模型，
所以用随附的 `serve.py` 起一个本地静态服务器就能跑，不需要原始数据集、不需要联网、不需要显卡。
注意不能直接双击 `index.html`，浏览器的本地文件协议会拦掉数据与模型文件的读取；
也不要用 Python 3.9 之前的 `python -m http.server`，它会把 `.wasm` 发成错误的 MIME，
导致模型加载失败。原因与做法见 `01_demo/README.md`。

要重新跑训练与搜索，才需要按上面第三节把原始数据集准备好。

## 七、数据的使用边界

- 本作品把问题简化为二维、不旋转、无展向流动的孤立翼型绕流。数据集本身就是这个假设下的数值解，不含三维旋转效应、展向流动、叶尖损失与动态失速。结论不得直接外推为真实旋转叶片的结论。
- 本作品不重复运行 CFD，数据集自带的解由官方提供。
- 目标只用升力，不用阻力。摩擦项占阻力的一半以上，而本作品只建模压力，算出的阻力会系统性偏低，报出去是误导。
- 数据集按 ODbL 许可发布，再分发与使用请遵循其条款。

# 数据驱动的翼型气动代理与反向设计

> 2026 年江苏省「人工智能 + 科学」大赛 · 硕博组 · 参赛编号 G012 · 队伍 PCDLab
>
> **在线体验：<https://ygrf0816.github.io/airfrans-MOR-demo/>**

## 这个云端仓库是做什么的

这是参赛作品的**在线演示部分**。它只承担一件事：把已经训练好、已经导出好的模型与数据资产，
以纯静态网页的形式放到公网上，让评审和读者**打开链接就能动手操作**，
不用装环境、不用下载数据集、不用显卡。

模型推理全部在访问者的浏览器里，由 ONNX Runtime 的 WebAssembly 后端完成，
没有后端服务，不联网取数，站点是纯静态的。

### 仓库里**没有**核心模型训练代码

**本作品的创新点与主要工作量不在这个仓库里。** 数据降维（POD 模态基）、
代理模型的结构设计与训练、ONNX 导出、可解释性分析方法、多目标反向设计搜索，
这些核心方法学代码属于参赛作品正文，**不在本仓库中**。

本仓库只包含以下三类内容：

| 类别 | 内容 |
|---|---|
| 演示前端 | `source/`，Vite + React 19 + TypeScript + Three.js + ECharts |
| 导出的资产 | `site/data/`，两个 ONNX 代理模型、模态基、样本与案例数据 |
| 构建与部署 | 已构建好的 `site/`、资产核对脚本 `tools/`、Pages 部署工作流 |

也就是说：**这是一个"结果展示"仓库，不是一个"方法复现"仓库。**
想了解方法与实验，请看竞赛提交材料中的报告与论文部分。

## 作品在做什么

一句话：把一千条二维翼型 RANS 数值解先降维成十几个模态坐标，再用二十万参数量级的小模型
在浏览器里毫秒级复现压力分布与流场，并在此基础上做带约束的反向设计。

这里的 AirfRANS 数据集只是**数据来源**（1000 条二维翼型 RANS 解），
本作品要解决的问题是**"降维 + 代理 + 反向设计"这条链路本身**：

- 一条 CFD 算例要几分钟到几小时，设计空间里动辄要评估几千个方案，算力根本不够；
- 所以先用降维把物理场压成极少数坐标，再用小模型直接预测这些坐标，
  把单次评估从"几分钟"压到"毫秒"；
- 有了这个速度，才能在上面做带约束的多目标反向设计搜索。

## 五个页面

| 页面 | 做什么 |
|---|---|
| 降维演示 | 一千条数据怎么用十几个数字概括；样本索引、秩截断与投影；可上传自己的曲线试投影 |
| 前向预测 | 改几何与来流参数，压力曲线与流场云图即时更新 |
| 可解释性分析 | 几何四分量对流场的影响：全局排序、逐弦向站位局部影响、两条腿一致性、交互作用 |
| 反向设计与对话 | 自然语言转设计目标，给出候选解；可接在线大模型，也可用本地规则模式 |
| 成果总览 | 前沿解、三个种子的收敛、数据集体检 |

设置面板里可以现场填写在线大模型的接口地址、模型名与密钥。
密钥只存在当次页面的内存中，不落盘、不进仓库。不填也能完整演示，会自动切到本地规则模式。

口径、资产清单与字段定义见 [`docs/01_DATA_CONTRACT.md`](docs/01_DATA_CONTRACT.md)，
界面设计与交互说明见 [`docs/02_UI_DESIGN.md`](docs/02_UI_DESIGN.md)。

## 本地运行

**不要直接双击 `site/index.html`。** 浏览器的本地文件协议会拦掉模型文件与数据文件的
读取，页面会是一片空白。用本目录随附的服务器脚本打开：

```
python serve.py
```

然后浏览器访问 `http://127.0.0.1:8080/`。端口可以改：`python serve.py 9000`。

### 为什么不能用 `python -m http.server`

Python 3.9 之前的标准库里没有 `application/wasm` 这条 MIME 记录，
`.wasm` 会被发成 `application/octet-stream`。ONNX Runtime Web 在这种情况下
**不会**退回用 ArrayBuffer 加载，而是直接失败。实测表现是：

- 页面顶部仍显示「真实数据」，说明 JSON 资产加载正常；
- 但四个指标卡全部显示「待接入」，压力曲线与流场云图全空；
- 模型状态写「未启用」，而且不给失败原因。

`serve.py` 只用标准库，把这几条 MIME 写死，因此不依赖 Python 的 MIME 表。
实测环境是 Python 3.12；脚本按 Python 3.6 以上的语法编写。
线上（GitHub Pages）不存在这个问题，它自己就会把 `.wasm` 发成 `application/wasm`。

### 环境要求

| 项 | 要求 | 不满足会怎样 |
|---|---|---|
| Python | 3.6 及以上，只用来起服务器 | 起不了服务器 |
| 静态服务器 | 随附的 `serve.py`，或任何会正确发送 `application/wasm` 的服务器 | 模型全部加载失败 |
| 浏览器 | 支持 WebAssembly SIMD：Chrome 91+ / Edge 91+ / Firefox 89+ / Safari 16.4+ | 模型全部加载失败 |

不需要联网，不需要显卡，不需要 Node.js，不需要装任何 Python 包。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `serve.py` | 随附的静态服务器，把 `.wasm` 的 MIME 写死，不挑 Python 版本 |
| `site/` | 构建好的静态站点，GitHub Pages 发布的就是这个目录 |
| `source/` | 演示前端源码 |
| `tools/` | 资产导出与打包核对脚本（导出环节需要上游数据与 GPU，不随仓库提供数据） |
| `docs/` | 作品计划、数据契约、界面设计说明，以及开发过程的任务单与轮次记录（见 [`docs/README.md`](docs/README.md)） |
| `LICENSE` | 代码的 MIT 许可 |
| `DATA_LICENSE.md` | 数据与模型资产的 ODbL 许可与引用要求 |

`source/` 里不含 `public/data/`，站点用的资产就是 `site/data/`。
原始数据集（14.3 GB 的 AirfRANS 原始 CFD 解）**不在仓库里**，需要时按
[`DATA_LICENSE.md`](DATA_LICENSE.md) 里的地址自行下载。

## 重新构建站点

```
cd source
npm ci
npm run build
```

构建产物输出到 `source/dist/`，把它复制到 `site/` 即可替换（`site/data/` 保持不变）。

`tools/` 下的脚本用于从上游数据重新生成资产，需要 GPU 与完整数据集，
不属于本仓库的复现范围：

```
python tools/make_geom_fixture.py      # 几何核对基准
python tools/make_proxy_fixture.py     # 代理量核对基准
python tools/make_web_data.py          # 汇总成网页资产
python tools/make_explain_data.py      # 可解释性分析页的数据
python tools/verify_demo_assets.py     # 一致性核对
```

## 前端一致性自检

`source/scripts/` 下有检查脚本，判据是「跑出固定字符串才算通过」：

| 脚本 | 判据 |
|---|---|
| `check-geom.mjs` | `PASS all 12 items < 1e-9`，浏览器端几何实现与官方生成器逐点比对 |
| `check-proxies.mjs` | `PROXY_CHECK_OK`，升力积分、局部马赫代理与分离代理 |
| `check-vtp.mjs` | `VTP_CHECK_OK`，浏览器端 `.vtp` 解析与 Python 参考实现比对到 5e-7 |
| `check-naca.mjs` | 翼型几何与官方实现逐点一致 |

## 实现要点

- **单线程 WASM。** `source/src/lib/models.ts` 里写死 `ort.env.wasm.numThreads = 1`。
  多线程后端要 `SharedArrayBuffer`，那要求服务器发 `Cross-Origin-Opener-Policy` 与
  `Cross-Origin-Embedder-Policy` 响应头，而 GitHub Pages 不允许自定义响应头。
- **不依赖站点根路径。** 构建时 `base: './'`，所有资产用相对路径取，
  因此放在 `https://<user>.github.io/<repo>/` 这样的子路径下能直接跑。
- **哈希路由。** 页面切换走 `#hash`，静态托管不需要任何 404 重写规则。
- **缺数据不崩。** 任何资产取不到就退回内置解析近似，并在页面上标出当前是示例数据。
- **体积。** 整站约 58 MB，其中 27 MB 是 ONNX Runtime 的 wasm；
  数据按页面按需拉取，不会一次性下载全部资产。

## 许可与引用

- **代码**（`source/`、`tools/`、`serve.py`）：MIT，见 [`LICENSE`](LICENSE)。
- **数据与模型资产**（`site/data/`）：AirfRANS 数据集的衍生作品，沿用上游
  ODbL v1.0，署名与再分发要求见 [`DATA_LICENSE.md`](DATA_LICENSE.md)。
- 本仓库仅用于竞赛演示与技术交流。

使用本项目或数据集，请引用官方论文：

```bibtex
@inproceedings{bonnet2022airfrans,
  title={Airf{RANS}: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions},
  author={Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://arxiv.org/abs/2212.07564}
}
```

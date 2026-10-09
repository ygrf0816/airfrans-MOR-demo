# AirfRANS 可交互演示

**在线体验：<https://ygrf0816.github.io/airfrans-demo/>**

纯静态的翼型气动演示站。模型推理全部在访问者的浏览器里由 ONNX Runtime 的
WebAssembly 后端完成，没有后端服务、不联网取数、也不需要显卡。

> 基于 AirfRANS 的 1000 条二维翼型 RANS 数值解，先把压力分布与体积流场降维成
> 十几个模态坐标，再用两个二十万参数量级的小模型在浏览器里毫秒级复现，
> 并在此基础上做带约束的反向设计。

口径、资产清单与字段定义见 [`docs/01_DATA_CONTRACT.md`](docs/01_DATA_CONTRACT.md)，
界面设计与交互说明见 [`docs/02_UI_DESIGN.md`](docs/02_UI_DESIGN.md)。

## 五个页面

| 页面 | 做什么 |
|---|---|
| 降维演示 | 一千条数据怎么用十几个数字概括；样本索引、秩截断与投影；可上传自己的曲线 |
| 前向预测 | 改几何与来流参数，压力曲线与流场云图即时更新 |
| 可解释性分析 | 几何四分量对流场的影响：全局排序、逐弦向站位局部影响、两条腿一致性、交互作用 |
| 反向设计与对话 | 自然语言转设计目标，给出候选解；可接在线大模型，也可用本地规则模式 |
| 成果总览 | 前沿解、三个种子的收敛、数据集体检 |

设置面板里可以现场填写在线大模型的接口地址、模型名与密钥。密钥只存在当次页面的
内存中，不落盘、不进仓库。不填也能完整演示，会自动切到本地规则模式。

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
| `source/` | 前端源码（Vite + React 19 + TypeScript + Three.js + ECharts） |
| `tools/` | 资产导出与打包核对脚本 |
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

资产两段式：远端导出要有 GPU 的工作站，本地汇总用 `tools/` 下的脚本：

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
  首次访问按页面按需拉取数据，不会一次性下载全部资产。

## 许可与引用

- **代码**（`source/`、`tools/`、`serve.py`）：MIT，见 [`LICENSE`](LICENSE)。
- **数据与模型资产**（`site/data/`）：AirfRANS 数据集的衍生作品，沿用上游
  ODbL v1.0，署名与再分发要求见 [`DATA_LICENSE.md`](DATA_LICENSE.md)。

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

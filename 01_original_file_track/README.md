# AirfRANS 可交互演示

四个页面的纯静态演示作品，模型推理全部在浏览器里完成，不依赖服务器。
口径、资产清单与字段定义见同目录的 `01_DATA_CONTRACT.md`。
界面设计与交互说明见同目录的 `02_UI_DESIGN.md`。

## 目录里有什么

| 路径 | 内容 |
|---|---|
| `app/` | 前端工程（Vite 加 React 加 TypeScript） |
| `app/public/data/` | 站点用的全部资产，由脚本生成，不要手改 |
| `app/dist/` | 构建产物，自带数据，可离线运行 |
| `app/scripts/` | 前端口径自检脚本 |
| `tools/` | 资产导出、汇总、核对与打包脚本 |
| `00_PLAN.md` | 作品计划与里程碑 |
| `01_DATA_CONTRACT.md` | 数据与模型的唯一口径 |

## 跑起来

```powershell
cd 04_results/demo/app
npm ci
npm run dev        # 开发模式，终端会给出 http://localhost:5173
```

或构建后用静态服务器打开：

```powershell
cd 04_results/demo/app
npm run build
cd dist
python -m http.server 8080
```

浏览器访问 `http://127.0.0.1:8080`。
**不要直接双击 `dist/index.html`**，本地文件协议会拦掉模型与数据文件的读取。

## 重新生成资产

资产分两段。远端导出要有 GPU 的工作站，本地汇总在本机就能做。
完整步骤见 `07_submission/02_复现说明.md`。本地这一段是：

```powershell
python 04_results/demo/tools/make_geom_fixture.py      # 几何核对基准
python 04_results/demo/tools/make_proxy_fixture.py     # 代理量核对基准
python 04_results/demo/tools/make_web_data.py          # 汇总成 public/data
```

## 核对

一条命令跑完全部核对，结论写到 `data_remote/verify_demo_assets.json`：

```powershell
python 04_results/demo/tools/verify_demo_assets.py
```

也可以单独跑前端的两个口径自检：

```powershell
cd 04_results/demo/app
node scripts/check-geom.mjs       # 几何生成与 128 维向量，12 组基准
node scripts/check-proxies.mjs    # 马赫代理、分离代理与升力权重，8 个案例
```

## 打包

```powershell
python 04_results/demo/tools/pack_check.py --audit                             # 只看体积构成
python 04_results/demo/tools/pack_check.py --build --dest 07_submission/package  # 组装并核对
```

## 关于 node_modules 与密钥

`node_modules/` 不进交付包，复现时用 `npm ci`，不要提交锁文件之外的依赖目录。

反向设计页面的在线对话需要在设置面板里现场填写接口信息，密钥只存在于当次页面的内存中，
关掉即丢，不写本地存储，代码与仓库里不含任何密钥。不填时自动使用本地规则模式。

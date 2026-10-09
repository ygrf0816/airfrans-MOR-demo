# Demo 搭建第 1 轮：写项目计划 + 搭基础框架（只做骨架，不做视觉设计）

> 你是 AGH，用的模型是 agnes-3.0-flash。本轮**只做两件事**：写一份项目计划，然后把
> 前端工程骨架搭起来并确认能构建成功。**视觉设计与炫酷效果不要在本轮做**，
> 后面会换更强的模型专门负责。
> 每次 `write`/`edit` **不超过 45 行**；不要停下来汇报，一次做完。

## 第 0 步：先读这三份文件

1. `00_docs/DEMO_RESEARCH.md`：容量体检与方案调研（本轮工作的依据，含 API key 设置面板的要求）。
2. `04_results/paper/paper_zh.md` 的目录与摘要：了解要展示的成果有哪些。
3. `04_results/inversion/FIG_DATA_INDEX.md`：图与原始数据的对应关系。

## 第一件事：写 `04_results/demo/00_PLAN.md`（约 120 行）

用平实的语言写，包含下面几节：

1. **目标与交付物**：一个可交互网页作品，四个页面（前向预测、MOR 降维、反向设计加对话、成果总览），
   论文与演示记录一并交付；总容量不超过 200 MB。
2. **技术选型**：Vite 加 React 加 TypeScript；三维用 Three.js；图表用 ECharts 或 Plotly；
   模型推理由 ONNX Runtime Web 在浏览器里完成；不依赖服务器。
3. **四个页面的功能与交互**（每页写清用户操作、页面反馈、用到的数据文件）。
4. **数据与模型资产清单**：列出每个文件的名字、来源、体积估计（照抄 DEMO_RESEARCH 的清单）。
5. **设置面板与密钥规则**：照抄 `00_docs/DEMO_RESEARCH.md` 第 3.5 节的五条要求。
6. **目录结构**：`04_results/demo/` 下面分 `app/`（前端工程）、`tools/`（数据与模型导出脚本）、
   `data/`（构建时把资产放进 `app/public/data/`）、`dist/`（构建产物）。
7. **里程碑与验收标准**：分四步（骨架、数据接入、视觉与交互、打包核对），每步写清完成判据。
8. **风险与对策**：照抄 DEMO_RESEARCH 第 4 节的表格（可精简到八行以内）。

## 第二件事：搭前端骨架 `04_results/demo/app/`

要求：**能用 `npm run build` 构建成功**。具体做这些：

1. 用命令创建工程（不要手写一堆文件）：
   ```
   cd 04_results/demo
   npm create vite@latest app -- --template react-ts
   cd app
   npm install
   npm install three echarts onnxruntime-web
   npm install -D @types/three
   ```
   如果 `npm create` 需要交互，就直接手写 `package.json`、`vite.config.ts`、`index.html`
   与 `src/` 下的最少文件，然后 `npm install`。
2. 在 `vite.config.ts` 里设置 `base: "./"`（便于用文件方式打开构建产物）。
3. 写四个页面组件与一个壳：`src/App.tsx`（顶部导航加页面切换）、
   `src/pages/Forward.tsx`、`src/pages/Mor.tsx`、`src/pages/Inverse.tsx`、`src/pages/Gallery.tsx`。
   本轮每页只要**占位内容加一句说明**，不要做视觉设计与真实计算。
4. 写设置面板 `src/components/SettingsPanel.tsx`：三个输入框（接口地址、模型名称、API key）
   加一个"仅本地规则模式"开关。**密钥只放内存变量，禁止写 localStorage 与 sessionStorage**，
   并在面板里写一行提示："密钥只保存在本次页面内存中，关闭页面即丢弃，不会写入本地或上传仓库"。
5. 写一个数据加载模块 `src/data/load.ts`：从 `public/data/manifest.json` 读取资产清单，
   并把 `public/data/` 与清单文件先建好（清单里先放一条示例记录，说明每项字段含义）。
6. 在 `04_results/demo/tools/` 下建四个脚本的**空壳**（只写用途注释与命令行参数说明，
   本轮不实现具体逻辑，后面由项目负责人实现）：
   `export_onnx.py`、`trim_field_basis.py`、`make_web_data.py`、`pack_check.py`。
7. 在 `04_results/demo/README.md` 里写：如何安装依赖、如何本地调试（`npm run dev`）、
   如何构建（`npm run build`）、构建产物在哪里、以及"`node_modules` 不进交付包"的说明。

## 第 3 步：自己验证并报告

1. 在 `04_results/demo/app` 目录里跑 `npm run build`，把输出末尾 10 行贴出来；
2. 报告生成的 `dist/` 体积与文件数；
3. 报告 `04_results/demo/` 下的文件清单与总体积（不含 `node_modules`）；
4. 报告 `04_results/demo/00_PLAN.md` 的行数。

## 硬约束

- **只新建 `04_results/demo/` 下的文件**，不要改动其他任何已有产物。
- 不要做视觉设计、不要做炫酷动画、不要接入真实模型推理（下一轮由更擅长前端的模型做）。
- 不要在代码或任何文件里写死 API key，也不要写示例 key。
- 不要出现"链路""闭环""耦合""回路"，正文不要用 →、≈、≠。
- 若 `npm install` 失败，把报错原文贴出来并停止，不要自己想办法绕。

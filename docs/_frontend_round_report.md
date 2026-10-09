# 前端这一轮的工作汇报

## 做完的部分

四个页面全部做完，按任务书顺序逐个落地。

1. 前向预测 `src/pages/Forward.tsx`
   - 三栏布局。左栏族别切换（四位 M、P、T，五位 L、P、Q、T，Q 用开关），
     再加攻角与来流速度滑块，每个控件旁边有当前值和一句影响说明。
   - 中栏翼型形状，用 `nacaOutline` 画轮廓、弦线、弯度线，前缘后缘有标注，
     底下填渐变。参数一动形状连续变化。
   - 右栏上半压力曲线，横轴归一化弧长 s，纵轴 Cp 反向（负值在上），曲线下填渐变，
     一个按钮把闭环曲线折成上下表面两条弦向曲线叠画。
   - 右栏下半流场云图，u、v、p、nu_t 四个量切换，canvas 渲染，掩膜外留黑，
     翼型描白边，右侧竖直色标标出单位与范围。
   - 底部四张指标卡（八度升力、带内损失、最大马赫数代理、分离代理），
     每张卡有约束线，超限变警告色并写出超出多少，每张卡标明数值来源。
   - 性能上参数变化只算一次 128 维几何，predictCp 用 rAF 节流。
   - b_cp.onnx 与 b_field.onnx 都在，走真实 ONNX 推理加 POD 基还原。
     模型不可用时不崩，退回解析近似并把原因写在图注里。

2. 降维演示 `src/pages/Mor.tsx`
   - 样本选择器支持按名字搜、随机抽、按攻角与雷诺数筛。
     选中后显示真实压力曲线（来自 samples_cp.bin）、几何形状与官方升力。
   - 中间秩滑块 1 到 8，用 cpBasis 与 samples_alpha.bin 的前 8 列重构，
     真实曲线与重构曲线叠画，下面一条逐点误差带，秩变化连续。
   - 右侧八条模态，每条缩略曲线加贡献条，点击单独叠加这条模态看修了什么，
     底部一条累计能量曲线并标出当前秩。
   - 底部上传区支持拖拽与选择文件，json 按 samples.json 结构，csv 列名
     name,uInf,aoa,s,cp，解析后按同一套基投影给出 POD 坐标与重构误差，
     格式不对时给人能看懂的错误。页面写明了 tools/airfrans_to_web.py 的用法。

3. 反向设计与对话 `src/pages/Inverse.tsx`
   - 左栏聊天区有气泡、时间、底部输入框、回车发送。
   - 右栏上面是三步状态块（解析需求、检索候选、出形状与流场确认），
     发一句话之后依次从等待变进行中再变完成。
   - 结果区给四个参数值、翼型形状、压力曲线复核、三项约束通过情况，
     加一个备选列表可以点着换。
   - 在线模式走 OpenAI 兼容接口（设置面板填，只在内存），
     没填时自动走本地规则模式（关键词解析加 fronts 检索），右上角徽标显示当前模式。
     代码里没有任何默认密钥、占位密钥，不写 localStorage 等本地存储。

4. 成果总览 `src/pages/Gallery.tsx`
   - 画廊按 figs/index.json 排列，加载失败显示图名不显示破图。
   - 折中前沿散点（横轴带内损失、纵轴八度升力），悬停显示四个参数和小翼型缩略图，
     数据库最好与理论上界画成参考线。
   - 收敛过程回放（三个种子各一条线），带播放按钮与进度条。
   - 数据集体检卡片区：样本数、族别比例、攻角分布、雷诺数分布、八个划分的规模。

## 共享组件与工具（这一轮新建）

- `src/lib/plot.ts` 轻量 canvas 绘图助手（比例尺、网格、折线、渐变填充、刻度）。
- `src/lib/cases.ts` 案例流场读取（cases_field.bin 按 json 的 blocks 切块）。
- `src/lib/design.ts` 反向设计的本地规则引擎与在线调用。
- `src/components/AirfoilShape.tsx`、`CpCurve.tsx`、`FlowField.tsx`、`SamplePicker.tsx`。
- `src/pages/pages.css` 四个页面共用的布局样式。

## 没动的地基文件

`src/lib/naca.ts`、`src/data/mock.ts`、`src/lib/models.ts`、`src/lib/settings.ts`、
`src/components/SettingsPanel.tsx`、`src/components/DataBadge.tsx`、
`src/theme/tokens.css`、`src/index.css`、`src/App.css`、`src/App.tsx` 全部没改。
`src/data/load.ts` 只加了两件事：把 FrontPoint 等类型 re-export 出去，
以及真实 samples.json 到页面 SampleItem 的适配（params、cp、alpha 三列从二进制按行取，
缺失时退回解析近似），真实数据优先、缺失整页退回示例数据的口径不变。
public/data 下任何文件都没动。

## 构建结果

`npm run build`（tsc -b && vite build）通过，tsc 无错。

产物 dist/ 共 46 个文件，合计 55.60 MB。
其中 onnxruntime 的 wasm 27.00 MB（ort-wasm-simd-threaded.jsep），
页面自身 js/css/html 与 onnxruntime 的 js 加载器约 0.70 MB，
data/ 下的真实数据与图片约 27.91 MB。远低于 200 MB 上限。

## 没做完的部分

- 真实 figs/index.json 已就位，画廊按索引显示；若个别 webp 缺失会显示图名占位。
- 反向设计的在线模式需要评委自己填接口与密钥，本地规则模式已可完整演示。
- 页面在 1366 与 1920 宽下用响应式三栏/两栏布局，窄屏自动堆叠，
  没有横向滚动条；未在真实浏览器里逐像素截图核对（本机无可用截图通道时以构建与类型检查为准）。

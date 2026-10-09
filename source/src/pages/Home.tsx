import './pages.css'
import { assetUrl, type DataState } from '../data/load.ts'

interface Props {
  data: DataState
  go: (page: 'mor' | 'forward' | 'inverse' | 'gallery') => void
}

/** 竞赛要求填写的技术信息，逐条对齐参赛指南第七节的材料要求。 */
const TECH_ROWS: { k: string; v: string }[] = [
  {
    k: 'Agnes Harness 的作用',
    v: '整个作品的智能体运行与执行底座。数据下载与预处理、代理模型训练、'
      + '多目标进化搜索、压力曲线反演、新形状流场预测、演示资产导出与容量核对，'
      + '全部由 AGH 的多个连续会话推进。每一步的工具调用与结果都留在会话账本里，'
      + '可按时间回溯核查。AGH 不是外围辅助，而是核心流程的执行者。',
  },
  {
    k: 'Agnes 模型名称与版本',
    v: 'agnes-3.0-flash。',
  },
  {
    k: '使用环节',
    v: '承担可行的全部执行环节。数据侧完成数据集下载、解压与结构核对。'
      + '建模侧完成压力曲线代理模型与流场代理模型的训练、导出与精度评估。'
      + '设计侧完成三维参数空间的多目标搜索、可行性网格扫描、压力曲线反演。'
      + '交付侧完成演示资产导出、网页数据打包与体积核对。',
  },
  {
    k: '调用方式',
    v: '通过 Agnes Harness 的命令行入口派发，会话按 id 续接，'
      + '单轮以纯文本模式返回。会话内使用文件读写、检索与 shell 工具，'
      + '远端长任务用后台进程提交后轮询日志。每个会话的完整事件账本随作品提交，'
      + '可供人工核查。',
  },
  {
    k: '用到的模型与分工',
    v: 'agnes-3.0-flash 十七次切换，是工作执行的主力。'
      + 'deepseek-flash 七次切换，负责整体规划，并在 agnes 不可用时接手执行。'
      + 'kimi-k3 两次切换，用于网页演示的前端设计与实现。'
      + '三者的调用记录都在会话账本里。',
  },
  {
    k: '数据',
    v: 'AirfRANS 公开数据集，一千条二维 NACA 翼型的 RANS 数值解，'
      + '含表面压力分布、体积流场与官方升力阻力系数。网格与全场解合计约 14.0 GB。',
  },
  {
    k: '代码与专业软件',
    v: '自研 Python 代码完成建模、搜索与反演，用到 NumPy、SciPy、PyTorch、'
      + 'ONNX Runtime、PyVista 与 Matplotlib。前端用 React、TypeScript、'
      + 'ECharts 与 Three.js。没有使用任何商业仿真软件。',
  },
  {
    k: '仿真环境',
    v: '数据集自带的 CFD 解由官方提供，本作品不重复运行 CFD。'
      + '作品自身在浏览器内运行两个神经网络代理模型，'
      + '用 ONNX Runtime 的 WebAssembly 后端完成推理，不依赖服务器。',
  },
  {
    k: '接口调用',
    v: '反向设计页的对话功能可调用 OpenAI 兼容的 chat/completions 接口，'
      + '把自然语言需求转成结构化设计目标。接口地址、模型名与密钥由使用者现场填写，'
      + '只存在于当次页面的内存中，不落盘、不进仓库。不填时自动使用本地规则模式，'
      + '关键词匹配加预计算结果，不联网也能完整演示。',
  },
  {
    k: '设备与远程接口',
    v: '训练与搜索在两块专业显卡的远程工作站上完成，'
      + '通过 SSH 提交任务并回取产物，累计训练与搜索时间约十小时。',
  },
]

/** 竞赛要求填写的基本信息。 */
const BASIC_ROWS: { k: string; v: string }[] = [
  { k: '项目名称', v: '数据驱动的翼型气动代理与反向设计' },
  {
    k: '参考方向',
    v: '科学计算与仿真，含流体仿真与参数反演。'
      + '科研 Agent 与 Harness 工程，含科研流程自动化、数据处理、长任务执行与失败恢复。',
  },
  {
    k: '项目摘要',
    v: '基于 AirfRANS 的一千条二维翼型 RANS 解，先用降维把一条二百五十六点的压力曲线'
      + '压成八个坐标、把体积流场压成每场三十二个模态，再用两个小规模神经网络'
      + '直接预测这些低维坐标，还原成物理量。模型只有二十万参数量级，'
      + '却能在毫秒级给出压力分布与流场。在此之上做反向设计，'
      + '以八度升力最大、工作攻角带内损失最小为目标，'
      + '在四条几何参数上做带约束的多目标搜索，三个随机种子独立收敛到一致结果。'
      + '作品把这些能力做成一个纯浏览器运行的交互演示，'
      + '四个页面分别展示前向预测、降维、反向设计与结果总览，不依赖服务器。',
  },
  {
    k: '问题来源与目标',
    v: '翼型设计里，一次高保真 CFD 计算要几十分钟到几小时，'
      + '而设计过程需要反复试算几百上千次，算力根本跟不上。'
      + '项目要解决的就是这个矛盾：能不能用很小的模型在毫秒级复现数值解。'
      + '同时要回答两个连带问题，模型的结论凭什么可信，'
      + '以及能不能反过来按设计目标直接搜出翼型。',
  },
]

const GLOSSARY: { term: string; text: string }[] = [
  {
    term: 'NACA',
    text: '美国国家航空咨询委员会。它给出的翼型族命名与生成公式至今仍在用，'
      + '四位族用三个数字描述弯度、弯度位置与厚度，五位族用四个数字描述。',
  },
  {
    term: 'RANS',
    text: '雷诺平均纳维-斯托克斯方程。把湍流脉动做时间平均之后的流体方程，'
      + '是工程上最常用的湍流数值求解方法，本数据集的一千条样本就是用它算出来的。',
  },
  {
    term: '攻角',
    text: '来流方向与翼型弦线的夹角，单位是度。攻角越大升力越高，过大则会失速。',
  },
  {
    term: '雷诺数',
    text: '惯性力与黏性力之比，用来刻画流动状态。数值越大流动越容易失稳。',
  },
  {
    term: '压力系数 Cp',
    text: '表面静压除以来流动压，是无量纲量。负值表示当地压力低于来流，也就是吸力。',
  },
  {
    term: 'MOR 与 POD',
    text: '用少数几个模态表示一整条曲线或一整个场，属于降维方法，目的是把数据压小。'
      + '本作品把一条二百五十六点的压力曲线压成八个坐标。',
  },
  {
    term: '代理模型',
    text: '用一个小的数学模型代替昂贵的数值求解。速度快很多，精度有限，'
      + '本作品用它在浏览器里做毫秒级预测。',
  },
  {
    term: '局部马赫数',
    text: '当地流速与声速之比。超过 1 是超音速，接近 1 时会出现激波，阻力骤增。',
  },
  {
    term: '分离',
    text: '边界层脱离翼面。一旦发生，升力会骤降、阻力骤增，是设计上要避开的形态。',
  },
]

const INFO_ROWS: [string, string][] = [
  ['队伍名称', 'PCDLab'],
  ['参赛组别', '硕博组'],
  ['参赛编号', 'G012'],
  ['单位', '苏州大学'],
  ['指导教师', '詹耀辉'],
  ['队伍成员', '张晨晖'],
  ['邮箱', 'sudazhangchenhui@163.com'],
  ['竞赛', '2026 年江苏省 AI+ 科学与工程创新实践黑客松'],
  ['选题方向', '科学计算与仿真（流体仿真与参数反演）'],
]

const CONCLUSIONS: { h: string; t: string }[] = [
  {
    h: '数据可靠，且可低秩压缩',
    t: '一千条样本的四条物理自检全部通过，14 GB 数据异常扫描零命中。'
      + '把表面压力曲线的起点锚定到前缘后，覆盖 99% 变化所需模态从 24 降到 5；'
      + '用 8 个模态重构未见样本的平均相对误差 4.81%，训练与测试的误差比值为 1.01，未见过拟合。',
  },
  {
    h: '降维坐标回归可行，优势来自计算范式的改变',
    t: '主路线只用 133 个输入数字回归低维坐标，场量精度优于均值场 2.4–4.4 倍、'
      + '优于岭回归 1.4–2.2 倍，升力精度与官方逐点大模型在同一量级；'
      + '单样本纯推理快约 5750 倍，训练成本低 689 倍。'
  },
  {
    h: '几何到流场的影响可以分三层刻画',
    t: '厚度是全局第一影响量，真实数据与模型扰动两条独立路径一致。'
      + '局部主导段在 8 个组合中只有 1 个一致，我们如实报出 7 处不一致；'
      + '模型给出的厚度与后缘压力关系方向与真实数据相反，'
      + '所以凡涉及厚度的结论一律回到真实样本复核。',
  },
  {
    h: '形状搜索能搜出可用设计，但必须连同代价一起给出',
    t: '三个随机种子独立收敛到同一片区域，设计攻角升力 1.347–1.351，'
      + '高于数据库同工况最优真实翼型的 1.146；代价是带内损失约 0.37 以上。'
      + '80 个前沿解中只有 8 个不贴参数边界，真正起作用的约束是分离代理。',
  },
  {
    h: '从压力反推外形严重多解，不能当作形状恢复',
    t: '11 个目标的压力拟合误差全部低于「模型地板」（用真形状也算不到的误差），'
      + '说明反演在补偿模型误差，而不是在恢复真实几何。'
      + '1% 容差下每个目标仍有大量等价候选，参数甚至会整体翻面。',
  },
  {
    h: '模型的排序可用，绝对值不可用',
    t: '官方留出集（模型完全没见过的 89 条）上升力排序相关 0.9936，'
      + '全部 511 条五位族样本上 0.9834，说明模型完全能分辨哪个翼型升力更高；'
      + '但绝对升力偏差在分布内就有约 ±1%。'
      + '因此本文所有结论都写成相对比较，绝对数值一律标注未验证。',
  },
]

export default function Home({ data, go }: Props) {
  const ds = data.results.dataset
  return (
    <section className="page page-in">
      <div className="page-head">
        <h2>数据驱动的翼型气动代理与反向设计</h2>
        {data.mode === 'mock' && <span className="badge mock">示例数据</span>}
      </div>

      <div className="panel">
        <div className="panel-title">参赛信息</div>
        <div className="panel-body" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: 'var(--sp-3)',
        }}>
          {INFO_ROWS.map(([k, v]) => (
            <div key={k} style={{
              background: 'var(--bg-3)',
              border: '1px solid var(--line)',
              borderRadius: 'var(--rd-1)',
              padding: 'var(--sp-2) var(--sp-3)',
            }}>
              <div style={{ fontSize: 'var(--fs-1)', color: 'var(--fg-2)' }}>{k}</div>
              <div style={{ fontSize: 'var(--fs-3)', color: 'var(--fg-0)', fontWeight: 600 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">项目基本信息</div>
        <div className="panel-body" style={{ display: 'grid', gap: 12 }}>
          {BASIC_ROWS.map((r) => (
            <div key={r.k} style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: 12 }}>
              <span style={{ fontWeight: 700, color: 'var(--acc-strong)', fontSize: 14 }}>{r.k}</span>
              <span style={{ color: 'var(--fg-0)', fontSize: 15, lineHeight: 1.75 }}>{r.v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 背景知识：为什么难、NACA 是什么。图片与介绍均来自 Wikipedia。 */}
      <div className="panel">
        <div className="panel-title">
          <span>背景知识：为什么算一个翼型这么贵，NACA 翼型又是什么</span>
          <span className="muted" style={{ fontSize: 12 }}>图片与介绍来源：Wikipedia</span>
        </div>
        <div className="panel-body" style={{ display: 'grid', gap: 'var(--sp-4)', fontSize: 15, lineHeight: 1.8 }}>
          <p style={{ margin: 0 }}>
            本作品处理的是<span style={{ color: 'var(--warm)' }}>二维翼型绕流</span>。
            要知道一个翼型的升力与阻力，工程上的做法是解
            <strong style={{ color: 'var(--acc-strong)' }}> 纳维–斯托克斯方程</strong>
            （描述流体运动的非线性偏微分方程，由质量守恒与动量守恒推出）。
            它难在三点：<strong>非线性</strong>（对流项 u·∇u 本身含未知量的乘积）、
            <strong>多尺度</strong>（湍流涡从米级一直垒到毫米级以下）、
            <strong>强耦合</strong>（速度与压力必须联立求解，没有显式解）。
            直接数值模拟的代价随雷诺数增长极快，工程上只能转用
            <strong style={{ color: 'var(--acc-strong)' }}>RANS</strong>
            （雷诺平均，把湍流脉动做时间平均）加湍流模型来近似，
            即使如此，<span style={{ color: 'var(--warm)' }}>算一条二维翼型仍需要几十分钟到几小时的 CPU 机时</span>。
            后续所有“降维 + 代理模型 + 反向设计”的工作，都是为了把这次昂贵计算
            在必要时换成毫秒级的快速估算。
          </p>

          <div className="home-figs">
            <figure className="home-fig">
              <img src={assetUrl('figs/naca_777_takeoff.jpg')} alt="客机起飞" loading="lazy" />
              <figcaption>
                全日空航空波音 777-300ER 从纽约肯尼迪国际机场起飞。
                机翼能产生升力，靠的正是绕流场的压力分布，而压力分布由纳维–斯托克斯方程决定。
                <span className="home-src">图片来源：Wikipedia</span>
              </figcaption>
            </figure>
            <figure className="home-fig">
              <img src={assetUrl('figs/turbulent_jet.jpg')} alt="湍流可视化" loading="lazy" />
              <figcaption>
                平面激光诱导荧光可视化下的湍流。这类尺度干扰是纳维–斯托克斯方程难以求解的直接原因，
                也是 RANS 必须先做平均的原因。
                <span className="home-src">图片来源：Wikipedia</span>
              </figcaption>
            </figure>
            <figure className="home-fig">
              <img src={assetUrl('figs/airfoil_geometry.svg')} alt="翼型几何参数" loading="lazy" />
              <figcaption>
                翼型参数：1 零升力线；2 前缘；3 鼻圆；4 最大厚度；5 弯度；
                6 上表面；7 后缘；8 中弧线；9 下表面。
                <span className="home-src">图片与图注来源：Wikipedia</span>
              </figcaption>
            </figure>
          </div>

          <div style={{ display: 'grid', gap: 8, borderTop: '1px solid var(--line-soft)', paddingTop: 'var(--sp-3)' }}>
            <div style={{ fontWeight: 700, color: 'var(--acc-strong)' }}>什么是 NACA 翼型</div>
            <p style={{ margin: 0 }}>
              <strong>NACA 翼型</strong>是美国国家航空咨询委员会（NACA）开发的一系列翼型。
              每个翼型的代号由“NACA”四个字母加一串数字组成，
              <span style={{ color: 'var(--warm)' }}>把这串数字描述的几何参数代入特定方程，就能得到翼型的精确形状</span>。
              按数字位数分为四位数翼型（如 NACA 2412）与五位数翼型（如 NACA 23112）。
            </p>
            <p style={{ margin: 0 }}>
              以四位数翼型为例，三个参数的含义是：
              <strong>首位数字</strong>是最大弯度占弦长的百分比；
              <strong>第二位数字</strong>是最大弯度距前缘的位置占弦长的十分之几；
              <strong>后两位数字</strong>是最大厚度占弦长的百分比。
              例如 <span className="num" style={{ color: 'var(--acc-strong)' }}>NACA 2412</span>：
              最大弯度为弦长的 2%，位于距前缘 0.4 弦长处，最大厚度为弦长的 12%
              （四位数翼型的最大厚度默认在距前缘 0.3 弦长处）。
              五位数翼型能描述更复杂的形状，多出的一位用来指定弯度曲线型。
            </p>
            <p style={{ margin: 0 }} className="muted">
              本作品用到的 AirfRANS 数据集就是这类翼型：四位族使用三个数字，五位族使用四个数字。
              我们的模型把翼型参数当输入直接生成几何，正是因为有这套解析式。
            </p>
            <p style={{ margin: 0, fontSize: 13 }} className="muted">
              以上 NACA 介绍与三张图片均来自 Wikipedia（中文页「NACA 翼型」与对应英文条目）。
            </p>
          </div>
        </div>
      </div>

      <div className="cols-2">
        <div className="panel">
          <div className="panel-title">研究的问题</div>
          <div className="panel-body" style={{ display: 'grid', gap: 8, fontSize: 15, lineHeight: 1.7 }}>
            <p style={{ margin: 0 }}>我们做数据驱动的翼型气动代理与反向设计，回答三个问题。</p>
            <p style={{ margin: 0 }}>一，能不能用很小的模型在毫秒级复现数值流场解。</p>
            <p style={{ margin: 0 }}>二，模型的结论凭什么可信，误差边界在哪里。</p>
            <p style={{ margin: 0 }}>三，能不能反过来，按设计目标直接搜出满足约束的翼型。</p>
          </div>
        </div>
        <div className="panel">
          <div className="panel-title">数据集来源</div>
          <div className="panel-body" style={{ display: 'grid', gap: 8, fontSize: 15, lineHeight: 1.7 }}>
            <p style={{ margin: 0 }}>
              AirfRANS 是公开数据集，含一千条二维 NACA 翼型的 RANS 数值解。
              四位族 {ds.nFourDigit} 条，五位族 {ds.nFiveDigit} 条。
            </p>
            <p style={{ margin: 0 }}>
              来流速度 {ds.uInfRange[0]} 到 {ds.uInfRange[1]} 米每秒，
              攻角 {ds.aoaRange[0]} 到 {ds.aoaRange[1]} 度，
              雷诺数 {(ds.reRange[0] / 1e6).toFixed(0)}00 万到 {(ds.reRange[1] / 1e6).toFixed(0)}00 万。
            </p>
            <p style={{ margin: 0 }} className="muted">
              官方仓库与论文：DAIS 实验室的 AirfRANS 项目（GitHub 仓库 Extrality/AirfRANS，
              论文 AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating
              Reynolds-Averaged Navier–Stokes Solutions）。
            </p>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">研究思路</div>
        <div className="panel-body" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 'var(--sp-4)' }}>
          <div>
            <h3 style={{ color: 'var(--acc-strong)', fontSize: 16, marginBottom: 6 }}>一 降维</h3>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.7 }}>
              用本征正交分解把一千条压力曲线与流场拆成少量模态。
              每条样本只剩下一小串模态坐标，数据量从 14 GB 压到十几 MB。
            </p>
          </div>
          <div>
            <h3 style={{ color: 'var(--acc-strong)', fontSize: 16, marginBottom: 6 }}>二 预测</h3>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.7 }}>
              用小型的全连接网络从来流条件与几何曲线直接预测模态坐标，
              再乘模态基还原成物理量。两个模型合计 2.55 MB，纯浏览器可跑。
            </p>
          </div>
          <div>
            <h3 style={{ color: 'var(--acc-strong)', fontSize: 16, marginBottom: 6 }}>三 反向设计</h3>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.7 }}>
              在代理模型上跑多目标进化搜索，八度升力尽量高，
              四度和十度不许掉太多，同时满足马赫数与分离约束。
            </p>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">
          <span>主要结论</span>
          <span className="muted" style={{ fontSize: 12 }}>六条，每条都是对「做成了什么」的断言</span>
        </div>
        <div className="panel-body" style={{ display: 'grid', gap: 14 }}>
          {CONCLUSIONS.map((c, i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '26px 1fr', gap: 10 }}>
              <span className="num" style={{
                color: 'var(--acc-strong)', fontWeight: 700, fontSize: 16, lineHeight: 1.5,
              }}>{i + 1}</span>
              <div style={{ display: 'grid', gap: 3 }}>
                <span style={{ color: 'var(--fg-0)', fontWeight: 700, fontSize: 15, lineHeight: 1.5 }}>
                  {c.h}
                </span>
                <span style={{ color: 'var(--fg-1)', fontSize: 14, lineHeight: 1.75 }}>{c.t}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">技术信息与模型分工</div>
        <div className="panel-body" style={{ display: 'grid', gap: 12 }}>
          {TECH_ROWS.map((r) => (
            <div key={r.k} style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: 12 }}>
              <span style={{ fontWeight: 700, color: 'var(--acc-strong)', fontSize: 14 }}>{r.k}</span>
              <span style={{ color: 'var(--fg-0)', fontSize: 15, lineHeight: 1.75 }}>{r.v}</span>
            </div>
          ))}
          <p style={{ margin: 0, fontSize: 13, color: 'var(--fg-2)', lineHeight: 1.7 }}>
            以上每一条都能在随作品提交的会话账本里按时间查到对应的工具调用与产物。
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 'var(--sp-3)' }}>
        {([
          ['mor', '降维演示', '一千条数据怎么用十几个数字概括'],
          ['forward', '前向预测', '改参数，压力分布与流场马上跟着变'],
          ['inverse', '反向设计', '说一句话就能得到一版设计'],
          ['gallery', '成果总览', '前沿、收敛与数据集体检'],
        ] as const).map(([key, title, desc]) => (
          <button key={key} onClick={() => go(key)} style={{
            textAlign: 'left',
            padding: 'var(--sp-4)',
            background: 'linear-gradient(180deg, var(--bg-2), var(--bg-1))',
            border: '1px solid var(--acc-line)',
            borderRadius: 'var(--rd-2)',
            display: 'grid',
            gap: 4,
          }}>
            <span style={{ fontSize: 17, fontWeight: 700, color: 'var(--acc-strong)' }}>{title} →</span>
            <span style={{ fontSize: 13, color: 'var(--fg-1)' }}>{desc}</span>
          </button>
        ))}
      </div>

      <div className="panel" style={{ marginTop: 'var(--sp-4)' }}>
        <div className="panel-title">名词解释</div>
        <div className="panel-body" style={{ display: 'grid', gap: 10 }}>
          {GLOSSARY.map((g) => (
            <div key={g.term} style={{ display: 'grid', gridTemplateColumns: '132px 1fr', gap: 10 }}>
              <span style={{ fontWeight: 700, color: 'var(--acc-strong)' }}>{g.term}</span>
              <span style={{ color: 'var(--fg-1)' }}>{g.text}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

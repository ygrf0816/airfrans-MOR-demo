// 可解释性分析页。数据来自 04_results/explain 的落盘结果，
// 由 tools/make_explain_data.py 压成 explain.json，页面只做展示、不重算。

import { useEffect, useMemo, useRef, useState } from 'react'
import './pages.css'
import { assetUrl } from '../data/load.ts'
import {
  colors, makeScale, niceTicks, setupCanvas, strokeLine, drawGrid, type Pt,
} from '../lib/plot.ts'

interface FamilyStat { n: number | null; data: number[]; model: number[]; modelAmp: number[] }
interface GlobalPart {
  components: string[]
  labels: Record<string, string>
  families: Record<string, FamilyStat>
}
interface LocalCurves { data: number[]; model: number[] }
interface LocalFamily { x: number[]; curves: Record<string, LocalCurves>; n: number }
interface VerdictRow {
  family: string; comp: string; dataDom: string; modelDom: string
  dataDomLabel: string; modelDomLabel: string; agree: boolean
}
interface LocalPart {
  labels: Record<string, string>
  families: Record<string, LocalFamily>
  verdict: VerdictRow[]
  segLabels: Record<string, string>
}
interface Interaction {
  family: string; response: string; term: string
  responseLabel?: string; termRaw?: string
  labelA?: string; labelB?: string
  coef: number; t: number; ratio: number; negligible: boolean; n: number
}
interface TheoryPart {
  slope: {
    theoryThin: number; theoryPgRef: number; bandLo: number; bandHi: number
    bandNote: string
    families: Record<string, {
      n: number; raw: number; rawR2: number; ctrl: number; ctrlSe: number
      model: number; modelR2: number; modelN: number
    }>
  }
  suction: Record<string, {
    dataSlope: number; dataR2: number; nData: number
    modelSlope: number; modelR2: number; nModel: number
  }>
  te: Record<string, {
    n: number; coef: number; r: number; rPartial: number; cpTeMean: number
    modelHalf: number; nModel: number
  }>
}
interface ScatterPart {
  n: number; skipped: number
  family: number[]; aoa: number[]; uinf: number[]; cl: number[]; cd: number[]
  components: string[]; labels: Record<string, string>
  values: Record<string, number[]>
  band: { lo: number; hi: number; n: number } | null
}
interface Top10Row {
  rank: number; name: string; family: string; uinf: number; aoa: number
  cl: number; cd: number; mLocal: number; ycMid: number; thPeakHalf: number
  thMaxXc: number; reRadius: number
}
interface DesignPart {
  rows: Top10Row[]
  dev: { comp: string; label: string; poolMedian: number; top10Mean: number; devMean: number }[]
}
interface ExplainData {
  source: string
  global: GlobalPart
  local: LocalPart
  interaction: Interaction[]
  theory: TheoryPart
  scatter: ScatterPart
  design: DesignPart
  /** 图 5-2 的三组真实数据，用于前端矢量绘制。 */
  figure52?: {
    stationIdx: number[]
    stationXChord: number[]
    mu: number[]
    spectrum: number[]
  }
}

const FAM_LABEL: Record<string, string> = { '4digit': '四位族', '5digit': '五位族' }

/** 受影响对象的中文名。 */
const RESP_LABEL: Record<string, string> = { CL: '升力', LE_DEP: '前缘段压力' }

/**
 * 迷你折线：纵轴自适应。marks 给定时在这些下标处画点并标出。
 * 不用 hook，配合 useEffect / ResizeObserver 调用。
 */
function drawMiniCurve(
  el: HTMLCanvasElement | null, ys: number[], stroke: string,
  marks: number[], xs: number[] | null,
) {
  if (!el) return
  const box = setupCanvas(el, 138)
  if (!box) return
  const { ctx, w, h } = box
  const C = colors()
  const pad = { l: 40, r: 10, t: 10, b: 20 }
  const n = ys.length
  if (!n) return
  const ymin = Math.min(...ys), ymax = Math.max(...ys)
  const span = (ymax - ymin) || 1
  const sx = (i: number) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r)
  const sy = (v: number) => pad.t + (1 - (v - ymin) / span) * (h - pad.t - pad.b)

  ctx.strokeStyle = C.grid
  ctx.lineWidth = 1
  for (let g = 0; g <= 3; g++) {
    const yy = pad.t + (g / 3) * (h - pad.t - pad.b)
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke()
  }
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    const px = sx(i), py = sy(ys[i])
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.lineWidth = 1.7
  ctx.strokeStyle = stroke
  ctx.shadowColor = stroke
  ctx.shadowBlur = 5
  ctx.stroke()
  ctx.shadowBlur = 0

  ctx.font = '10px "JetBrains Mono", monospace'
  ctx.fillStyle = C.fg2
  const top = ymax >= 1e4 || ymax <= 1e-3 ? ymax.toExponential(1) : ymax.toPrecision(3)
  const bot = ymin >= 1e4 || ymin <= 1e-3 ? ymin.toExponential(1) : ymin.toPrecision(3)
  ctx.fillText(String(top), 3, pad.t + 9)
  ctx.fillText(String(bot), 3, h - pad.b + 3)

  for (const i of marks) {
    if (i >= n) continue
    const px = sx(i), py = sy(ys[i])
    ctx.beginPath(); ctx.arc(px, py, 3.2, 0, Math.PI * 2)
    ctx.fillStyle = C.ok; ctx.fill()
    ctx.fillStyle = C.fg
    const lab = String(i)
    ctx.fillText(lab, Math.min(px + 4, w - 30), Math.max(py - 5, pad.t + 8))
  }
  void xs
}

function Verdict({ ok, text }: { ok: boolean; text: string }) {
  return <span className={ok ? 'ex-chip ok' : 'ex-chip bad'}>{text}</span>
}


/** 画布封装：按容器宽度重画，draw 用 ref 持有，避免每次渲染都重建观察器。 */
function useCanvas(height: number, deps: unknown[], draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
  const ref = useRef<HTMLCanvasElement>(null)
  const drawRef = useRef(draw)
  drawRef.current = draw
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const run = () => {
      const box = setupCanvas(el, height)
      if (box) drawRef.current(box.ctx, box.w, box.h)
    }
    run()
    const ro = new ResizeObserver(run)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return ref
}

/** 全局依赖：四个分量的影响大小，数据侧带符号、模型侧为位移范数。 */
function GlobalChart({ stat, labels, components, mode }: {
  stat: FamilyStat; labels: Record<string, string>; components: string[]
  mode: 'data' | 'model'
}) {
  const values = mode === 'data' ? stat.data : stat.model
  const ref = useCanvas(230, [values, labels, components, mode], (ctx, w, h) => {
    const C = colors()
    const pad = { l: 58, r: 12, t: 16, b: 48 }
    const lo = Math.min(0, ...values)
    const hi = Math.max(0, ...values)
    const span = hi - lo || 1
    const y0 = lo - span * 0.12
    const y1 = hi + span * 0.18
    const sx = makeScale([-0.5, components.length - 0.5], [pad.l, w - pad.r])
    const sy = makeScale([y0, y1], [h - pad.b, pad.t])
    drawGrid(ctx, w, h, pad, [], niceTicks(y0, y1, 5, (v) => v.toFixed(1)), sx, sy)
    const zero = sy(0)
    const bw = (w - pad.l - pad.r) / components.length * 0.46
    // 注意要先取绝对值再找下标：直接拿绝对值去有符号数组里 indexOf
    // 在负值最大时会返回 -1，高亮就静默失效了。
    const amps = values.map((v) => Math.abs(v))
    const best = amps.indexOf(Math.max(...amps))
    components.forEach((c, i) => {
      const v = values[i]
      const x = sx(i)
      const top = Math.min(zero, sy(v))
      const hgt = Math.abs(sy(v) - zero)
      ctx.fillStyle = i === best ? C.acc : (v < 0 ? C.bad : 'rgba(34,211,238,0.42)')
      ctx.fillRect(x - bw / 2, top, bw, Math.max(1, hgt))
      ctx.fillStyle = C.fg
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText(v.toFixed(2), x, v >= 0 ? top - 5 : top + hgt + 12)
      ctx.fillStyle = C.fg2
      ctx.font = '11px "PingFang SC", "Microsoft YaHei", sans-serif'
      ctx.fillText(labels[c] ?? c, x, h - pad.b + 16)
      ctx.fillStyle = C.fg2
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillText(c, x, h - pad.b + 30)
    })
    ctx.strokeStyle = C.axis
    ctx.beginPath()
    ctx.moveTo(pad.l, zero)
    ctx.lineTo(w - pad.r, zero)
    ctx.stroke()
  })
  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

/** 局部影响：沿弦向的逐站位敏感性，数据侧与模型侧两条腿同图对比。 */
function LocalChart({ fam, comp, segLabels }: {
  fam: LocalFamily; comp: string; segLabels: Record<string, string>
}) {
  const curve = fam.curves[comp]
  const ref = useCanvas(240, [curve, comp, segLabels], (ctx, w, h) => {
    const C = colors()
    const pad = { l: 54, r: 12, t: 18, b: 40 }
    const xs = fam.x
    const d = curve.data
    const m = curve.model
    const lo = Math.min(...d, ...m)
    const hi = Math.max(...d, ...m)
    const span = hi - lo || 1
    const y0 = lo - span * 0.12
    const y1 = hi + span * 0.22
    const sx = makeScale([0, 1], [pad.l, w - pad.r])
    const sy = makeScale([y0, y1], [h - pad.b, pad.t])

    // 三段底色：前缘 0–0.2，中弦 0.2–0.8，后缘 0.8–1
    const bands: [number, number, string][] = [
      [0, 0.2, 'rgba(34,211,238,0.07)'],
      [0.2, 0.8, 'rgba(148,163,184,0.05)'],
      [0.8, 1, 'rgba(251,146,60,0.07)'],
    ]
    for (const [a, b, fill] of bands) {
      ctx.fillStyle = fill
      ctx.fillRect(sx(a), pad.t, sx(b) - sx(a), h - pad.t - pad.b)
      ctx.strokeStyle = 'rgba(148,163,184,0.18)'
      ctx.beginPath()
      ctx.moveTo(sx(b), pad.t)
      ctx.lineTo(sx(b), h - pad.b)
      ctx.stroke()
    }
    const xTicks = niceTicks(0, 1, 6, (v) => v.toFixed(2))
    const yTicks = niceTicks(y0, y1, 5, (v) => v.toFixed(3))
    drawGrid(ctx, w, h, pad, xTicks, yTicks, sx, sy)
    for (const [a, b, fill] of bands) {
      ctx.fillStyle = fill
      ctx.fillRect(sx(a), pad.t, sx(b) - sx(a), h - pad.t - pad.b)
    }
    const zero = sy(0)
    ctx.strokeStyle = 'rgba(148,163,184,0.35)'
    ctx.beginPath()
    ctx.moveTo(pad.l, zero)
    ctx.lineTo(w - pad.r, zero)
    ctx.stroke()

    const toPts = (arr: number[]): Pt[] =>
      arr.map((v, i) => ({ x: sx(xs[i]), y: sy(v) }))
    strokeLine(ctx, toPts(d), C.acc, 1.9)
    strokeLine(ctx, toPts(m), C.warm, 1.6)

    // 段名标在顶部
    ctx.font = '10px "PingFang SC", "Microsoft YaHei", sans-serif'
    ctx.textAlign = 'center'
    ctx.fillStyle = C.fg2
    ctx.fillText(segLabels.front, sx(0.1), pad.t + 11)
    ctx.fillText(segLabels.mid, sx(0.5), pad.t + 11)
    ctx.fillText(segLabels.rear, sx(0.9), pad.t + 11)

    // 图例靠右放，左上角留给段名，免得两者挤在一起
    ctx.textAlign = 'left'
    const legend = [['办法一 · 真实数据', C.acc], ['办法二 · 扰动模型', C.warm]] as const
    const legendW = 176
    legend.forEach(([text, color], i) => {
      const lx = w - pad.r - legendW * (2 - i) + 6
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(lx, pad.t - 7)
      ctx.lineTo(lx + 16, pad.t - 7)
      ctx.stroke()
      ctx.fillStyle = C.fg2
      ctx.font = '10px "PingFang SC", "Microsoft YaHei", sans-serif'
      ctx.fillText(text, lx + 21, pad.t - 3)
    })
    ctx.fillStyle = C.fg2
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'center'
    ctx.fillText('展平后的弦向位置 x/c', pad.l + (w - pad.l - pad.r) / 2, h - 6)
  })
  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

/** 升力随几何量的散布，用来把设计策略画出来。 */
function ScatterChart({ part, keyName, label }: {
  part: ScatterPart; keyName: string; label: string
}) {
  const xs = keyName === 'aoa' ? part.aoa
    : keyName === 'uinf' ? part.uinf
      : part.values[keyName]
  const ref = useCanvas(280, [part, keyName], (ctx, w, h) => {
    const C = colors()
    const pad = { l: 58, r: 12, t: 16, b: 42 }
    const x0 = Math.min(...xs)
    const x1 = Math.max(...xs)
    const y0 = Math.min(...part.cl)
    const y1 = Math.max(...part.cl)
    const px = (x1 - x0) * 0.04 || 0.01
    const py = (y1 - y0) * 0.06 || 0.01
    const sx = makeScale([x0 - px, x1 + px], [pad.l, w - pad.r])
    const sy = makeScale([y0 - py, y1 + py], [h - pad.b, pad.t])
    drawGrid(
      ctx, w, h, pad,
      niceTicks(x0 - px, x1 + px, 6, (v) => (Math.abs(v) < 0.01 ? v.toExponential(0) : v.toFixed(3))),
      niceTicks(y0 - py, y1 + py, 5, (v) => v.toFixed(2)),
      sx, sy,
    )
    for (let i = 0; i < part.n; i++) {
      ctx.fillStyle = part.family[i] === 1 ? 'rgba(34,211,238,0.5)' : 'rgba(251,146,60,0.5)'
      ctx.beginPath()
      ctx.arc(sx(xs[i]), sy(part.cl[i]), 1.7, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.fillStyle = C.fg2
    ctx.textAlign = 'center'
    ctx.fillText(label, pad.l + (w - pad.l - pad.r) / 2, h - 8)
    ctx.textAlign = 'left'
    const legend = [['四位族', 'rgba(251,146,60,0.85)'], ['五位族', 'rgba(34,211,238,0.85)']] as const
    legend.forEach(([text, color], i) => {
      const lx = pad.l + 6 + i * 74
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(lx, pad.t - 5, 3.4, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = C.fg2
      ctx.font = '10px "PingFang SC", "Microsoft YaHei", sans-serif'
      ctx.fillText(text, lx + 8, pad.t - 1)
    })
  })
  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

/** 判定角标。 */
/**
 * 图 5-2：位置认识改正的三条依据。全部由真实数据在前端矢量绘制，不用静态图片。
 *   (a) 曲线序号 -> 真实弦向位置，证明序号两端都在前缘
 *   (b) 一千条样本的平均压力曲线，看两端的吸气峰
 *   (c) 奇异值谱，证明被误当成坐标的那一列其实是谱
 */
function StationEvidence({ d }: {
  d: { stationIdx: number[]; stationXChord: number[]; mu: number[]; spectrum: number[] }
}) {
  const refA = useRef<HTMLCanvasElement>(null)
  const refB = useRef<HTMLCanvasElement>(null)
  const refC = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const draw = () => {
      drawMiniCurve(refA.current, d.stationXChord, '#22d3ee', [0, 127, 255], d.stationIdx)
      drawMiniCurve(refB.current, d.mu, '#22d3ee', [], null)
      drawMiniCurve(refC.current, d.spectrum, '#fb923c', [], null)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ;[refA, refB, refC].forEach((r) => r.current?.parentElement && ro.observe(r.current.parentElement))
    return () => ro.disconnect()
  }, [d])

  return (
    <div className="ex-figdata">
      <div className="ex-figdata-grid">
        <div className="ex-figdata-cell">
          <div className="ex-figdata-cap">(a) 曲线序号折算回真实位置</div>
          <canvas ref={refA} />
          <div className="ex-figdata-note">
            横轴为曲线序号 0–255，纵轴为真实弦向位置 0–1。
            序号 <b>0</b> 与 <b>255</b> 都落在 0（前缘），序号 <b>127</b> 落在 1（后缘）。
            这就说明压力曲线是绕机翼一圈的闭合曲线，不能当直线读。
          </div>
        </div>
        <div className="ex-figdata-cell">
          <div className="ex-figdata-cap">(b) 一千条样本的平均压力曲线</div>
          <canvas ref={refB} />
          <div className="ex-figdata-note">
            纵轴为压力系数 Cp（<b>越负越靠上</b>，气动惯例）。曲线两端各出现一次吸气峰，
            最负的一处在序号 255，也就是前缘处闭合的位置。
          </div>
        </div>
        <div className="ex-figdata-cell">
          <div className="ex-figdata-cap">(c) 被误当成坐标的那一列</div>
          <canvas ref={refC} />
          <div className="ex-figdata-note">
            这一列单调不增、末值仅 <b>9.67×10⁻⁶</b>，一路降了七个数量级，
            是降维算出的<b>奇异值谱</b>，并不是位置坐标。
            早期工作把它当成弦向坐标，才得出「四个量的影响都集中在后缘」的错误结论。
          </div>
        </div>
      </div>
    </div>
  )
}


export default function Explain() {
  const [data, setData] = useState<ExplainData | null>(null)
  const [failed, setFailed] = useState(false)
  const [fam, setFam] = useState('4digit')
  const [comp, setComp] = useState('th_peak_half')
  const [xKey, setXKey] = useState('th_peak_half')

  useEffect(() => {
    let alive = true
    fetch(assetUrl('explain.json'), { cache: 'no-cache' })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: ExplainData) => { if (alive) setData(j) })
      .catch(() => { if (alive) setFailed(true) })
    return () => { alive = false }
  }, [])

  // 数据规模小，判定分组一次算好。
  const grouped = useMemo(() => {
    if (!data) return null
    const g: Record<string, Interaction[]> = {}
    for (const it of data.interaction) {
      const key = it.family + '|' + it.response
      g[key] = g[key] ?? []
      g[key].push(it)
    }
    return g
  }, [data])

  if (failed) {
    return (
      <section className="page page-in">
        <div className="page-head"><h2>可解释性分析</h2></div>
        <div className="panel">
          <div className="panel-body">
            没有找到 <code>data/explain.json</code>，本页无法显示。
            运行 <code>tools/make_explain_data.py</code> 重新生成该文件。
          </div>
        </div>
      </section>
    )
  }
  if (!data || !grouped) {
    return (
      <section className="page page-in">
        <div className="page-head"><h2>可解释性分析</h2></div>
        <div className="skeleton" style={{ height: 220 }} />
      </section>
    )
  }

  const glob = data.global
  const stat = glob.families[fam]
  const localFam = data.local.families[fam]
  const verdict = data.local.verdict.filter((v) => v.family === fam)
  const agreeCount = data.local.verdict.filter((v) => v.agree).length

  return (
    <section className="page page-in">
      <div className="page-head">
        <h2>可解释性分析</h2>
        <span className="hint">
          几何形状怎么改变流场、改多少、改在哪一段
        </span>
      </div>

      <div className="panel">
        <div className="panel-title">这一页在问什么</div>
        <div className="panel-body" style={{ display: 'grid', gap: 10, fontSize: 15, lineHeight: 1.85 }}>
          <p style={{ margin: 0 }}>
            前面几页回答的是模型准不准、能不能反着用。这一页回答另一个问题：
            几何形状究竟怎么影响流场——<b>哪一项影响最大、改在机翼的哪一段、
            这些影响和流体力学常识对不对得上</b>。
          </p>
          <p style={{ margin: 0 }}>
            为了把话说清楚，先把翼型形状拆成<b>四个可以分别讨论的量</b>：
            <b>厚度</b>（机翼有多厚）、<b>弯度</b>（中间鼓起多少）、
            <b>最厚的位置</b>（最厚处靠前还是靠后）、
            <b>前缘半径</b>（机翼最前端是尖还是钝）。
          </p>
          <p style={{ margin: 0 }}>
            然后用<b>两套互不相干的办法</b>各算一遍，看它们是否给出同样的答案：
          </p>
          <div className="ex-method">
            <div className="ex-method-col">
              <div className="ex-method-h" style={{ color: 'var(--acc)' }}>办法一 · 看真实数据</div>
              直接拿一千条真实 CFD 样本做统计，不经过任何模型。
              <span className="muted">好处是数字来自真值，不受模型自身缺陷影响。</span>
            </div>
            <div className="ex-method-col">
              <div className="ex-method-h" style={{ color: 'var(--acc-strong)' }}>办法二 · 扰动模型</div>
              把基准形状的每个量分别加大一点、减小一点，看模型预测的流场变化多大。
              <span className="muted">好处是只动了这一个量，能单独看它的作用。</span>
            </div>
          </div>
          <p style={{ margin: 0 }}>
            两套办法<b>对得上的才下结论</b>，对不上的如实报出。这是本章最重要的一条纪律：
            不一致本身就是结论的一部分，不做修饰。下面每一条结论都配有论文里的插图，
            图注写明这张图能佐证什么。
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">一 哪一项几何量最要紧</div>
        <div className="panel-body">
          <p className="ex-note" style={{ marginTop: 0 }}>
            把四个量的影响摆在一起比。两边用同一套横轴，看柱子高低即可。
          </p>
          <div className="seg" style={{ marginBottom: 10 }}>
            {Object.keys(glob.families).map((k) => (
              <button key={k} className={fam === k ? 'active' : ''} onClick={() => setFam(k)}>
                {FAM_LABEL[k]} n={glob.families[k].n}
              </button>
            ))}
          </div>
          <div className="cols-2" style={{ alignItems: 'start' }}>
            <div>
              <div className="ex-sub">办法一 · 真实数据：每个量对升力的影响（负号＝这个量变大，升力变小）</div>
              <GlobalChart stat={stat} labels={glob.labels} components={glob.components} mode="data" />
            </div>
            <div>
              <div className="ex-sub">办法二 · 扰动模型：把每个量加减一点后，模型的预测变化多少</div>
              <GlobalChart stat={stat} labels={glob.labels} components={glob.components} mode="model" />
            </div>
          </div>
          <p className="ex-note">
            <b>厚度</b>在两边都是最高的柱子（真实数据侧 {stat.data[1].toFixed(2)}、模型侧 {stat.model[1].toFixed(2)}），
            这是本页<b>唯一两套办法都同意的量级结论</b>，也是最可靠的一条：
            机翼越厚，上表面的低压吸气区被压得越浅。
          </p>
          <p className="ex-note">
            <b>前缘半径</b>在真实数据里排第二、在扰动模型里却掉到第三，两边排序不一致，如实列出。
            但方向两边一致：前缘越钝，吸气峰越弱、压力恢复越慢，这与薄翼理论的预期相符。
            <b>最厚的位置</b>两边都几乎不显著——挪动最厚处更像是把厚度重新分布，而不是整体加厚。
            <b>弯度</b>在四位族有一点可测的弱影响，在五位族几乎没有，原因见下面第二节。
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">二 改在机翼的哪一段最有效</div>
        <div className="panel-body">
          <p className="ex-note" style={{ marginTop: 0 }}>
            先要看清压力曲线是怎么排布的，这一条很关键。
            <b>它是绕机翼一圈画出来的闭合曲线</b>：序号 0 和序号 255 都在前缘，
            序号 127 在后缘。也就是说，<b>不能把它当成从前缘到后缘的一条直线来读</b>。
            早期工作正是把它当成直线读，才得出「四个量的影响都集中在后缘」这个错误结论。
            纠正后改用真实的弦向位置（机翼从前到后的方向）重新分三段：
            <b>前缘段</b>、<b>中段</b>、<b>后缘段</b>。
          </p>
          {data.figure52 && <StationEvidence d={data.figure52} />}
          <div className="seg" style={{ marginBottom: 8 }}>
            {Object.keys(glob.families).map((k) => (
              <button key={k} className={fam === k ? 'active' : ''} onClick={() => setFam(k)}>
                {FAM_LABEL[k]}
              </button>
            ))}
          </div>
          <div className="seg" style={{ marginBottom: 10 }}>
            {glob.components.map((c) => (
              <button key={c} className={comp === c ? 'active' : ''} onClick={() => setComp(c)}>
                {glob.labels[c]}
              </button>
            ))}
          </div>
          <LocalChart fam={localFam} comp={comp} segLabels={data.local.segLabels} />
          <p className="ex-note">
            曲线在哪个区间抬得高，就说明改动这个量主要在那一带起作用。
            底色从左到右分别是前缘段、中段、后缘段。
            当前选的是{' '}<b>{FAM_LABEL[fam]} · {glob.labels[comp]}</b>：
            真实数据判在{verdict.find((v) => v.comp === comp)?.dataDomLabel}，
            扰动模型判在{verdict.find((v) => v.comp === comp)?.modelDomLabel}，
            {verdict.find((v) => v.comp === comp)?.agree ? '两边一致。' : '两边不一致。'}
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">三 两套办法在哪里一致、在哪里分歧</div>
        <div className="panel-body">
          <p className="ex-note" style={{ marginTop: 0 }}>
            用改正后的位置坐标重新核对后，八个「族 × 几何量」组合里
            <b>只有 {agreeCount} 个两边都判在同一段</b>（四位族的前缘半径，两套办法都判在前缘段），
            其余 {8 - agreeCount} 个都不一致。这说明两族、四个量之间的作用区域本来就各不相同，
            「全都集中在后缘」那个旧结论不再成立。
          </p>
          <table className="ex-table">
            <thead>
              <tr>
                <th>族</th><th>几何量</th><th>真实数据判在哪段</th>
                <th>扰动模型判在哪段</th><th>是否一致</th>
              </tr>
            </thead>
            <tbody>
              {data.local.verdict.map((v) => (
                <tr key={v.family + v.comp}>
                  <td>{FAM_LABEL[v.family]}</td>
                  <td>{data.local.labels[v.comp] ?? v.comp}</td>
                  <td>{v.dataDomLabel}</td>
                  <td>{v.modelDomLabel}</td>
                  <td>{v.agree
                    ? <Verdict ok text="一致" />
                    : <Verdict ok={false} text="不一致" />}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="ex-note">
            <b>怎么理解这七个不一致。</b>
            扰动模型在挪动最厚位置、磨钝前缘时，压力变化都集中在前缘段——
            这符合几何事实，因为这些操作本身主要发生在前五分之一弦长。
            而真实数据反而多判后缘段与中段，说明真实样本里这些量的自然变化，
            与人为的干净扰动在空间分布上并不相同。
            这是真实存在的分歧，不是坐标没对齐造成的，本文如实报出。
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">四 改动一个量时，别的量会不会跟着变</div>
        <div className="panel-body">
          <p className="ex-note" style={{ marginTop: 0 }}>
            通俗地说，这里问的是<b>几个量之间会不会互相放大或抵消</b>。
            如果它们各管各的，那就可以一次只调一个；如果会互相作用，
            那就必须一起考虑。本文先立三个猜想，再用数据逐条判定。
          </p>
          <div className="ex-guess">
            <div className="ex-guess-row">
              <span className="ex-guess-tag bad">猜想 A 不成立</span>
              <span>
                「几何量之间的互相作用可以忽略」——<b>不成立</b>。
                四位族里，弯度与厚度的互相作用达到主效应的 38%，
                厚度与前缘半径达到 26%，都远高于 10% 的判定线。
                五位族里弯度与最厚位置、弯度与前缘半径也同样不可忽略。
              </span>
            </div>
            <div className="ex-guess-row">
              <span className="ex-guess-tag bad">猜想 B 不成立</span>
              <span>
                「几何与来流条件的互相作用不可忽略」——<b>不成立</b>。
                弯度与攻角的互相作用在两族都判可忽略。
                把弯度分成三档分别看升力随攻角的增长，三条线的斜率分别是
                0.1005、0.1014、0.0952，几乎平行，差异不到 6%，
                所以弯度并不会放大或削弱攻角的作用。
              </span>
            </div>
            <div className="ex-guess-row">
              <span className="ex-guess-tag ok">猜想 C 成立</span>
              <span>
                「最厚位置的互相作用其实是分组效应，不能当连续的作用来读」——<b>成立</b>。
                <b>原因很简单：最厚位置全库只有 5 个取值</b>，本来就不是连续变化的量，
                它与别的量之间看起来的「互相作用」，本质上是按离散档位分组带来的差别。
                这也是后面第五节反复强调「它不能当连续变量求导」的由来。
              </span>
            </div>
          </div>
          <p className="ex-note">
            下面的横条就是具体的数值依据：<b>条越长，说明这两个量之间的相互影响越大</b>。
            竖虚线是 10% 判定线，超过它就算不可忽略。
            横条按<b>受影响的对象</b>分两组：一组看升力，一组看前缘段的压力。
            同一对量在两组的比例可以差很多，因为分母是各自主效应里的最大者。
          </p>
          <div style={{ display: 'grid', gap: 'var(--sp-4)' }}>
            {['CL', 'LE_DEP'].map((resp) => (
              <div key={resp}>
                <div className="ex-sub">
                  受影响对象：<b>{RESP_LABEL[resp] ?? resp}</b>
                  {resp === 'LE_DEP' ? '（机翼前段表面的压力强弱）' : '（整条翼型的升力）'}
                </div>
                <div className="cols-2" style={{ alignItems: 'start' }}>
                  {['4digit', '5digit'].map((f) => {
                    const rows = grouped[f + '|' + resp] ?? []
                    if (!rows.length) return null
                    const maxPct = Math.max(...rows.map((r) => r.ratio * 100), 30)
                    return (
                      <div key={f}>
                        <div className="ex-figdata-cap" style={{ fontSize: 12.5, marginBottom: 2 }}>
                          {FAM_LABEL[f]}
                        </div>
                        <div className="ex-bars">
                          {rows.slice(0, 5).map((it) => {
                            const pct = it.ratio * 100
                            const over = pct >= 10
                            return (
                              <div key={f + resp + it.term} className="ex-bar-row">
                                <span className="ex-bar-label" title={it.termRaw ?? it.term}>
                                  {it.term}
                                </span>
                                <span className="ex-bar-track">
                                  <span
                                    className="ex-bar-fill"
                                    style={{
                                      width: `${Math.min(100, (pct / (maxPct * 1.15)) * 100)}%`,
                                      background: over ? 'var(--warm)' : 'var(--acc)',
                                    }}
                                  />
                                  <span className="ex-bar-threshold"
                                    style={{ left: `${(10 / (maxPct * 1.15)) * 100}%` }} />
                                </span>
                                <span className="num ex-bar-val"
                                  style={{ color: over ? 'var(--warm)' : 'var(--fg-2)' }}>
                                  {pct.toFixed(0)}%
                                </span>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
          <p className="ex-note">
            <b>为什么比例会超过 100%。</b>这里的百分比是
            <b>该项交互的系数 ÷ 同一模型里主效应系数绝对值的最大值</b>。
            也就是说，超过 100% 表示<b>这两个量配合起来产生的影响，
            比任何单独一个量的影响还要大</b>。这不是错误，而是一条有意义的发现：
            改厚度时如果同时改了攻角，效果会比单独改其中任何一个都强。
            （该项出现在「前缘段压力」那一组，因为厚度与攻角的配合主要作用在前缘。）
          </p>
          <p className="ex-note">
            <b>一条诚实说明。</b>本节关于互相作用的结论来自混了两族、按全局三等分做的初步证据，
            还没有做到「族内控制其他量之后」的严格版本。这个限定如实写在这里，
            不影响定性方向，但不应被当成精确数值引用。
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">五 从这些结论得到的设计建议</div>
        <div className="panel-body">
          <p className="ex-note" style={{ marginTop: 0 }}>
            把上面的结论翻成设计语言：<b>在数据库这一千个真实翼型里，
            升力高的那些有什么共同形状特征</b>。
            注意下面是统计出来的规律，不是唯一的答案——同一个升力水平可以有多种形状达到。
          </p>
          <div className="seg" style={{ marginBottom: 10 }}>
            {[['th_peak_half', '厚度'], ['yc_mid', '弯度'],
              ['re_radius_c', '前缘半径'], ['aoa', '攻角'], ['uinf', '来流速度']].map(([k, label]) => (
              <button key={k} className={xKey === k ? 'active' : ''} onClick={() => setXKey(k)}>
                {label}
              </button>
            ))}
          </div>
          <ScatterChart
            part={data.scatter}
            keyName={xKey}
            label={
              [['th_peak_half', '厚度'], ['yc_mid', '弯度'],
                ['re_radius_c', '前缘半径'], ['aoa', '攻角 度'],
                ['uinf', '来流速度 米每秒']].find(([k]) => k === xKey)?.[1] ?? ''
            }
          />
          <p className="ex-note">
            横轴是选中的几何量，纵轴是官方给出的真实升力，每个点是一条真实样本。
            切换横轴，就能看出升力随哪个量变化最明显。
            工况带内共 {data.scatter.band?.n ?? 0} 个真实样本，
            官方升力从 {data.scatter.band?.lo ?? 0} 到 {data.scatter.band?.hi ?? 0}。
            目标若落在这个区间之外，后续的反向设计会直接判为不可达，不去硬凑。
          </p>
          <div className="cols-2" style={{ alignItems: 'start' }}>
            <div>
              <div className="ex-sub">升力前十名的形状，相对全体样本中位数偏了多少</div>
              <table className="ex-table">
                <thead>
                  <tr><th>几何量</th><th>全体中位数</th><th>前十名均值</th><th>偏差</th></tr>
                </thead>
                <tbody>
                  {data.design.dev.map((d) => (
                    <tr key={d.comp}>
                      <td>{d.label}</td>
                      <td className="num">{d.poolMedian.toFixed(5)}</td>
                      <td className="num">{d.top10Mean.toFixed(5)}</td>
                      <td className="num" style={{ color: d.devMean >= 0 ? 'var(--ok, #34d399)' : 'var(--bad, #f87171)' }}>
                        {d.devMean >= 0 ? '+' : ''}{d.devMean.toFixed(5)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="ex-note">
                共同方向是<b>更薄、弯度中等偏大、最厚处偏前、前缘偏尖</b>。
                <b>但前十名的散布很宽</b>，说明这不是唯一的最优形状，
                而是若干条可选的调形方向——这一点对反向设计很重要，它意味着解不唯一。
              </p>
            </div>
            <div>
              <div className="ex-sub">官方升力排名前十的真实样本</div>
              <table className="ex-table">
                <thead>
                  <tr><th>名次</th><th>族</th><th>攻角</th><th>升力</th><th>局部马赫</th></tr>
                </thead>
                <tbody>
                  {data.design.rows.map((r) => (
                    <tr key={r.rank}>
                      <td className="num">{r.rank}</td>
                      <td>{r.family === '4digit' ? '四位' : '五位'}</td>
                      <td className="num">{r.aoa.toFixed(2)}</td>
                      <td className="num">{r.cl.toFixed(4)}</td>
                      <td className="num">{r.mLocal.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="ex-note">
                这个排名用的是官方算出的真实升力，不是模型预测值。
                前十名里四位族占七席，局部马赫都小于 0.6，约束还有余量。
              </p>
            </div>
          </div>
          <div className="ex-guess">
            <div className="ex-guess-row">
              <span className="ex-guess-tag ok">可以同时加大</span>
              <span>
                <b>前缘半径与弯度</b>都提升升力，方向相同，可以一起加大。
              </span>
            </div>
            <div className="ex-guess-row">
              <span className="ex-guess-tag bad">彼此有冲突</span>
              <span>
                <b>厚度</b>这一项内部方向相反：厚一些能让后缘分离更晚（有利于大攻角不失速），
                但升力会降低。两者不能同时要，必须明确取舍。
              </span>
            </div>
            <div className="ex-guess-row">
              <span className="ex-guess-tag bad">不能当连续变量用</span>
              <span>
                有三类量不能当作连续可调：<b>最厚的位置</b>（全库只有 5 个取值，是离散量）、
                四位族的模型几何与攻角映射（前面发现它不可信）、
                以及区分度不足的分离判据。这三条限制也直接写进了反向设计的约束条件。
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">六 这些结论的适用范围</div>
        <div className="panel-body" style={{ display: 'grid', gap: 8, fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ margin: 0 }}>
            <b>只适用于二维、不旋转、无展向流动的孤立翼型。</b>
            数据集本身就是这个假设下算出来的数值解，不含三维旋转效应、叶尖损失与动态失速，
            不能直接外推到真实的旋转叶片。
          </p>
          <p style={{ margin: 0 }}>
            <b>只谈升力，不谈阻力。</b>摩擦阻力占实际阻力的一半以上，
            而现有方法只对压力建模，算出的阻力会系统性偏低，报出去是误导。
          </p>
          <p style={{ margin: 0 }}>
            <b>两个指标是近似值。</b>页面里的「局部马赫数」是不可压近似下的估算，
            「分离」判据也只是一个代理量，两者都不是直接求解出来的物理量，只宜作相对比较。
          </p>
          <p style={{ margin: 0 }}>
            <b>最可靠的一条结论是厚度。</b>厚度在真实数据与扰动模型两套办法里都是第一影响量，
            且方向明确（越厚升力越低），可以在设计里放心使用。
            其余结论凡是有「两边不一致」标记的，都应先回到真实样本复核再使用。
          </p>
        </div>
      </div>

    </section>
  )
}

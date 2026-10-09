// 降维原理流程图：高维快照 → 8 条 POD 模态 → 重构对比。
//
// 不用任何 AI 生成图。所有画面由真实数据现场矢量绘制：
//   · 左边：抽中的一条真实样本（samples.json #280），它的压力曲线（256 点）
//     与模型预测的流场（速度 u / 压力 p，b_field.onnx 离线算好的 256×128 网格）；
//   · 中间：cp_basis.json 里真实的 8 条 POD 模态曲线 φ1..φ8，
//     以及这条样本对应的 8 个模态坐标 α1..α8；
//   · 右边：同一条样本的「真值 vs 重构」压力曲线（重构 = 均值 + Σ α_k φ_k），
//     并标注重构相对误差。
//
// 数据文件：mor_diagram.json（元信息与小数组）+ mor_diagram.bin（u/p 两个场）。
// 配色与全站 tokens.css 一致（深色藏蓝底、青 #22d3ee、橙 #fb923c、绿 #34d399）。

import { useEffect, useRef, useState } from 'react'
import { colors } from '../lib/plot.ts'

interface DiagramData {
  name: string
  L: number
  P: number
  Q: number
  T: number
  aoa: number
  uInf: number
  relL2: number
  nx: number
  ny: number
  fieldOrder: string[]
  s: number[]
  truth: number[]
  rec: number[]
  alpha8: number[]
  mu: number[]
  phi: number[][]
  outlineX: number[]
  outlineY: number[]
  x: number[]
  y: number[]
}

interface Loaded {
  meta: DiagramData
  fields: Record<string, Float32Array>
}

const ACC = '#22d3ee'
const ACC_STRONG = '#67e8f9'
const OK = '#34d399'

let cache: Promise<Loaded | null> | null = null
async function loadDiagram(base: string): Promise<Loaded | null> {
  if (!cache) {
    cache = (async () => {
      try {
        const [j, b] = await Promise.all([
          fetch(base + 'data/mor_diagram.json', { cache: 'no-cache' }),
          fetch(base + 'data/mor_diagram.bin', { cache: 'no-cache' }),
        ])
        if (!j.ok || !b.ok) return null
        const meta = (await j.json()) as DiagramData
        const raw = new Float32Array(await b.arrayBuffer())
        const n = meta.nx * meta.ny
        if (raw.length !== meta.fieldOrder.length * n) return null
        const fields: Record<string, Float32Array> = {}
        meta.fieldOrder.forEach((f, k) => { fields[f] = raw.subarray(k * n, (k + 1) * n) })
        return { meta, fields }
      } catch {
        return null
      }
    })()
  }
  return cache
}

function setupCanvas(el: HTMLCanvasElement, height: number) {
  const parent = el.parentElement
  if (!parent) return null
  const w = parent.clientWidth
  if (!w) return null
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  el.width = Math.round(w * dpr)
  el.height = Math.round(height * dpr)
  el.style.width = w + 'px'
  el.style.height = height + 'px'
  const ctx = el.getContext('2d')
  if (!ctx) return null
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { ctx, w, h: height }
}

/** 与 FlowField 同款的 jet 配色。 */
function colorMap(t: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0.0, [22, 10, 60]],
    [0.25, [24, 60, 140]],
    [0.5, [16, 130, 160]],
    [0.7, [34, 211, 238]],
    [0.88, [250, 200, 100]],
    [1.0, [251, 146, 60]],
  ]
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i]
    const [t1, c1] = stops[i + 1]
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0)
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
      ]
    }
  }
  return stops[stops.length - 1][1]
}

function pct(sorted: number[], p: number): number {
  if (!sorted.length) return 0
  const i = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))
  return sorted[i]
}

/** 8 条模态从亮青渐变到青。 */
function modeColor(i: number): string {
  const t = i / 7
  return `rgb(${Math.round(103 - 69 * t)},${Math.round(232 - 21 * t)},${Math.round(249 - 11 * t)})`
}

/** 流场热图 + 翼型轮廓叠画。 */
function drawField(
  el: HTMLCanvasElement, values: Float32Array, meta: DiagramData, title: string,
) {
  const box = setupCanvas(el, 122)
  if (!box) return
  const { ctx, w, h } = box
  const { nx, ny } = meta
  const valid: number[] = []
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) valid.push(values[i])
  valid.sort((a, b) => a - b)
  const lo = pct(valid, 0.02)
  const hi = pct(valid, 0.98) || lo + 1e-6

  const off = document.createElement('canvas')
  off.width = nx; off.height = ny
  const octx = off.getContext('2d')!
  const img = octx.createImageData(nx, ny)
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = values[j * nx + i]
      const o = (j * nx + i) * 4
      if (!Number.isFinite(v)) { img.data[o + 3] = 0; continue }
      const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)))
      const [r, g, b] = colorMap(t)
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255
    }
  }
  octx.putImageData(img, 0, 0)

  const xr = [meta.x[0], meta.x[meta.x.length - 1]]
  const yr = [meta.y[0], meta.y[meta.y.length - 1]]
  const aspect = (xr[1] - xr[0]) / (yr[1] - yr[0])
  let bw = w, bh = bw / aspect
  if (bh > h) { bh = h; bw = bh * aspect }
  const ox = (w - bw) / 2, oy = (h - bh) / 2
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(off, ox, oy, bw, bh)

  const sx = (xx: number) => ox + ((xx - xr[0]) / (xr[1] - xr[0])) * bw
  const sy = (yy: number) => oy + bh - ((yy - yr[0]) / (yr[1] - yr[0])) * bh
  ctx.beginPath()
  meta.outlineX.forEach((xx, k) => {
    const px = sx(xx), py = sy(meta.outlineY[k])
    if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
  })
  ctx.closePath()
  ctx.lineWidth = 1.4
  ctx.strokeStyle = 'rgba(232,238,252,0.95)'
  ctx.fillStyle = 'rgba(5,7,13,0.5)'
  ctx.fill(); ctx.stroke()

  ctx.fillStyle = 'rgba(174,186,219,0.92)'
  ctx.font = '11px "JetBrains Mono", monospace'
  ctx.fillText(title, 8, 14)
}

/** 单条 Cp 曲线（负值在上，气动惯例）。 */
function drawCpSingle(el: HTMLCanvasElement, cp: number[], height: number) {
  const box = setupCanvas(el, height)
  if (!box) return
  const { ctx, w, h } = box
  const n = cp.length
  const pad = 6
  const ymin = Math.min(...cp), ymax = Math.max(...cp)
  const span = (ymax - ymin) || 1
  const sx = (i: number) => pad + (i / (n - 1)) * (w - 2 * pad)
  const sy = (v: number) => pad + ((ymax - v) / span) * (h - 2 * pad)
  ctx.beginPath()
  cp.forEach((v, i) => { if (i === 0) ctx.moveTo(sx(i), sy(v)); else ctx.lineTo(sx(i), sy(v)) })
  ctx.lineWidth = 1.8
  ctx.strokeStyle = ACC
  ctx.shadowColor = ACC; ctx.shadowBlur = 5
  ctx.stroke(); ctx.shadowBlur = 0
}

/** 迷你模态曲线。 */
function drawSpark(el: HTMLCanvasElement, ys: number[], stroke: string) {
  const box = setupCanvas(el, 46)
  if (!box) return
  const { ctx, w, h } = box
  const n = ys.length
  const pad = 4
  const ymin = Math.min(...ys), ymax = Math.max(...ys)
  const span = (ymax - ymin) || 1
  const sx = (i: number) => pad + (i / (n - 1)) * (w - 2 * pad)
  const sy = (v: number) => h - pad - ((v - ymin) / span) * (h - 2 * pad)
  ctx.beginPath()
  ys.forEach((v, i) => { if (i === 0) ctx.moveTo(sx(i), sy(v)); else ctx.lineTo(sx(i), sy(v)) })
  ctx.lineWidth = 1.4
  ctx.strokeStyle = stroke
  ctx.stroke()
}

/** 真值 vs 重构对比。 */
function drawCompare(el: HTMLCanvasElement, m: DiagramData) {
  const box = setupCanvas(el, 200)
  if (!box) return
  const { ctx, w, h } = box
  const C = colors()
  const n = m.truth.length
  const pad = { l: 34, r: 8, t: 8, b: 22 }
  const all = [...m.truth, ...m.rec]
  const ymin = Math.min(...all), ymax = Math.max(...all)
  const span = (ymax - ymin) || 1
  const sx = (i: number) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r)
  const sy = (v: number) => pad.t + ((ymax - v) / span) * (h - pad.t - pad.b)

  ctx.strokeStyle = C.grid; ctx.lineWidth = 1
  for (let g = 0; g <= 4; g++) {
    const yy = pad.t + (g / 4) * (h - pad.t - pad.b)
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke()
  }
  // 真值（淡青）
  ctx.beginPath()
  m.truth.forEach((v, i) => { if (i === 0) ctx.moveTo(sx(i), sy(v)); else ctx.lineTo(sx(i), sy(v)) })
  ctx.lineWidth = 1.6; ctx.strokeStyle = 'rgba(174,186,219,0.85)'; ctx.stroke()
  // 重构（绿，几乎重合）
  ctx.beginPath()
  m.rec.forEach((v, i) => { if (i === 0) ctx.moveTo(sx(i), sy(v)); else ctx.lineTo(sx(i), sy(v)) })
  ctx.lineWidth = 2; ctx.strokeStyle = OK
  ctx.shadowColor = OK; ctx.shadowBlur = 5
  ctx.stroke(); ctx.shadowBlur = 0

  ctx.font = '11px "JetBrains Mono", monospace'
  ctx.fillStyle = 'rgba(174,186,219,0.9)'
  ctx.fillText('— 真值（256 点）', pad.l, h - 6)
  ctx.fillStyle = OK
  ctx.fillText('— 重构（8 个坐标）', pad.l + 130, h - 6)
}

interface Props { base: string }

export default function MorFlowDiagram({ base }: Props) {
  const [data, setData] = useState<Loaded | null>(null)
  const [err, setErr] = useState('')
  const uRef = useRef<HTMLCanvasElement>(null)
  const pRef = useRef<HTMLCanvasElement>(null)
  const cpRef = useRef<HTMLCanvasElement>(null)
  const cmpRef = useRef<HTMLCanvasElement>(null)
  const modeRefs = useRef<Array<HTMLCanvasElement | null>>([])

  useEffect(() => {
    let live = true
    loadDiagram(base).then((d) => {
      if (!live) return
      if (d) setData(d); else setErr('mor_diagram 数据缺失')
    })
    return () => { live = false }
  }, [base])

  const m = data?.meta ?? null

  useEffect(() => {
    if (!data || !m) return
    const draw = () => {
      if (uRef.current) drawField(uRef.current, data.fields.u, m, '速度场 u (m/s)')
      if (pRef.current) drawField(pRef.current, data.fields.p, m, '压力场 p (Pa)')
      if (cpRef.current) drawCpSingle(cpRef.current, m.truth, 92)
      if (cmpRef.current) drawCompare(cmpRef.current, m)
      for (let k = 0; k < 8; k++) {
        const el = modeRefs.current[k]
        if (el) drawSpark(el, m.phi.map((row) => row[k]), modeColor(k))
      }
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (uRef.current?.parentElement) ro.observe(uRef.current.parentElement)
    return () => ro.disconnect()
  }, [data, m])

  const C = colors()
  if (err) return <div className="panel-body muted">降维原理图数据没加载到：{err}</div>
  if (!m || !data) return <div className="panel-body muted">正在加载降维原理图…</div>

  return (
    <>
      {/* 先把“MOR 到底在干什么”用一句话讲清，再看图。 */}
      <div className="mor-intro">
        <div className="mor-intro-q">
          <span className="mor-intro-tag">问题</span>
          一条 256 点的压力曲线 = <b>256 个数字</b>。一千条样本就是 25.6 万个数字，
          存、算、传都很贵。但它们其实长得非常像——能不能用<span style={{ color: 'var(--warm)' }}>少数几个数字</span>就把它描述清楚？
        </div>
        <div className="mor-intro-a">
          <span className="mor-intro-tag">MOR 的做法</span>
          先在一千条曲线里找出一组<b>公共的“基形状”</b>（POD 模态），再让每条曲线
          只记录<b>每个基形状的份量</b>（坐标）。基形状全样本共用，只存一份；
          每条样本只存那几个份量。
        </div>
        <div className="mor-intro-num">
          <span><b>256</b> 个数字</span>
          <span className="mor-intro-arrow">→</span>
          <span><b>8</b> 个坐标 + 8 条共用基</span>
          <span className="mor-intro-arrow">=</span>
          <span className="mor-intro-win">样本自身只剩 8 个数，仍能还原到 96% 以上</span>
        </div>
      </div>

      <div className="mor-flow">
      {/* ============ 面板 1：高维快照 ============ */}
      <div className="mor-panel" style={{ padding: '12px 14px' }}>
        <div className="mor-panel-title" style={{ color: ACC }}>
          <span className="mor-num">1</span> 高维快照 · 真实样本
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          {m.name.replace('airFoil2D_SST_', '')}（{m.aoa.toFixed(1)}°，{m.uInf.toFixed(0)} m/s）
        </div>
        <canvas ref={uRef} />
        <canvas ref={pRef} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span className="muted" style={{ fontSize: 12 }}>压力曲线 Cp(s)</span>
          <span className="num" style={{ fontSize: 11, color: C.fg2 }}>256 点</span>
        </div>
        <canvas ref={cpRef} />
      </div>

      {/* ============ 面板 2：8 条模态 ============ */}
      <div className="mor-flow-mid">
        <div className="mor-arrow"><span>POD 分解</span></div>
        <div className="mor-panel" style={{ padding: '12px 14px', alignContent: 'start' }}>
          <div className="mor-panel-title" style={{ color: ACC_STRONG }}>
            <span className="mor-num">2</span> 8 条公共模态
          </div>
          <div className="muted" style={{ fontSize: 12 }}>
            一千条曲线的共性拆成 8 条基曲线。每条基都是一条完整的 256 点曲线，
            <b>全样本共用</b>，只存一份。
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            {Array.from({ length: 8 }, (_, k) => (
              <div key={k} style={{ display: 'grid', gap: 1 }}>
                <span className="num" style={{ fontSize: 11, color: modeColor(k) }}>φ{k + 1}</span>
                <canvas ref={(el) => { modeRefs.current[k] = el }} />
              </div>
            ))}
          </div>
          <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
            这条样本自己的 8 个坐标（每个数字 = 对应基曲线要加多少）
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 3 }}>
            {m.alpha8.map((a, k) => (
              <div key={k} className="mor-chip" title={`α${k + 1} = ${a.toFixed(3)}`}>
                <span className="num" style={{ fontSize: 10, color: modeColor(k) }}>α{k + 1}</span>
                <span className="num" style={{ fontSize: 11 }}>{a.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="mor-formula">
            任意一条曲线 = <span className="mor-mean">平均曲线</span>
            <span className="mor-plus">+</span>
            α₁φ₁ <span className="mor-plus">+</span> α₂φ₂ <span className="mor-plus">+</span> … <span className="mor-plus">+</span> α₈φ₈
          </div>
        </div>
        <div className="mor-arrow"><span>加权加和</span></div>
      </div>

      {/* ============ 面板 3：重构对比 ============ */}
      <div className="mor-panel" style={{ padding: '12px 14px', alignContent: 'start' }}>
        <div className="mor-panel-title" style={{ color: OK }}>
          <span className="mor-num">3</span> 重构对比
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          均值 + 8 个坐标 × 8 条模态，能不能还原这条曲线
        </div>
        <canvas ref={cmpRef} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span className="muted" style={{ fontSize: 12 }}>重构相对误差</span>
          <span className="num" style={{ fontSize: 22, color: OK }}>{(m.relL2 * 100).toFixed(1)}%</span>
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          这条 256 点的曲线，8 个数字就还原到 {((1 - m.relL2) * 100).toFixed(1)}%
        </div>
      </div>
      </div>

      {/* 把“为什么能这样”讲透：降维的本质是找共性。 */}
      <div className="mor-why">
        <div className="mor-why-col">
          <div className="mor-why-h" style={{ color: ACC }}>为什么 8 个数能代替 256 个数</div>
          因为一千条压力曲线的<b>形状高度相关</b>：翼型绕流的压力分布基本由
          前缘吸力峰与后缘恢复决定，形状都差不多，只是幅度与位置略有差别。
          POD（本征正交分解）就是把这堆“差不多的曲线”里最显著的几个
          <b>变化方向</b>提取出来当基，剩下的部分小到可以丢掉。
        </div>
        <div className="mor-why-col">
          <div className="mor-why-h" style={{ color: ACC_STRONG }}>这里到底降了多少</div>
          一条样本的原始表面文件约 0.06 MB（约一千个轮廓点）；
          降维后每条样本只需 <b>8 个 float32 = 32 字节</b> 的坐标，
          加上全样本共用的基（8×256 个 float32，约 8 KB，不按样本重复计）。
          这正是“小模型 + 降维”能在浏览器里毫秒出结果的前提。
        </div>
        <div className="mor-why-col">
          <div className="mor-why-h" style={{ color: OK }}>降维之后拿它干什么</div>
          模型不再需要预测 256 个点，只需预测这 <b>8 个数</b>，
          输出维度降下来后小模型也能准。
          反过来，反向设计需要搜形状时，也是在<b>低维坐标</b>上搜，比直接搜几何快得多。
        </div>
      </div>
    </>
  )
}

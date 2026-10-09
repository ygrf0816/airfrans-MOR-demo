// 轻量 canvas 绘图助手。四个页面共用，不引入大型图表库，保证构建产物小。

export interface Pt {
  x: number
  y: number
}

export interface PlotBox {
  el: HTMLCanvasElement
  ctx: CanvasRenderingContext2D
  w: number
  h: number
}

/** 按设备像素比准备画布，返回可用上下文与逻辑宽高。 */
export function setupCanvas(el: HTMLCanvasElement, height: number): PlotBox | null {
  const parent = el.parentElement
  if (!parent) return null
  const w = parent.clientWidth
  if (w === 0) return null
  const dpr = window.devicePixelRatio || 1
  el.width = Math.round(w * dpr)
  el.height = Math.round(height * dpr)
  el.style.width = w + 'px'
  el.style.height = height + 'px'
  const ctx = el.getContext('2d')
  if (!ctx) return null
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, height)
  return { el, ctx, w, h: height }
}

const COLOR = {
  grid: 'rgba(148, 163, 184, 0.12)',
  axis: '#3b4a6b',
  fg: '#aebadb',
  fg2: '#6b7aa3',
  acc: '#22d3ee',
  accSoft: 'rgba(34, 211, 238, 0.16)',
  warm: '#fb923c',
  ok: '#34d399',
  bad: '#f87171',
  white: '#e8eefc',
}

export function colors() {
  return COLOR
}

/** 画折线。 */
export function strokeLine(
  ctx: CanvasRenderingContext2D,
  pts: Pt[],
  color: string,
  width = 1.5,
): void {
  if (pts.length < 2) return
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.beginPath()
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
  ctx.stroke()
}

/** 画折线并向下填充渐变。 */
export function fillLine(
  ctx: CanvasRenderingContext2D,
  pts: Pt[],
  baseY: number,
  colorTop: string,
  colorBottom: string,
  height: number,
): void {
  if (pts.length < 2) return
  const grad = ctx.createLinearGradient(0, 0, 0, height)
  grad.addColorStop(0, colorTop)
  grad.addColorStop(1, colorBottom)
  ctx.fillStyle = grad
  ctx.beginPath()
  ctx.moveTo(pts[0].x, baseY)
  for (const p of pts) ctx.lineTo(p.x, p.y)
  ctx.lineTo(pts[pts.length - 1].x, baseY)
  ctx.closePath()
  ctx.fill()
}

export interface Scale {
  x: (v: number) => number
  y: (v: number) => number
}

/** 线性比例尺工厂。 */
export function makeScale(
  dom: [number, number],
  range: [number, number],
): (v: number) => number {
  const [d0, d1] = dom
  const [r0, r1] = range
  const span = d1 - d0 || 1
  return (v: number) => r0 + ((v - d0) / span) * (r1 - r0)
}

/** 画坐标网格与刻度文字。 */
export function drawGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  pad: { l: number; r: number; t: number; b: number },
  xTicks: { v: number; label: string }[],
  yTicks: { v: number; label: string }[],
  sx: (v: number) => number,
  sy: (v: number) => number,
): void {
  ctx.font = '10px "JetBrains Mono", Consolas, monospace'
  ctx.strokeStyle = COLOR.grid
  ctx.fillStyle = COLOR.fg2
  ctx.lineWidth = 1
  for (const t of xTicks) {
    const px = sx(t.v)
    ctx.beginPath()
    ctx.moveTo(px, pad.t)
    ctx.lineTo(px, h - pad.b)
    ctx.stroke()
    ctx.textAlign = 'center'
    ctx.fillText(t.label, px, h - pad.b + 13)
  }
  for (const t of yTicks) {
    const py = sy(t.v)
    ctx.beginPath()
    ctx.moveTo(pad.l, py)
    ctx.lineTo(w - pad.r, py)
    ctx.stroke()
    ctx.textAlign = 'right'
    ctx.fillText(t.label, pad.l - 6, py + 3)
  }
  ctx.strokeStyle = COLOR.axis
  ctx.strokeRect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b)
}

/** 好read的刻度生成。 */
export function niceTicks(
  d0: number,
  d1: number,
  n: number,
  fmt: (v: number) => string,
): { v: number; label: string }[] {
  const out: { v: number; label: string }[] = []
  if (d1 <= d0) return out
  const step = (d1 - d0) / Math.max(1, n - 1)
  const mag = 10 ** Math.floor(Math.log10(step))
  const norm = step / mag
  const nice = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1
  const s = (nice * mag) / 2
  const start = Math.ceil(d0 / s) * s
  for (let v = start; v <= d1 + 1e-12; v += s) {
    out.push({ v, label: fmt(v) })
  }
  return out
}

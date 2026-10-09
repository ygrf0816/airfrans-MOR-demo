import { useEffect, useRef } from 'react'
import type { Grid } from '../data/load.ts'
import { colors } from '../lib/plot.ts'

interface Props {
  grid: Grid
  /**
   * 场值，必须是**整网格**数组，长度 nx*ny，下标 j*nx+i，掩膜外填 NaN。
   * 不要把"掩膜序"的紧凑数组传进来，那样整张图会被错位铺开。
   */
  field: Float32Array
  unit: string
  /** 翼型轮廓（弦长坐标），描白边用 */
  outline?: [number, number][]
  height?: number
  /**
   * 显示窗口 [x0, x1, y0, y1]。缺省裁到翼型附近：
   * 弦向 -0.5 到 1.8，法向 -0.9 到 0.9，再按画布长宽比等比放宽一边。
   * 坐标映射不变，只裁显示范围。
   */
  window?: [number, number, number, number]
}

/** 发散色标：深紫到青到暖黄。 */
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

/** 数组的 p 分位数（p 0..1），线性插值。 */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = p * (sorted.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(sorted.length - 1, lo + 1)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

/**
 * 流场云图。坐标映射与数据源不动，只裁显示窗口并把配色范围
 * 改成窗口内有效值的 2% 与 98% 分位，近翼型结构就不会被压成一片。
 * 掩膜外留黑，翼型描亮白边，色标单独一列、标签在右侧。
 */
export default function FlowField({
  grid,
  field,
  unit,
  outline,
  height = 300,
  window: win,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const parent = el.parentElement
      if (!parent) return
      const w = parent.clientWidth
      if (w === 0) return
      const dpr = window.devicePixelRatio || 1
      el.width = Math.round(w * dpr)
      el.height = Math.round(height * dpr)
      el.style.width = w + 'px'
      el.style.height = height + 'px'
      const ctx = el.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const C = colors()

      const barW = 12
      const barLabelW = 64
      // 底部留 64：单位、来流方向箭头、窗口范围三行错开摆放。
      const pad = { l: 8, r: barW + barLabelW + 12, t: 8, b: 64 }
      const iw = w - pad.l - pad.r
      const ih = height - pad.t - pad.b
      const canvasAspect = iw / ih

      // 显示窗口。数据集掩膜是一千条样本计算域的并集，法向超过约 1.5 之后
      // 最外侧几行没有有效格（±1.65 时最小行有效率是 0），直接显示就是黑边。
      // 所以这里把法向**硬夹**在 ±1.5 以内，长宽比靠横向伸缩去配；
      // 横向也配不下时再压缩法向。总之既不变形，也不露黑边。
      const YLIM = 1.5
      const req = win ?? [grid.x0, grid.x1, -YLIM, YLIM]
      let wy0 = Math.max(req[2], -YLIM)
      let wy1 = Math.min(req[3], YLIM)
      if (wy1 - wy0 < 0.2) { wy0 = -YLIM; wy1 = YLIM }
      const winCy = (wy0 + wy1) / 2
      let bh = wy1 - wy0
      let bw = bh * canvasAspect
      const domainW = grid.x1 - grid.x0
      if (bw > domainW) {
        bw = domainW
        bh = bw / canvasAspect
      }
      if (bh > 2 * YLIM) {
        bh = 2 * YLIM
        bw = bh * canvasAspect
      }
      const winCx = (req[0] + req[1]) / 2
      let wx0 = winCx - bw / 2
      let wx1 = winCx + bw / 2
      if (wx0 < grid.x0) { wx0 = grid.x0; wx1 = wx0 + bw }
      if (wx1 > grid.x1) { wx1 = grid.x1; wx0 = wx1 - bw }
      wy0 = winCy - bh / 2
      wy1 = winCy + bh / 2

      const { nx, ny, mask, x, y } = grid
      // 形状自检：必须是整网格数组。传成掩膜序数组时会静默画错，
      // 表现是背景被放大且与翼型对不上，所以这里直接拦下来。
      if (field.length !== nx * ny) {
        throw new Error('流场数组长度 ' + field.length + ' 与网格 ' + (nx * ny)
          + ' 不符，需要整网格数组而不是掩膜序数组')
      }
      // 窗口内有效格的下标与值
      const winIdx: number[] = []
      const winVals: number[] = []
      for (let j = 0; j < ny; j++) {
        if (y[j] < wy0 || y[j] > wy1) continue
        for (let i = 0; i < nx; i++) {
          if (x[i] < wx0 || x[i] > wx1) continue
          const idx = j * nx + i
          if (!mask[idx]) continue
          const v = field[idx]
          if (Number.isNaN(v)) continue
          winIdx.push(idx)
          winVals.push(v)
        }
      }
      // 配色范围：窗口内 2% 与 98% 分位
      let vMin = 0
      let vMax = 1
      if (winVals.length > 0) {
        winVals.sort((a, b) => a - b)
        vMin = percentile(winVals, 0.02)
        vMax = percentile(winVals, 0.98)
        if (vMax - vMin < 1e-12) {
          vMin -= 0.5
          vMax += 0.5
        }
      }

      // 离屏小画布渲染整网格（窗口外的像素直接透明），再按窗口范围放大
      const off = document.createElement('canvas')
      off.width = nx
      off.height = ny
      const octx = off.getContext('2d')!
      const img = octx.createImageData(nx, ny)
      const span = vMax - vMin
      for (const idx of winIdx) {
        const t = Math.max(0, Math.min(1, (field[idx] - vMin) / span))
        const [r, g, b] = colorMap(t)
        const o = idx * 4
        img.data[o] = r
        img.data[o + 1] = g
        img.data[o + 2] = b
        img.data[o + 3] = 255
      }
      octx.putImageData(img, 0, 0)

      ctx.fillStyle = '#05070d'
      ctx.fillRect(0, 0, w, height)
      // 源矩形：窗口对应的像素范围（注意 y 行序是从 y0 到 y1）
      const gx0 = grid.x0
      const gx1 = grid.x1
      const gy0 = grid.y0
      const gy1 = grid.y1
      const srcX = ((wx0 - gx0) / (gx1 - gx0)) * nx
      const srcW = ((wx1 - wx0) / (gx1 - gx0)) * nx
      // 行 0 对应 y0（底部），所以源矩形的顶行对应 wy1
      const srcY = ((gy1 - wy1) / (gy1 - gy0)) * ny
      const srcH = ((wy1 - wy0) / (gy1 - gy0)) * ny
      ctx.imageSmoothingEnabled = true
      ctx.save()
      ctx.translate(pad.l, pad.t)
      ctx.scale(1, -1)
      ctx.translate(0, -ih)
      ctx.drawImage(off, srcX, srcY, srcW, srcH, 0, 0, iw, ih)
      ctx.restore()

      // 翼型亮白边（加粗到 1.8，颜色更亮）
      if (outline && outline.length > 2) {
        const sx = (v: number) => pad.l + ((v - wx0) / (wx1 - wx0)) * iw
        const sy = (v: number) => pad.t + ih - ((v - wy0) / (wy1 - wy0)) * ih
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)'
        ctx.lineWidth = 1.8
        ctx.beginPath()
        let started = false
        for (const [px, py] of outline) {
          if (px < wx0 - 0.05 || px > wx1 + 0.05 || py < wy0 - 0.05 || py > wy1 + 0.05) {
            started = false
            continue
          }
          if (!started) {
            ctx.moveTo(sx(px), sy(py))
            started = true
          } else {
            ctx.lineTo(sx(px), sy(py))
          }
        }
        ctx.closePath()
        ctx.stroke()
      }

      // 色标单独一列，标签在色标右侧、左对齐，上下分别对齐顶底
      const barX = w - pad.r + 12
      const grad = ctx.createLinearGradient(0, pad.t + ih, 0, pad.t)
      for (let k = 0; k <= 10; k++) {
        const [r, g, b] = colorMap(k / 10)
        grad.addColorStop(k / 10, `rgb(${r},${g},${b})`)
      }
      ctx.fillStyle = grad
      ctx.fillRect(barX, pad.t, barW, ih)
      ctx.strokeStyle = C.axis
      ctx.lineWidth = 1
      ctx.strokeRect(barX, pad.t, barW, ih)
      const labX = barX + barW + 6
      ctx.fillStyle = C.fg
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      ctx.fillText(vMax.toExponential(2), labX, pad.t)
      ctx.textBaseline = 'bottom'
      ctx.fillText(vMin.toExponential(2), labX, pad.t + ih)
      ctx.textBaseline = 'alphabetic'
      // 单位横排在色标下方，避免竖排被裁
      ctx.fillStyle = C.fg2
      ctx.font = '10px "PingFang SC", "Microsoft YaHei", sans-serif'
      ctx.textAlign = 'left'
      ctx.fillText(unit, barX - 2, pad.t + ih + 14)

      // 来流方向指示。数据集的自由来流沿 +x，屏幕右方就是 +x，
      // 所以箭头一律向右，和云图里尾迹被吹向下游的方向一致。
      const arrowY = pad.t + ih + 34
      const ax0 = pad.l + 4
      const ax1 = pad.l + iw - 4
      const flowColor = 'rgba(34, 211, 238, 0.9)'
      ctx.strokeStyle = flowColor
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(ax0, arrowY)
      ctx.lineTo(ax1, arrowY)
      ctx.stroke()
      const head = 9
      ctx.fillStyle = flowColor
      ctx.beginPath()
      ctx.moveTo(ax1, arrowY)
      ctx.lineTo(ax1 - head, arrowY - head * 0.55)
      ctx.lineTo(ax1 - head, arrowY + head * 0.55)
      ctx.closePath()
      ctx.fill()
      // 文字压在箭头上，两侧用底色挖空，箭头看起来是断开的，字更清楚
      const flowLabel = '来流方向'
      ctx.font = '11px "PingFang SC", "Microsoft YaHei", sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      const labelW = ctx.measureText(flowLabel).width
      const labelCx = (ax0 + ax1) / 2
      ctx.fillStyle = '#05070d'
      ctx.fillRect(labelCx - labelW / 2 - 6, arrowY - 8, labelW + 12, 16)
      ctx.fillStyle = 'rgba(34, 211, 238, 0.95)'
      ctx.fillText(flowLabel, labelCx, arrowY)
      ctx.textBaseline = 'alphabetic'

      // 底部窗口范围提示
      ctx.fillStyle = C.fg2
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText(
        `x ${wx0.toFixed(2)} ~ ${wx1.toFixed(2)} · y ${wy0.toFixed(2)} ~ ${wy1.toFixed(2)}`,
        pad.l + iw / 2,
        height - 6,
      )
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [grid, field, unit, outline, height, win])

  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

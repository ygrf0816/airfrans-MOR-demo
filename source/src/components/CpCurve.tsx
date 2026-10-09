import { useEffect, useRef } from 'react'
import { colors, fillLine, makeScale, niceTicks, setupCanvas, strokeLine, drawGrid } from '../lib/plot.ts'

interface Props {
  cp: number[]
  height?: number
  /** true 时把闭环折成上下表面两条弦向曲线 */
  splitSurfaces?: boolean
  /** 叠画用的对比曲线（可选，虚线） */
  compare?: number[]
  compareLabel?: string
  label?: string
}

/**
 * 压力系数曲线。横轴归一化弧长 s 0..1，纵轴 Cp 反向（负值在上，气动惯例）。
 */
export default function CpCurve({
  cp,
  height = 240,
  splitSurfaces = false,
  compare,
  compareLabel = '真实',
  label = '重构',
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const box = setupCanvas(el, height)
      if (!box) return
      const { ctx, w, h } = box
      const C = colors()
      const pad = { l: 46, r: 14, t: 14, b: 26 }
      const n = cp.length

      let cMin = Math.min(...cp)
      let cMax = Math.max(...cp)
      if (compare) {
        cMin = Math.min(cMin, ...compare)
        cMax = Math.max(cMax, ...compare)
      }
      cMin = Math.min(cMin, -0.2)
      cMax = Math.max(cMax, 0.4)
      const span = cMax - cMin
      cMin -= span * 0.08
      cMax += span * 0.08
      // Cp 反向：负值在上
      const sy = makeScale([cMax, cMin], [pad.t, h - pad.b])
      const yTicks = niceTicks(cMin, cMax, 5, (v) => v.toFixed(1))

      if (!splitSurfaces) {
        const sx = makeScale([0, 1], [pad.l, w - pad.r])
        drawGrid(ctx, w, h, pad, [
          { v: 0, label: '0 前缘' },
          { v: 0.25, label: '0.25' },
          { v: 0.5, label: '0.5 后缘' },
          { v: 0.75, label: '0.75' },
          { v: 1, label: '1 前缘' },
        ], yTicks, sx, sy)
        if (compare) {
          strokeLine(ctx, compare.map((v, i) => ({ x: sx(i / (n - 1)), y: sy(v) })), C.fg2, 1.2)
        }
        const pts = cp.map((v, i) => ({ x: sx(i / (n - 1)), y: sy(v) }))
        fillLine(ctx, pts, sy(cMax), 'rgba(34,211,238,0.28)', 'rgba(34,211,238,0.02)', h)
        strokeLine(ctx, pts, C.acc, 1.8)
      } else {
        // 上下表面两条弦向曲线：下标 0..127 上表面，255..127 下表面
        const half = Math.floor(n / 2)
        const sx = makeScale([0, 1], [pad.l, w - pad.r])
        drawGrid(ctx, w, h, pad, [
          { v: 0, label: '0 前缘' },
          { v: 0.25, label: '0.25' },
          { v: 0.5, label: '0.5' },
          { v: 0.75, label: '0.75' },
          { v: 1, label: '1 后缘' },
        ], yTicks, sx, sy)
        const up = cp.slice(0, half).map((v, i) => ({ x: sx(i / (half - 1)), y: sy(v) }))
        const lo: { x: number; y: number }[] = []
        for (let i = 0; i < half; i++) {
          const idx = n - 1 - i
          lo.push({ x: sx(i / (half - 1)), y: sy(cp[idx]) })
        }
        fillLine(ctx, up, sy(cMax), 'rgba(34,211,238,0.22)', 'rgba(34,211,238,0.02)', h)
        strokeLine(ctx, up, C.acc, 1.8)
        strokeLine(ctx, lo, C.warm, 1.8)
        if (compare) {
          const cup = compare.slice(0, half).map((v, i) => ({ x: sx(i / (half - 1)), y: sy(v) }))
          const clo: { x: number; y: number }[] = []
          for (let i = 0; i < half; i++) {
            clo.push({ x: sx(i / (half - 1)), y: sy(compare[n - 1 - i]) })
          }
          ctx.setLineDash([4, 3])
          strokeLine(ctx, cup, C.fg2, 1)
          strokeLine(ctx, clo, C.fg2, 1)
          ctx.setLineDash([])
        }
        // 图例
        ctx.font = '11px "PingFang SC", "Microsoft YaHei", sans-serif'
        ctx.textAlign = 'left'
        ctx.fillStyle = C.acc
        ctx.fillText('— 上表面（吸力侧）', pad.l + 8, pad.t + 12)
        ctx.fillStyle = C.warm
        ctx.fillText('— 下表面（压力侧）', pad.l + 8, pad.t + 26)
      }
      if (!splitSurfaces && compare) {
        ctx.font = '11px "PingFang SC", "Microsoft YaHei", sans-serif'
        ctx.textAlign = 'left'
        ctx.fillStyle = C.fg2
        ctx.fillText(`— ${compareLabel}`, pad.l + 8, pad.t + 12)
        ctx.fillStyle = C.acc
        ctx.fillText(`— ${label}`, pad.l + 8, pad.t + 26)
      }
      // 纵轴标签
      ctx.save()
      ctx.translate(11, h / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillStyle = C.fg2
      ctx.font = '11px "PingFang SC", "Microsoft YaHei", sans-serif'
      ctx.fillText('压力系数 Cp（负值在上）', 0, 0)
      ctx.restore()
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [cp, compare, height, splitSurfaces, compareLabel, label])

  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

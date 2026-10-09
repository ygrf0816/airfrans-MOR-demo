import { useEffect, useRef } from 'react'
import type { NacaParams } from '../lib/naca.ts'
import { camberCurve, nacaOutline } from '../lib/naca.ts'
import { colors, setupCanvas } from '../lib/plot.ts'

interface Props {
  params: NacaParams
  height?: number
  /** 是否显示弦线与弯度线 */
  showGuides?: boolean
  /** 是否填充渐变 */
  fill?: boolean
}

/**
 * 翼型形状视图。横轴按弦长归一化，画轮廓、弦线、弯度线，前缘加点标注。
 */
export default function AirfoilShape({
  params,
  height = 260,
  showGuides = true,
  fill = true,
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
      const { pts } = nacaOutline(params)
      const pad = { l: 36, r: 16, t: 18, b: 24 }
      const iw = w - pad.l - pad.r
      const ih = h - pad.t - pad.b
      // y 范围按轮廓自适应，至少 ±0.16
      let yMax = 0.16
      for (const [, py] of pts) yMax = Math.max(yMax, Math.abs(py) * 1.15)
      const sx = (v: number) => pad.l + v * iw
      const sy = (v: number) => pad.t + ih / 2 - (v / yMax) * (ih / 2)

      // 网格
      ctx.strokeStyle = C.grid
      ctx.lineWidth = 1
      for (let i = 0; i <= 4; i++) {
        const gx = pad.l + (iw * i) / 4
        ctx.beginPath()
        ctx.moveTo(gx, pad.t)
        ctx.lineTo(gx, pad.t + ih)
        ctx.stroke()
        ctx.fillStyle = C.fg2
        ctx.font = '10px "JetBrains Mono", monospace'
        ctx.textAlign = 'center'
        ctx.fillText((i / 4).toFixed(2), gx, h - 8)
      }
      for (let i = -1; i <= 1; i++) {
        const gy = sy((i * yMax) / 2)
        ctx.beginPath()
        ctx.moveTo(pad.l, gy)
        ctx.lineTo(pad.l + iw, gy)
        ctx.stroke()
        ctx.textAlign = 'right'
        ctx.fillStyle = C.fg2
        ctx.fillText(((i * yMax) / 2).toFixed(2), pad.l - 6, gy + 3)
      }

      // 弦线
      if (showGuides) {
        ctx.setLineDash([5, 4])
        ctx.strokeStyle = C.fg2
        ctx.beginPath()
        ctx.moveTo(sx(0), sy(0))
        ctx.lineTo(sx(1), sy(0))
        ctx.stroke()
        ctx.setLineDash([])
        // 弯度线
        const camber = camberCurve(params)
        ctx.strokeStyle = C.warm
        ctx.lineWidth = 1.2
        ctx.beginPath()
        camber.forEach(([cx, cy], i) => {
          if (i === 0) ctx.moveTo(sx(cx), sy(cy))
          else ctx.lineTo(sx(cx), sy(cy))
        })
        ctx.stroke()
      }

      // 轮廓填充
      if (fill) {
        const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih)
        grad.addColorStop(0, 'rgba(34, 211, 238, 0.30)')
        grad.addColorStop(0.5, 'rgba(34, 211, 238, 0.06)')
        grad.addColorStop(1, 'rgba(34, 211, 238, 0.22)')
        ctx.fillStyle = grad
        ctx.beginPath()
        pts.forEach(([px, py], i) => {
          if (i === 0) ctx.moveTo(sx(px), sy(py))
          else ctx.lineTo(sx(px), sy(py))
        })
        ctx.closePath()
        ctx.fill()
      }

      // 轮廓线
      ctx.strokeStyle = C.acc
      ctx.lineWidth = 2
      ctx.shadowColor = 'rgba(34, 211, 238, 0.55)'
      ctx.shadowBlur = 8
      ctx.beginPath()
      pts.forEach(([px, py], i) => {
        if (i === 0) ctx.moveTo(sx(px), sy(py))
        else ctx.lineTo(sx(px), sy(py))
      })
      ctx.closePath()
      ctx.stroke()
      ctx.shadowBlur = 0

      // 前缘标注
      const le = pts.reduce((a, b) => (b[0] < a[0] ? b : a), pts[0])
      ctx.fillStyle = C.white
      ctx.beginPath()
      ctx.arc(sx(le[0]), sy(le[1]), 3.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.font = '11px "PingFang SC", "Microsoft YaHei", sans-serif'
      ctx.textAlign = 'left'
      ctx.fillStyle = C.fg
      ctx.fillText('前缘', sx(le[0]) + 8, sy(le[1]) - 6)
      // 后缘标注
      ctx.textAlign = 'right'
      ctx.fillText('后缘', sx(1) - 6, sy(0) - 8)
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [params, height, showGuides, fill])

  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

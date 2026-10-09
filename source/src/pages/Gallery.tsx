import { useEffect, useMemo, useRef, useState } from 'react'
import './pages.css'
import { assetUrl, type DataState, type FrontPoint } from '../data/load.ts'
import { nacaOutline } from '../lib/naca.ts'
import { colors, makeScale, niceTicks, setupCanvas, strokeLine, drawGrid } from '../lib/plot.ts'

interface FigItem {
  file: string
  title: string
  source: string
  caption: string
}

/** 论文图画廊的一张图，加载失败显示图名，点击放大到接近全屏。 */
function FigCard({ item, onZoom }: { item: FigItem; onZoom: (item: FigItem) => void }) {
  const [failed, setFailed] = useState(false)
  return (
    <div className="panel">
      <div className="panel-title">
        <span>{item.title}</span>
        <span className="muted">{item.source}</span>
      </div>
      <div className="panel-body">
        {failed ? (
          <div className="skeleton" style={{
            height: 150, display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--fg-2)', fontSize: 13,
          }}>
            {item.title}（图片待接入）
          </div>
        ) : (
          <img
            src={assetUrl('figs/' + item.file)}
            alt={item.title}
            loading="lazy"
            onClick={() => onZoom(item)}
            title="点击放大"
            style={{ width: '100%', borderRadius: 'var(--rd-1)', display: 'block', cursor: 'zoom-in' }}
            onError={() => setFailed(true)}
          />
        )}
      </div>
      <div className="panel-body muted">{item.caption}</div>
    </div>
  )
}

/** 图片放大遮罩层：固定定位接近全屏，点遮罩或关闭按钮或按 Esc 复原。 */
function FigLightbox({ item, onClose }: { item: FigItem; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(3, 5, 10, 0.88)',
        backdropFilter: 'blur(6px)',
        zIndex: 60,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: '92vw',
          maxHeight: '92vh',
          display: 'grid',
          gap: 8,
          justifyItems: 'center',
        }}
      >
        <img
          src={assetUrl('figs/' + item.file)}
          alt={item.title}
          onClick={onClose}
          style={{
            maxWidth: '92vw',
            maxHeight: '78vh',
            objectFit: 'contain',
            borderRadius: 'var(--rd-2)',
            cursor: 'zoom-out',
            boxShadow: 'var(--sh-2)',
          }}
        />
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: 'var(--fg-0)', fontSize: 16, fontWeight: 600 }}>
            {item.title}
            <span className="muted" style={{ marginLeft: 8, fontWeight: 400 }}>{item.source}</span>
          </div>
          <div className="muted" style={{ marginTop: 4 }}>{item.caption}</div>
        </div>
      </div>
      <button
        onClick={onClose}
        style={{
          position: 'absolute',
          top: 18,
          right: 22,
          fontSize: 15,
          padding: '6px 14px',
          background: 'var(--bg-2)',
          border: '1px solid var(--line)',
          borderRadius: 'var(--rd-1)',
          color: 'var(--fg-0)',
        }}
      >
        关闭 ✕
      </button>
    </div>
  )
}

/** 小翼型缩略图，画在散点悬停提示里。 */
function MiniFoil({ L, P, Q, T, w = 120, h = 44 }: {
  L: number; P: number; Q: number; T: number; w?: number; h?: number
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const dpr = window.devicePixelRatio || 1
    el.width = w * dpr
    el.height = h * dpr
    const ctx = el.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const { pts } = nacaOutline([L, P, Q, T], 120)
    let yMax = 0.12
    for (const [, py] of pts) yMax = Math.max(yMax, Math.abs(py) * 1.2)
    const sx = (v: number) => 4 + v * (w - 8)
    const sy = (v: number) => h / 2 - (v / yMax) * (h / 2 - 3)
    ctx.strokeStyle = '#22d3ee'
    ctx.lineWidth = 1.4
    ctx.beginPath()
    pts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(sx(px), sy(py)) : ctx.lineTo(sx(px), sy(py))))
    ctx.closePath()
    ctx.stroke()
  }, [L, P, Q, T, w, h])
  return <canvas ref={ref} style={{ width: w, height: h }} />
}

/** 折中前沿散点：横轴带内损失，纵轴八度升力，悬停看参数与缩略图。 */
function FrontierPlot({ fronts, dbBest, upperBound }: {
  fronts: FrontPoint[]
  dbBest: number
  upperBound: number
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<FrontPoint | null>(null)
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 })
  const geomRef = useRef<{ pts: { x: number; y: number }[]; items: FrontPoint[] }>({ pts: [], items: [] })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const box = setupCanvas(el, 340)
      if (!box) return
      const { ctx, w, h } = box
      const C = colors()
      const pad = { l: 52, r: 16, t: 14, b: 30 }
      const dMax = Math.max(0.32, ...fronts.map((f) => f.delta)) * 1.08
      const cMax = Math.max(upperBound * 1.05, ...fronts.map((f) => f.cl8)) * 1.05
      const sx = makeScale([0, dMax], [pad.l, w - pad.r])
      const sy = makeScale([0, cMax], [h - pad.b, pad.t])
      drawGrid(ctx, w, h, pad,
        niceTicks(0, dMax, 6, (v) => v.toFixed(2)),
        niceTicks(0, cMax, 6, (v) => v.toFixed(1)), sx, sy)
      // 参考线：数据库最好与上界
      for (const [v, lab] of [[dbBest, '数据库最好 ' + dbBest.toFixed(3)], [upperBound, '理论上界 ' + upperBound.toFixed(4)]] as const) {
        ctx.strokeStyle = lab.startsWith('理论') ? C.warm : C.ok
        ctx.setLineDash([6, 4])
        ctx.beginPath()
        ctx.moveTo(pad.l, sy(v))
        ctx.lineTo(w - pad.r, sy(v))
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = lab.startsWith('理论') ? C.warm : C.ok
        ctx.font = '10px "PingFang SC", sans-serif'
        ctx.textAlign = 'right'
        ctx.fillText(lab, w - pad.r - 4, sy(v) - 4)
      }
      const pts: { x: number; y: number }[] = []
      for (const f of fronts) {
        const px = sx(f.delta)
        const py = sy(f.cl8)
        pts.push({ x: px, y: py })
        ctx.fillStyle = f.feasible ? C.acc : 'rgba(148, 163, 184, 0.5)'
        ctx.beginPath()
        ctx.arc(px, py, hover === f ? 5 : 3, 0, Math.PI * 2)
        ctx.fill()
        if (hover === f) {
          ctx.strokeStyle = C.white
          ctx.lineWidth = 1.2
          ctx.stroke()
        }
      }
      geomRef.current = { pts, items: fronts }
      // 轴标签
      ctx.fillStyle = C.fg2
      ctx.font = '11px "PingFang SC", sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('带内损失 Δ（越小越好）', pad.l + (w - pad.l - pad.r) / 2, h - 6)
      ctx.save()
      ctx.translate(12, pad.t + (h - pad.t - pad.b) / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('八度升力 CL8（越大越好）', 0, 0)
      ctx.restore()
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [fronts, dbBest, upperBound, hover])

  const onMove = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    let best: FrontPoint | null = null
    let bd = 14 * 14
    geomRef.current.pts.forEach((p, i) => {
      const d = (p.x - mx) ** 2 + (p.y - my) ** 2
      if (d < bd) { bd = d; best = geomRef.current.items[i] }
    })
    setHover(best)
    setHoverPos({ x: Math.min(mx + 14, rect.width - 170), y: Math.max(8, my - 20) })
  }

  return (
    <div ref={wrapRef} style={{ position: 'relative' }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
      {hover && (
        <div style={{
          position: 'absolute',
          left: hoverPos.x,
          top: hoverPos.y,
          background: 'var(--bg-1)',
          border: '1px solid var(--acc-line)',
          borderRadius: 'var(--rd-1)',
          padding: '8px 10px',
          pointerEvents: 'none',
          boxShadow: 'var(--sh-2)',
          zIndex: 5,
        }}>
          <MiniFoil L={hover.L} P={hover.P} Q={hover.Q} T={hover.T} />
          <div className="num" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
            L={hover.L.toFixed(2)} P={hover.P.toFixed(2)} Q={hover.Q} T={hover.T.toFixed(2)}<br />
            CL8={hover.cl8.toFixed(3)} Δ={hover.delta.toFixed(3)}<br />
            {hover.feasible ? '三项约束满足' : '有约束不满足'}
          </div>
        </div>
      )}
    </div>
  )
}

/** 收敛过程回放：三个种子各一条线，带播放与进度条。 */
function HistoryReplay({ history }: { history: DataState['results']['history'] }) {
  const seeds = useMemo(() => [...new Set(history.map((h) => h.seed))].sort(), [history])
  const genMax = useMemo(() => Math.max(...history.map((h) => h.gen), 1), [history])
  const [t, setT] = useState(genMax)
  const [playing, setPlaying] = useState(false)
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      setT((v) => (v >= genMax ? 0 : v + Math.max(1, Math.round(genMax / 90))))
    }, 80)
    return () => window.clearInterval(id)
  }, [playing, genMax])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const box = setupCanvas(el, 260)
      if (!box) return
      const { ctx, w, h } = box
      const C = colors()
      const pad = { l: 46, r: 14, t: 12, b: 28 }
      const cMax = Math.max(...history.map((p) => p.bestCl8)) * 1.06
      const cMin = Math.min(...history.map((p) => p.bestCl8)) * 0.94
      const sx = makeScale([0, genMax], [pad.l, w - pad.r])
      const sy = makeScale([cMin, cMax], [h - pad.b, pad.t])
      drawGrid(ctx, w, h, pad,
        niceTicks(0, genMax, 6, (v) => String(Math.round(v))),
        niceTicks(cMin, cMax, 5, (v) => v.toFixed(2)), sx, sy)
      const seedColors = [C.acc, C.warm, C.ok]
      seeds.forEach((seed, si) => {
        const pts = history
          .filter((p) => p.seed === seed && p.gen <= t)
          .sort((a, b) => a.gen - b.gen)
          .map((p) => ({ x: sx(p.gen), y: sy(p.bestCl8) }))
        strokeLine(ctx, pts, seedColors[si % 3], 1.8)
      })
      // 进度竖线
      ctx.strokeStyle = C.white
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      ctx.moveTo(sx(t), pad.t)
      ctx.lineTo(sx(t), h - pad.b)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = C.fg2
      ctx.font = '11px "PingFang SC", sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('进化代数', pad.l + (w - pad.l - pad.r) / 2, h - 6)
      seeds.forEach((seed, si) => {
        ctx.fillStyle = seedColors[si % 3]
        ctx.textAlign = 'left'
        ctx.fillText(`种子 ${seed}`, pad.l + 8 + si * 76, pad.t + 10)
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [history, seeds, genMax, t])

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
        <button onClick={() => setPlaying((p) => !p)}
          style={{ background: 'var(--acc-dim)', borderColor: 'var(--acc-line)', color: 'var(--acc-strong)' }}>
          {playing ? '暂停' : '播放'}
        </button>
        <input type="range" min={0} max={genMax} step={1} value={Math.min(t, genMax)}
          style={{ flex: 1 }}
          onChange={(e) => { setPlaying(false); setT(Number(e.target.value)) }} />
        <span className="num" style={{ fontSize: 12, color: 'var(--fg-1)' }}>
          第 {Math.min(t, genMax)} / {genMax} 代
        </span>
      </div>
      <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
    </div>
  )
}

/** 直方图小图。 */
function MiniHist({ edges, counts, label, fmt }: {
  edges: number[]
  counts: number[]
  label: string
  fmt: (v: number) => string
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const box = setupCanvas(el, 90)
    if (!box) return
    const { ctx, w, h } = box
    const C = colors()
    const cMax = Math.max(...counts, 1)
    const bw = w / counts.length
    counts.forEach((c, i) => {
      const bh = (c / cMax) * (h - 22)
      ctx.fillStyle = 'rgba(34, 211, 238, 0.55)'
      ctx.fillRect(i * bw + 1, h - 16 - bh, bw - 2, bh)
    })
    ctx.fillStyle = C.fg2
    ctx.font = '9px "JetBrains Mono", monospace'
    ctx.textAlign = 'left'
    ctx.fillText(fmt(edges[0]), 0, h - 4)
    ctx.textAlign = 'right'
    ctx.fillText(fmt(edges[edges.length - 1]), w, h - 4)
  }, [edges, counts, label, fmt])
  // 标题放到画布外面。原先画在画布内部顶端，最高那根柱子会正好顶到它，重叠在一起。
  return (
    <div>
      <div className="muted" style={{ fontSize: 11, marginBottom: 2 }}>{label}</div>
      <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
    </div>
  )
}

export default function Gallery({ data }: { data: DataState }) {
  const [figs, setFigs] = useState<FigItem[]>([])
  const [figsMissing, setFigsMissing] = useState(false)
  const [zoom, setZoom] = useState<FigItem | null>(null)
  const r = data.results
  const ds = r.dataset

  useEffect(() => {
    let alive = true
    fetch(assetUrl('figs/index.json'))
      .then((res) => (res.ok ? res.json() : null))
      .then((j) => {
        if (!alive) return
        if (j && Array.isArray(j.items)) setFigs(j.items)
        else setFigsMissing(true)
      })
      .catch(() => alive && setFigsMissing(true))
    return () => { alive = false }
  }, [])

  return (
    <section className="page page-in">
      <div className="page-head">
        <h2>成果总览</h2>
        <span className="hint">
          折中前沿、收敛过程与数据集体检。带内损失是八度升力与两端攻角升力的最大落差，越小越稳。
        </span>
        {data.mode === 'mock' && <span className="badge mock">示例数据</span>}
      </div>

      <div className="cols-2">
        <div className="panel">
          <div className="panel-title">
            <span>折中前沿（悬停看参数与形状）</span>
            <span className="muted">{r.fronts.length} 个方案</span>
          </div>
          <div className="panel-body">
            <FrontierPlot fronts={r.fronts} dbBest={r.objective.dbDesignBest}
              upperBound={r.objective.bandUpperBound} />
          </div>
        </div>
        <div className="panel">
          <div className="panel-title">
            <span>收敛过程回放</span>
            <span className="muted">三个随机种子</span>
          </div>
          <div className="panel-body">
            <HistoryReplay history={r.history} />
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">数据集体检</div>
        <div className="panel-body" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--sp-4)' }}>
          <div>
            <div className="muted">总样本数</div>
            <div className="num" style={{ fontSize: 24, color: 'var(--fg-0)' }}>{ds.nTotal}</div>
            <div className="muted">训练 {ds.nTrain} · 测试 {ds.nTest}</div>
          </div>
          <div>
            <div className="muted">族别比例</div>
            <div style={{ display: 'flex', height: 12, borderRadius: 6, overflow: 'hidden', margin: '8px 0 4px' }}>
              <span style={{ width: `${(ds.nFiveDigit / ds.nTotal) * 100}%`, background: 'var(--acc)' }} />
              <span style={{ flex: 1, background: 'var(--warm)' }} />
            </div>
            <div className="muted">五位族 {ds.nFiveDigit} · 四位族 {ds.nFourDigit}</div>
          </div>
          <MiniHist edges={ds.aoaHist.binEdges} counts={ds.aoaHist.counts} label="攻角分布"
            fmt={(v) => v.toFixed(0) + '°'} />
          <MiniHist edges={ds.reHist.binEdges} counts={ds.reHist.counts} label="雷诺数分布"
            fmt={(v) => (v / 1e6).toFixed(1) + 'M'} />
          <div>
            <div className="muted">八个划分的规模</div>
            <div style={{ display: 'grid', gap: 3, marginTop: 6 }}>
              {ds.splitCards.map((s) => (
                <div key={s.name} style={{ display: 'grid', gridTemplateColumns: '64px 1fr 84px', gap: 6, alignItems: 'center', fontSize: 11 }}>
                  <span className="num">{s.name}</span>
                  {/* 只留数值条。原先底下还有一层浅色槽底，两根叠在一起看着像虚影。 */}
                  <span style={{ height: 5, borderRadius: 3, overflow: 'hidden', display: 'block' }}>
                    <span style={{ display: 'block', height: '100%', width: `${(s.train / 800) * 100}%`, background: 'var(--acc)' }} />
                  </span>
                  <span className="num" style={{ color: 'var(--fg-2)' }}>{s.train}/{s.test}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">
          <span>论文图画廊</span>
          <span className="muted">{figs.length > 0 ? `${figs.length} 张` : '索引待接入'}</span>
        </div>
        {figsMissing && (
          <div className="panel-body muted">
            figs/index.json 还没生成。图片由导出脚本写进 public/data/figs/，生成后这里自动显示。
          </div>
        )}
        <div className="panel-body" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 'var(--sp-4)' }}>
          {figs.map((f) => (
            <FigCard key={f.file} item={f} onZoom={setZoom} />
          ))}
          {figs.length === 0 && !figsMissing &&
            ['数据集总览', '折中前沿', '收敛曲线', '案例对比', '反演多解性', '可信度核对'].map((t) => (
              <FigCard key={t} item={{ file: t + '.webp', title: t, source: '占位', caption: '图片待接入' }} onZoom={setZoom} />
            ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">结果使用说明</div>
        <div className="panel-body" style={{ display: 'grid', gap: 4 }}>
          {r.caveats.map((c, i) => (
            <span key={i} className="muted">· {c}</span>
          ))}
        </div>
      </div>
      {zoom && <FigLightbox item={zoom} onClose={() => setZoom(null)} />}
    </section>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import './pages.css'
import type { DataState } from '../data/load.ts'
import { analyticCp, type SampleItem } from '../data/mock.ts'
import SamplePicker from '../components/SamplePicker.tsx'
import CpCurve from '../components/CpCurve.tsx'
import AirfoilShape from '../components/AirfoilShape.tsx'
import MorFlowDiagram from '../components/MorFlowDiagram.tsx'
import { colors, setupCanvas, strokeLine } from '../lib/plot.ts'

const base = import.meta.env.BASE_URL || './'

/** 用 cpBasis 与样本 POD 坐标重构压力曲线。秩 r 取前 r 个模态。 */
function reconstruct(mu: number[], phi: number[][], alpha: number[], r: number): number[] {
  const n = mu.length
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let v = mu[i]
    for (let k = 0; k < r; k++) v += alpha[k] * phi[i][k]
    out[i] = v
  }
  return out
}

/** 逐点绝对误差。 */
function pointErr(a: number[], b: number[]): number[] {
  return a.map((v, i) => Math.abs(v - b[i]))
}

/** 相对 L2 误差。 */
function relL2(a: number[], b: number[]): number {
  let num = 0
  let den = 0
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - b[i]) ** 2
    den += b[i] ** 2
  }
  return Math.sqrt(num / (den || 1))
}

/** 模态贡献 = |alpha_k| * 模态范数，占所有模态的比例。 */
function modeContrib(phi: number[][], alpha: number[]): number[] {
  const n = phi.length
  const r = phi[0].length
  const out: number[] = []
  let sum = 0
  for (let k = 0; k < r; k++) {
    let norm = 0
    for (let i = 0; i < n; i++) norm += phi[i][k] ** 2
    const c = Math.abs(alpha[k]) * Math.sqrt(norm)
    out.push(c)
    sum += c
  }
  return out.map((v) => (sum > 0 ? v / sum : 0))
}

/** 累计能量：模态方差占比（用 alpha 平方近似）。 */
function cumulativeEnergy(alphaAll: number[][], r: number): number[] {
  const k = alphaAll[0].length
  const varK = new Array<number>(k).fill(0)
  for (const a of alphaAll) for (let j = 0; j < k; j++) varK[j] += a[j] * a[j]
  const total = varK.reduce((x, y) => x + y, 0) || 1
  const out: number[] = []
  let acc = 0
  for (let j = 0; j < Math.min(r, k); j++) {
    acc += varK[j]
    out.push(acc / total)
  }
  return out
}

/** 误差带小图：逐点误差随弧长的曲线，下方填充。 */
function ErrBand({ err, height = 90 }: { err: number[]; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const box = setupCanvas(el, height)
      if (!box) return
      const { ctx, w, h } = box
      const C = colors()
      const pad = { l: 46, r: 14, t: 6, b: 20 }
      const eMax = Math.max(...err, 1e-6)
      const sx = (v: number) => pad.l + v * (w - pad.l - pad.r)
      const sy = (v: number) => pad.t + (h - pad.t - pad.b) * (1 - v / eMax)
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = C.fg2
      ctx.textAlign = 'right'
      ctx.fillText(eMax.toFixed(3), pad.l - 6, pad.t + 8)
      ctx.fillText('0', pad.l - 6, h - pad.b)
      const pts = err.map((v, i) => ({ x: sx(i / (err.length - 1)), y: sy(v) }))
      ctx.fillStyle = 'rgba(251, 146, 60, 0.18)'
      ctx.beginPath()
      ctx.moveTo(pts[0].x, sy(0))
      pts.forEach((p) => ctx.lineTo(p.x, p.y))
      ctx.lineTo(pts[pts.length - 1].x, sy(0))
      ctx.closePath()
      ctx.fill()
      strokeLine(ctx, pts, C.warm, 1.4)
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [err, height])
  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

/** 累计能量曲线并标当前秩。 */
function EnergyCurve({ energy, rank, height = 120 }: { energy: number[]; rank: number; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const box = setupCanvas(el, height)
      if (!box) return
      const { ctx, w, h } = box
      const C = colors()
      const pad = { l: 40, r: 12, t: 10, b: 22 }
      const sx = (k: number) => pad.l + ((k - 1) / (energy.length - 1)) * (w - pad.l - pad.r)
      const sy = (v: number) => pad.t + (h - pad.t - pad.b) * (1 - v)
      strokeLine(ctx, energy.map((v, i) => ({ x: sx(i + 1), y: sy(v) })), C.acc, 1.6)
      for (let i = 0; i < energy.length; i++) {
        ctx.fillStyle = i + 1 <= rank ? C.acc : C.fg2
        ctx.beginPath()
        ctx.arc(sx(i + 1), sy(energy[i]), 3, 0, Math.PI * 2)
        ctx.fill()
      }
      // 当前秩竖线
      ctx.strokeStyle = C.warm
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      ctx.moveTo(sx(rank), pad.t)
      ctx.lineTo(sx(rank), h - pad.b)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = C.fg2
      ctx.font = '10px "PingFang SC", sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(`秩 ${rank} 累计能量 ${(energy[rank - 1] * 100).toFixed(1)}%`, sx(rank), pad.t + 4)
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (el.parentElement) ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [energy, rank, height])
  return <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
}

/** 单个模态缩略曲线与贡献条。 */
function ModeRow({
  k, phi, contrib, active, onToggle,
}: {
  k: number
  phi: number[][]
  contrib: number
  active: boolean
  onToggle: () => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const box = setupCanvas(el, 36)
    if (!box) return
    const { ctx, w, h } = box
    const C = colors()
    const col = phi.map((row) => row[k])
    const m = Math.max(...col.map(Math.abs), 1e-9)
    const pts = col.map((v, i) => ({
      x: (i / (col.length - 1)) * w,
      y: h / 2 - (v / m) * (h / 2 - 3),
    }))
    ctx.strokeStyle = C.grid
    ctx.beginPath()
    ctx.moveTo(0, h / 2)
    ctx.lineTo(w, h / 2)
    ctx.stroke()
    strokeLine(ctx, pts, active ? C.warm : C.acc, 1.3)
  }, [phi, k, active])
  return (
    <button
      onClick={onToggle}
      style={{
        display: 'grid',
        gridTemplateColumns: '34px 1fr 52px',
        gap: 8,
        alignItems: 'center',
        textAlign: 'left',
        padding: '3px 6px',
        ...(active
          ? { background: 'var(--warm-dim)', borderColor: 'var(--warm)' }
          : {}),
      }}
    >
      <span className="num" style={{ fontSize: 11 }}>φ{k + 1}</span>
      <canvas ref={ref} style={{ display: 'block', width: '100%' }} />
      <span style={{ display: 'grid', gap: 2 }}>
        <span style={{ height: 4, borderRadius: 2, background: 'var(--bg-3)', overflow: 'hidden', display: 'block' }}>
          <span style={{
            display: 'block', height: '100%', borderRadius: 2,
            width: `${Math.min(100, contrib * 100 * 2.2)}%`,
            background: active ? 'var(--warm)' : 'var(--acc)',
            transition: 'width 240ms',
          }} />
        </span>
        <span className="num" style={{ fontSize: 10, color: 'var(--fg-2)' }}>
          {(contrib * 100).toFixed(1)}%
        </span>
      </span>
    </button>
  )
}

interface UploadResult {
  name: string
  alpha: number[]
  err: number
  cp: number[]
}

/** 字节数转成好读的 MB 文本，缺值时给占位。 */
function mb(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  return (bytes / 1048576).toFixed(2) + ' MB'
}

export default function Mor({ data }: { data: DataState }) {
  const [selected, setSelected] = useState<SampleItem | null>(null)
  const [rank, setRank] = useState(4)
  const [soloMode, setSoloMode] = useState<number | null>(null)
  const [upload, setUpload] = useState<UploadResult | null>(null)
  const [uploadErr, setUploadErr] = useState('')
  const [dragOver, setDragOver] = useState(false)

  const basis = data.cpBasis
  const sample = selected ?? data.samples[0]

  // 真实曲线：优先样本自带 cp，缺失时用解析近似
  const truth = useMemo(
    () => sample.cp ?? analyticCp(sample.params, sample.uInf, sample.aoa),
    [sample],
  )
  const alpha = sample.alpha

  const rec = useMemo(() => {
    if (soloMode !== null) {
      // 单独看某一个模态修了什么：均值 + 只加这一条
      return reconstruct(basis.mu, basis.phi, alpha, 0).map(
        (v, i) => v + alpha[soloMode] * basis.phi[i][soloMode],
      )
    }
    return reconstruct(basis.mu, basis.phi, alpha, rank)
  }, [basis, alpha, rank, soloMode])

  const err = useMemo(() => pointErr(rec, truth), [rec, truth])
  const errL2 = useMemo(() => relL2(rec, truth), [rec, truth])
  const contrib = useMemo(() => modeContrib(basis.phi, alpha), [basis, alpha])
  const energy = useMemo(
    () => cumulativeEnergy(data.samples.map((s) => s.alpha), basis.podR),
    [data.samples, basis.podR],
  )

  // 上传结果统一走同一套模态投影：重采样到 basis.s、投影、重构、算误差。
  const projectAndStore = (name: string, sArr: number[], cpArr: number[]) => {
    if (cpArr.length < 2) throw new Error('压力曲线点数太少')
    const cp: number[] = basis.s.map((sv) => {
      const pos = sv * (sArr.length - 1)
      const i0 = Math.floor(pos)
      const i1 = Math.min(sArr.length - 1, i0 + 1)
      const f = pos - i0
      return cpArr[i0] * (1 - f) + cpArr[i1] * f
    })
    const a: number[] = []
    for (let k = 0; k < basis.podR; k++) {
      let dot = 0
      for (let i = 0; i < basis.nPts; i++) dot += (cp[i] - basis.mu[i]) * basis.phi[i][k]
      a.push(dot)
    }
    const rc = reconstruct(basis.mu, basis.phi, a, basis.podR)
    setUpload({ name, alpha: a, err: relL2(rc, cp), cp })
  }

  const SUPPORTED =
    '支持三种输入。一，原始 AirfRANS 翼型文件 *_aerofoil.vtp（文件名带风速与攻角）。' +
    '二，本作品导出的 JSON（airfrans_to_web.py 的产物，含 256 点 cp 数组）。' +
    '三，列名为 name,uInf,aoa,s,cp 的 CSV（正好 256 行）。'

  const parseOneFile = async (file: File) => {
    const lower = file.name.toLowerCase()
    if (lower.endsWith('.vtp')) {
      // 原始 AirfRANS 翼型文件，浏览器直接解析，不需要预先跑 Python
      try {
        const buf = await file.arrayBuffer()
        const { parseAerofoilVtp } = await import('../lib/vtp.ts')
        const got = await parseAerofoilVtp(file.name, buf)
        projectAndStore(got.name, got.cp256.map((_: number, i: number) => i / (got.cp256.length - 1)), got.cp256)
        return
      } catch (e) {
        throw new Error(
          '这个 vtp 文件解析不了：' + (e instanceof Error ? e.message : String(e)) +
          '。需要的是 AirfRANS 数据集里的 *_aerofoil.vtp 翼型文件。',
        )
      }
    }
    const text = await file.text()
    let name = file.name
    let sArr: number[] = []
    let cpArr: number[] = []
    if (lower.endsWith('.json')) {
      const j = JSON.parse(text)
      const item = Array.isArray(j) ? j[0] : j.items?.[0] ?? j
      if (!item || !Array.isArray(item.cp) || item.cp.length !== 256) {
        throw new Error('json 里需要有 cp 数组，长度 256（与 samples.json 同结构）')
      }
      name = item.name ?? name
      cpArr = item.cp.map(Number)
      sArr = item.s ?? basis.s
    } else if (lower.endsWith('.csv')) {
      const lines = text.trim().split(/\r?\n/)
      const head = lines[0].split(',').map((s) => s.trim())
      const need = ['name', 'uInf', 'aoa', 's', 'cp']
      if (head.join(',') !== need.join(',')) {
        throw new Error(`csv 列名必须是 ${need.join(',')}，当前是 ${head.join(',')}`)
      }
      for (const ln of lines.slice(1)) {
        const cells = ln.split(',')
        if (cells.length < 5) continue
        name = cells[0]
        sArr.push(Number(cells[3]))
        cpArr.push(Number(cells[4]))
      }
      if (cpArr.length !== 256) {
        throw new Error(`csv 需要正好 256 行数据，当前 ${cpArr.length} 行`)
      }
    } else {
      throw new Error('认不出这个文件的格式。' + SUPPORTED)
    }
    projectAndStore(name, sArr, cpArr)
  }

  // 入口：单文件或整目录。目录里有多个 _aerofoil.vtp 时取第一个并说明。
  const parseUpload = async (files: FileList | File[]) => {
    setUploadErr('')
    setUpload(null)
    try {
      const list = Array.from(files)
      if (list.length === 0) throw new Error('没有收到文件')
      let file = list[0]
      if (list.length > 1) {
        const vtp = list.filter((f) => f.name.toLowerCase().endsWith('_aerofoil.vtp'))
        if (vtp.length > 0) {
          file = vtp[0]
          setUploadErr(`目录里有 ${vtp.length} 个翼型文件，取了第一个：${file.name}`)
        } else {
          throw new Error('目录里没找到 *_aerofoil.vtp 翼型文件。' + SUPPORTED)
        }
      }
      await parseOneFile(file)
      if (list.length === 1) setUploadErr('')
    } catch (e) {
      setUploadErr(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <section className="page page-in">
      <div className="page-head">
        <h2>降维演示</h2>
        <span className="hint">
          一千条真实压力曲线，用十几个数字就能概括。POD（本征正交分解）把所有曲线的
          共性拆成 8 条模态，每条样本只需要 8 个坐标。
        </span>
        {data.mode === 'mock' && <span className="badge mock">示例数据</span>}
      </div>

      {/* 降维原理流程图：高维快照 → 8 条模态 → 重构对比。
          由真实样本与真实模态现场矢量绘制，不是静态图片。 */}
      <MorFlowDiagram base={base} />

      <div className="cols-2">
        <SamplePicker data={data} onSelect={setSelected} selected={selected} />
        <div className="panel">
          <div className="panel-title">
            <span>选中样本</span>
            <span className="muted">{sample.series === '5digit' ? '五位族' : '四位族'}</span>
          </div>
          <div className="panel-body">
            <AirfoilShape params={sample.params} height={170} showGuides={false} />
          </div>
          <div className="panel-body" style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <span className="muted">名称 <span className="num">{sample.name}</span></span>
            <span className="muted">攻角 <span className="num">{sample.aoa.toFixed(2)}°</span></span>
            <span className="muted">来流 <span className="num">{sample.uInf.toFixed(1)} m/s</span></span>
            <span className="muted">官方升力 <span className="num">{sample.clOfficial.toFixed(4)}</span></span>
            <span className="muted">划分 <span className="num">{sample.split}</span></span>
          </div>
          {/* 体积与点数对照：数字全部取自 samples.json 的真实字段，不写死 */}
          <div className="panel-body" style={{ display: 'grid', gap: 6, fontSize: 12 }}>
            <div className="muted">
              原始样本：内部体积文件{' '}
              <span className="num">{mb(sample.bytesVol)}</span>，
              <span className="num">{sample.nVol?.toLocaleString() ?? '—'}</span> 个网格点；
              表面文件 <span className="num">{mb(sample.bytesSurf)}</span>，
              <span className="num">{sample.nSurf?.toLocaleString() ?? '—'}</span> 个表面点。
              本次降维只用表面文件，体积文件没有进网页。
            </div>
            <div className="muted">
              降维结果：压力曲线用 <span className="num">8</span> 个模态坐标表示，
              8 个 float32 就是 <span className="num">32 字节</span>；
              乘的曲线基是 8 乘 256 个 float32，约 <span className="num">8 KB</span>，
              这份基全部样本共用，不按样本重复计；样本自己的几何 128 维是
              <span className="num">512 字节</span>。
            </div>
            <div className="muted">
              一条样本的表面文件约 {mb(sample.bytesSurf ?? 60000)}，压到 8 个坐标只有 32 字节，
              加上共用的基也只占几十 KB。
            </div>
          </div>
        </div>
      </div>

      <div className="cols-3" style={{ gridTemplateColumns: 'minmax(0,1.6fr) minmax(0,1fr)' }}>
        <div className="panel">
          <div className="panel-title">
            <span>重构对比 · 秩 r = {rank}</span>
            <span className="muted">相对 L2 误差 {(errL2 * 100).toFixed(2)}%</span>
          </div>
          <div className="ctl">
            <input type="range" min={1} max={basis.podR} step={1} value={rank}
              onChange={(e) => { setRank(Number(e.target.value)); setSoloMode(null) }} />
            <span className="ctl-note">
              前几个模态抓前缘吸力峰，后面的模态修小细节。
              {soloMode !== null && ' 当前在单独查看模态 φ' + (soloMode + 1) + '，点秩滑块返回。'}
            </span>
          </div>
          <div className="panel-body">
            <CpCurve cp={rec} compare={truth} compareLabel="真实曲线"
              label={soloMode !== null ? `仅模态 φ${soloMode + 1}` : `秩 ${rank} 重构`} height={260} />
          </div>
          <div className="panel-title">逐点误差带</div>
          <div className="panel-body">
            <ErrBand err={err} />
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">
            <span>8 条模态（点击单独叠加）</span>
          </div>
          <div className="panel-body" style={{ display: 'grid', gap: 2 }}>
            {contrib.map((c, k) => (
              <ModeRow key={k} k={k} phi={basis.phi} contrib={c}
                active={soloMode === k}
                onToggle={() => setSoloMode((v) => (v === k ? null : k))} />
            ))}
          </div>
          <div className="panel-title">累计能量</div>
          <div className="panel-body">
            <EnergyCurve energy={energy} rank={rank} />
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">
          <span>上传你自己的数据</span>
          <span className="muted">支持原始 airRANS 格式数据</span>
        </div>
        <div
          className="panel-body"
          style={{
            border: `1px dashed ${dragOver ? 'var(--acc)' : 'var(--line)'}`,
            borderRadius: 'var(--rd-1)',
            margin: '0 var(--sp-4)',
            padding: 'var(--sp-4)',
            textAlign: 'center',
            background: dragOver ? 'var(--acc-dim)' : 'transparent',
            transition: 'background 160ms, border-color 160ms',
          }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            if (e.dataTransfer.files.length > 0) void parseUpload(e.dataTransfer.files)
          }}
        >
          <p style={{ margin: 0 }}>
            把文件或整个目录拖到这里，或
            <label style={{ color: 'var(--acc-strong)', cursor: 'pointer', marginLeft: 4 }}>
              选择文件
              <input type="file" accept=".json,.csv,.vtp" style={{ display: 'none' }}
                onChange={(e) => { if (e.target.files?.length) void parseUpload(e.target.files) }} />
            </label>
            <label style={{ color: 'var(--acc-strong)', cursor: 'pointer', marginLeft: 10 }}>
              选择目录
              <input type="file" style={{ display: 'none' }}
                // @ts-expect-error 非标准属性，浏览器普遍支持
                webkitdirectory=""
                onChange={(e) => { if (e.target.files?.length) void parseUpload(e.target.files) }} />
            </label>
          </p>
          <p className="muted" style={{ margin: '8px 0 0' }}>
            不需要预先跑 Python 脚本，原始数据可以直接拖进来。
            支持三种输入：原始 AirfRANS 翼型文件 *_aerofoil.vtp（文件名带风速与攻角），
            本作品导出的 JSON（airfrans_to_web.py 的产物），
            列名为 name,uInf,aoa,s,cp 的 CSV。自动识别，不需要选格式。
            目录里有多个翼型文件时取第一个并说明取了哪个。
          </p>
        </div>
        {uploadErr && (
          <div className="panel-body" style={{ color: uploadErr.startsWith('目录里有') ? 'var(--fg-2)' : 'var(--bad)' }}>
            {uploadErr.startsWith('目录里有') ? uploadErr : '格式不对：' + uploadErr}
          </div>
        )}
        {upload && (
          <div className="panel-body" style={{ display: 'grid', gap: 8 }}>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <span className="muted">名称 <span className="num">{upload.name}</span></span>
              <span className="muted">
                全秩重构相对 L2 误差 <span className="num">{(upload.err * 100).toFixed(2)}%</span>
              </span>
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {upload.alpha.map((a, k) => (
                <span key={k} className="badge">
                  α{k + 1}={a.toFixed(3)}
                </span>
              ))}
            </div>
            <CpCurve cp={upload.cp} label="上传曲线" height={200} />
          </div>
        )}
      </div>
    </section>
  )
}

import { useMemo, useState } from 'react'
import '../pages/pages.css'
import type { DataState } from '../data/load.ts'
import type { SampleItem } from '../data/mock.ts'

interface Props {
  data: DataState
  onSelect: (s: SampleItem) => void
  selected: SampleItem | null
}

/** 样本选择器：按名字搜、随机抽、按攻角与雷诺数筛。 */
export default function SamplePicker({ data, onSelect, selected }: Props) {
  const [query, setQuery] = useState('')
  const [aoaMin, setAoaMin] = useState('')
  const [aoaMax, setAoaMax] = useState('')
  const [reMin, setReMin] = useState('')
  const [reMax, setReMax] = useState('')

  const reOf = (s: SampleItem) => s.uInf * 68500 // 弦长 1m、标准大气的近似雷诺数

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return data.samples.filter((s) => {
      if (q && !s.name.toLowerCase().includes(q)) return false
      if (aoaMin !== '' && s.aoa < Number(aoaMin)) return false
      if (aoaMax !== '' && s.aoa > Number(aoaMax)) return false
      const re = reOf(s)
      if (reMin !== '' && re < Number(reMin) * 1e6) return false
      if (reMax !== '' && re > Number(reMax) * 1e6) return false
      return true
    })
  }, [data.samples, query, aoaMin, aoaMax, reMin, reMax])

  const randomPick = () => {
    const pool = filtered.length > 0 ? filtered : data.samples
    const idx = Math.floor((Date.now() % 1000) / 1000 * pool.length) % pool.length
    onSelect(pool[idx])
  }

  return (
    <div className="panel">
      <div className="panel-title">
        <span>样本选择（共 {data.samples.length} 条，筛出 {filtered.length} 条）</span>
        <button style={{ fontSize: 11, padding: '2px 8px' }} onClick={randomPick}>
          随机抽一条
        </button>
      </div>
      <div className="panel-body" style={{ display: 'grid', gap: 8 }}>
        <input
          type="text"
          placeholder="按名字搜索，例如 demo5 或 demo4"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <span className="muted">攻角°</span>
          <input style={{ width: 64 }} type="number" placeholder="最小" value={aoaMin}
            onChange={(e) => setAoaMin(e.target.value)} />
          <span className="muted">到</span>
          <input style={{ width: 64 }} type="number" placeholder="最大" value={aoaMax}
            onChange={(e) => setAoaMax(e.target.value)} />
          <span className="muted">雷诺数(百万)</span>
          <input style={{ width: 64 }} type="number" placeholder="最小" value={reMin}
            onChange={(e) => setReMin(e.target.value)} />
          <span className="muted">到</span>
          <input style={{ width: 64 }} type="number" placeholder="最大" value={reMax}
            onChange={(e) => setReMax(e.target.value)} />
        </div>
        <div style={{ maxHeight: 180, overflow: 'auto', display: 'grid', gap: 2 }}>
          {filtered.slice(0, 80).map((s) => (
            <button
              key={s.i}
              style={{
                textAlign: 'left',
                fontFamily: 'var(--mono)',
                fontSize: 11,
                padding: '3px 8px',
                ...(selected?.i === s.i
                  ? { background: 'var(--acc-dim)', borderColor: 'var(--acc-line)', color: 'var(--acc-strong)' }
                  : {}),
              }}
              onClick={() => onSelect(s)}
            >
              {s.name} · α={s.aoa.toFixed(2)}° · CL={s.clOfficial.toFixed(3)} · {s.split}
            </button>
          ))}
          {filtered.length > 80 && (
            <span className="muted">… 只显示前 80 条，继续加筛选条件缩小范围</span>
          )}
        </div>
      </div>
    </div>
  )
}

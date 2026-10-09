// 反向求解的三个代表结果：最佳 / 边界 / 失效。
//
// 直接画出三个翼型轮廓，并列出对应的评价指标。数据取自 results.json 的 cases 里
// tag="ga" 的前三个案例（与论文第 7 章「五个案例」中的前三例一致）：
//   最佳  案例 1：八度升力最高且约束全部满足
//   边界  案例 2：分离约束正好贴边（binding = sep）
//   失效  案例 3：局部马赫与分离都超限，判定不可行
//
// 指标口径与页面上其它地方一致：cl4/cl8/cl10 为 4/8/10 度升力，
// delta 为带内损失（固定 4/8/10 口径），上限取 results.objective 里的值。

import { useEffect, useRef } from 'react'
import type { CaseItem } from '../data/mock.ts'
import { nacaOutline } from '../lib/naca.ts'
import { colors, setupCanvas } from '../lib/plot.ts'

interface Props {
  cases: CaseItem[]
  machLimit: number
  sepLimit: number
  /** 升力参照：数据库同工况最优真实翼型（论文里的 1.1464）。 */
  dbBest: number
  /** 带内损失参照：数据库带内最优。 */
  deltaLimit: number
}

interface Slot {
  key: string
  name: string
  tone: 'ok' | 'warn' | 'bad'
  desc: string
  c: CaseItem
}

/** 按效果从 cases 里挑出最佳 / 边界 / 失效三个。 */
function pickSlots(cases: CaseItem[]): Slot[] {
  const ga = cases.filter((c) => (c as CaseItem & { tag?: string }).tag === 'ga')
  const pool = ga.length >= 3 ? ga : cases
  const feasible = pool.filter((c) => c.feasible)
  const best = [...feasible].sort((a, b) => b.cl8 - a.cl8)[0]
  // 边界：可行但最贴近约束的那个（binding 标了 sep，或分离代理最接近上限）
  const others = feasible.filter((c) => c !== best)
  const boundary = [...others].sort((a, b) => b.sepMax - a.sepMax)[0]
  // 失效：不可行的里升力最高的那个
  const failed = [...pool.filter((c) => !c.feasible)].sort((a, b) => b.cl8 - a.cl8)[0]
  const out: Slot[] = []
  if (best) {
    out.push({
      key: 'best', name: '最佳', tone: 'ok',
      desc: '八度升力最高，三项约束全部满足', c: best,
    })
  }
  if (boundary) {
    out.push({
      key: 'boundary', name: '边界', tone: 'warn',
      desc: '可行但分离约束正好贴边，余量最小', c: boundary,
    })
  }
  if (failed) {
    out.push({
      key: 'failed', name: '失效', tone: 'bad',
      desc: '约束超限，判定为不可行', c: failed,
    })
  }
  return out
}

const TONE_COLOR: Record<Slot['tone'], string> = {
  ok: '#34d399',
  warn: '#fb923c',
  bad: '#f87171',
}

/** 单个翼型轮廓。 */
function Outline({ params, color }: { params: [number, number, number, number], color: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const box = setupCanvas(el, 96)
      if (!box) return
      const { ctx, w, h } = box
      const { pts } = nacaOutline(params)
      let yMax = 0.14
      for (const [, py] of pts) yMax = Math.max(yMax, Math.abs(py) * 1.2)
      const pad = { l: 14, r: 14, t: 10, b: 10 }
      const iw = w - pad.l - pad.r
      const ih = h - pad.t - pad.b
      const sc = Math.min(iw / 1.06, ih / (2 * yMax))
      const cx = pad.l + iw / 2
      const cy = pad.t + ih / 2
      const sx = (x: number) => cx + (x - 0.5) * sc
      const sy = (y: number) => cy - y * sc
      ctx.beginPath()
      pts.forEach(([px, py], i) => {
        if (i === 0) ctx.moveTo(sx(px), sy(py))
        else ctx.lineTo(sx(px), sy(py))
      })
      ctx.closePath()
      ctx.lineWidth = 1.8
      ctx.strokeStyle = color
      ctx.shadowColor = color
      ctx.shadowBlur = 6
      ctx.stroke()
      ctx.shadowBlur = 0
      // 弦线
      const C = colors()
      ctx.beginPath()
      ctx.setLineDash([3, 3])
      ctx.moveTo(sx(0), sy(0))
      ctx.lineTo(sx(1), sy(0))
      ctx.strokeStyle = C.grid
      ctx.lineWidth = 1
      ctx.stroke()
      ctx.setLineDash([])
    }
    draw()
    const ro = new ResizeObserver(draw)
    if (ref.current?.parentElement) ro.observe(ref.current.parentElement)
    return () => ro.disconnect()
  }, [params, color])
  return <canvas ref={ref} />
}

export default function InverseCases({ cases, machLimit, sepLimit, dbBest, deltaLimit }: Props) {
  const slots = pickSlots(cases)
  if (slots.length === 0) return null

  return (
    <div className="panel" style={{ marginTop: 'var(--sp-4)' }}>
      <div className="panel-title">
        <span>三个有代表性的反向求解结果</span>
        <span className="muted">对应论文第 7 章五个案例中的前三例</span>
      </div>
      <div className="panel-body" style={{ display: 'grid', gap: 'var(--sp-3)',
        gridTemplateColumns: `repeat(${slots.length}, minmax(0, 1fr))` }}>
        {slots.map((s) => {
          const col = TONE_COLOR[s.tone]
          // 判定参照与论文 §6.3/§6.4 一致：
          //   硬约束（打勾/叉）：
          //     局部马赫 <= 0.6（工程上限）
          //     分离代理 <  0.007874（真实样本九十分位；等于阈值即「顶线」，按不合格算）
          //   优化目标（不打勾叉，只与参照比较）：
          //     升力，越大越好，参照 = 数据库同工况最优真实翼型 1.1464。
          //       注意这是「目前真实设计能达到的最好水平」，不是必须越过的门槛；
          //       达不到只说明还没超过现有真实设计，不等于这个解不合格。
          //     带内损失，越小越好，参照 = 数据库带内最优 0.367。
          const EPS = 1e-9
          const rows: { k: string; v: string; ok?: boolean; best?: boolean; ref?: string }[] = [
            { k: '八度升力 CL8', v: s.c.cl8.toFixed(4),
              best: s.c.cl8 >= dbBest - EPS,
              ref: `数据库最优 ${dbBest}` },
            { k: '四度升力 CL4', v: s.c.cl4.toFixed(4) },
            { k: '十度升力 CL10', v: s.c.cl10.toFixed(4) },
            // 带内损失是优化目标，不是硬约束：只与参照比较，不给勾叉
            { k: '带内损失 Δ', v: s.c.delta.toFixed(4), best: s.c.delta <= deltaLimit + EPS,
              ref: `参照 ${deltaLimit}` },
            { k: '最大马赫代理', v: s.c.machMax.toFixed(4), ok: s.c.machMax <= machLimit + EPS,
              ref: `上限 ${machLimit}` },
            { k: '分离代理', v: s.c.sepMax.toFixed(6),
              ok: s.c.sepMax < sepLimit - EPS,
              best: s.c.sepMax < sepLimit - EPS,
              ref: `上限 ${sepLimit}` },
          ]
          return (
            <div key={s.key} className="inv-case" style={{ borderColor: col }}>
              <div className="inv-case-head">
                <span className="inv-case-tag" style={{ background: col }}>{s.name}</span>
                <span className="inv-case-feasible" style={{ color: s.c.feasible ? 'var(--ok)' : 'var(--bad)' }}>
                  {s.c.feasible ? '可行' : '不可行'}
                </span>
              </div>
              <div className="inv-case-desc">{s.desc}</div>
              <Outline params={[s.c.L, s.c.P, s.c.Q, s.c.T]} color={col} />
              <div className="inv-case-params">
                L {s.c.L.toFixed(3)} · P {s.c.P.toFixed(3)} · Q {s.c.Q} · T {s.c.T.toFixed(3)}
              </div>
              <div className="inv-case-rows">
                {rows.map((r) => (
                  <div key={r.k} className="inv-case-row">
                    <span className="inv-case-k">{r.k}</span>
                    <span className="inv-case-v">
                      <span className="num" style={{
                        color: r.ok === undefined
                          ? (r.best ? 'var(--acc-strong)' : 'var(--fg-1)')
                          : (r.ok ? 'var(--ok)' : 'var(--bad)'),
                      }}>
                        {r.v}
                      </span>
                      {r.ok !== undefined ? (
                        <span className="inv-case-mark" style={{ color: r.ok ? 'var(--ok)' : 'var(--bad)' }}>
                          {r.ok ? '✓' : '✗'}
                        </span>
                      ) : (
                        r.best ? <span className="inv-case-mark" style={{ color: 'var(--acc-strong)' }}>★</span> : null
                      )}
                    </span>
                    {r.ref && <span className="inv-case-ref">{r.ref}</span>}
                  </div>
                ))}
              </div>
              {s.c.reason && <div className="inv-case-reason">{s.c.reason}</div>}
            </div>
          )
        })}
      </div>
      <div className="panel-body muted" style={{ fontSize: 12.5 }}>
        <b>两类指标要分开看。</b>
        <b>带勾叉的是硬约束</b>：局部马赫不超过 {machLimit}；分离代理须小于真实样本第 90 百分位 {sepLimit}
        （<b>达到即为顶线，按不合格标记</b>）。
        <b>带参照值的是优化目标，不是门槛</b>：八度升力的参照 {dbBest} 是<b>数据库里同工况最好的真实翼型</b>，
        超过它说明已经优于现有真实设计（给星号），没超过也只说明还没赶上，<b>不等于这个解不合格</b>；
        带内损失同理，参照 {deltaLimit} 是数据库带内最优，越小说明对攻角偏移越不敏感。
        指标由本作品的降维代理模型给出，与页面上方推荐结果同一套口径。
      </div>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import './pages.css'
import type { DataState } from '../data/load.ts'
import { predictCp } from '../lib/models.ts'
import { subscribeSettings } from '../lib/settings.ts'
import {
  cannedReply,
  chatReplyOnline,
  clAtAoa,
  classifyIntentLocal,
  classifyIntentOnline,
  onlineModeActive,
  parseGoalLocal,
  parseGoalOnline,
  searchCandidates,
  type Candidate,
  type DesignGoal,
  type Intent,
} from '../lib/design.ts'
import AirfoilShape from '../components/AirfoilShape.tsx'
import CpCurve from '../components/CpCurve.tsx'
import InverseCases from '../components/InverseCases.tsx'

const base = import.meta.env.BASE_URL || './'

interface Msg {
  role: 'user' | 'assistant'
  text: string
  time: string
}

type StepState = 'wait' | 'run' | 'done' | 'fail'

interface Steps {
  parse: StepState
  search: StepState
  confirm: StepState
}

const now = () =>
  new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

function StepBlock({
  title, state, children,
}: {
  title: string
  state: StepState
  children?: React.ReactNode
}) {
  const color =
    state === 'done' ? 'var(--ok)' : state === 'fail' ? 'var(--bad)' : state === 'run' ? 'var(--acc)' : 'var(--pending)'
  const label =
    state === 'done' ? '完成' : state === 'fail' ? '失败' : state === 'run' ? '进行中' : '等待'
  return (
    <div className="panel" style={{ paddingBottom: 8 }}>
      <div className="panel-title">
        <span>{title}</span>
        <span className="badge" style={{ borderColor: color, color }}>
          {label}
        </span>
      </div>
      {children && <div className="panel-body" style={{ fontSize: 13 }}>{children}</div>}
    </div>
  )
}

function ConstraintRow({ label, value, limit, unit, invert, mode }: {
  label: string
  value: number
  limit: number
  unit: string
  /** true 表示“越大越好”（如升力），判定与配色反过来。 */
  invert?: boolean
  /**
   * 'gate'（默认）＝硬约束，给 达标/未达 判定；
   * 'ref'＝优化目标，只与参照值比较，不给达标判定
   * （避免把「最好的真实样本」这种参照当成必须达到的门槛）。
   */
  mode?: 'gate' | 'ref'
}) {
  const good = invert ? value >= limit : value <= limit
  if (mode === 'ref') {
    return (
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
        <span>{label}</span>
        <span style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
          <span className="num">{value.toFixed(4)}</span>
          <span className="muted" style={{ fontSize: 12 }}>参照 {limit}</span>
          <span className={good ? 'badge real' : 'badge mock'}
            style={good ? {} : { borderColor: 'var(--fg-2)', color: 'var(--fg-2)' }}>
            {good ? '优于参照' : '未及参照'}
          </span>
        </span>
      </div>
    )
  }
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
      <span>{label}</span>
      <span style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
        <span className="num">{value.toFixed(4)} / {invert ? '≥' : '≤'} {limit} {unit}</span>
        <span className={good ? 'badge real' : 'badge mock'} style={good ? {} : { borderColor: 'var(--bad)', color: 'var(--bad)', background: 'var(--bad-dim)' }}>
          {good ? '✓ 达标' : '✗ 未达'}
        </span>
      </span>
    </div>
  )
}

export default function Inverse({ data }: { data: DataState }) {
  const [msgs, setMsgs] = useState<Msg[]>([
    {
      role: 'assistant',
      text: '用一句话说出你想要的翼型，例如"我要一个八度附近升力高而且十度别掉的翼型"。' +
        '我会解析需求、在候选里检索、出形状和压力曲线确认。',
      time: now(),
    },
  ])
  const [input, setInput] = useState('')
  const [steps, setSteps] = useState<Steps>({ parse: 'wait', search: 'wait', confirm: 'wait' })
  const [goal, setGoal] = useState<DesignGoal | null>(null)
  const [cands, setCands] = useState<Candidate[]>([])
  const [picked, setPicked] = useState<Candidate | null>(null)
  const [confirmCp, setConfirmCp] = useState<number[] | null>(null)
  const [online, setOnline] = useState(onlineModeActive())
  const [busy, setBusy] = useState(false)
  const chatRef = useRef<HTMLDivElement>(null)

  useEffect(() => subscribeSettings(() => setOnline(onlineModeActive())), [])
  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' })
  }, [msgs])

  const obj = data.results.objective

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setBusy(true)
    setMsgs((m) => [...m, { role: 'user', text, time: now() }])
    setSteps({ parse: 'run', search: 'wait', confirm: 'wait' })
    setGoal(null)
    setCands([])
    setPicked(null)
    setConfirmCp(null)

    // 第一步不再是“直接解目标”，而是先分意图。
    // 只有确实是设计需求才进检索链路；问候/提问/其它只回一句话。
    // 这样避免“一句你好也回车一个方案”的僵硬感。
    let intent: Intent = 'other'
    try {
      intent = online ? await classifyIntentOnline(text) : classifyIntentLocal(text)
    } catch (e) {
      intent = classifyIntentLocal(text)
      setMsgs((m) => [...m, {
        role: 'assistant',
        text: '意图分类调用失败（' + (e instanceof Error ? e.message : String(e)) +
          '），已改用本地关键词判断。',
        time: now(),
      }])
    }

    if (intent !== 'design') {
      setSteps({ parse: 'wait', search: 'wait', confirm: 'wait' })
      let reply = cannedReply(intent)
      if (online) {
        try {
          reply = await chatReplyOnline(text)
        } catch {
          // 在线闲聊失败就用手写兜底，不再弹错误干扰用户
        }
      }
      setMsgs((m) => [...m, { role: 'assistant', text: reply, time: now() }])
      setBusy(false)
      return
    }

    let g: DesignGoal
    try {
      g = online ? await parseGoalOnline(text) : parseGoalLocal(text)
    } catch (e) {
      g = parseGoalLocal(text)
      setMsgs((m) => [...m, {
        role: 'assistant',
        text: '在线解析失败（' + (e instanceof Error ? e.message : String(e)) + '），已退回本地规则。',
        time: now(),
      }])
    }
    setGoal(g)
    setSteps({ parse: 'done', search: 'run', confirm: 'wait' })
    await new Promise((r) => setTimeout(r, 350))

    const cs = searchCandidates(g, data.results)
    setCands(cs)
    setSteps({ parse: 'done', search: 'done', confirm: 'run' })
    await new Promise((r) => setTimeout(r, 350))

    if (cs.length > 0) {
      const best = cs[0]
      setPicked(best)
      const r = await predictCp([best.L, best.P, best.Q, best.T], obj.uInf, g.targetAoa, data.cpBasis, base)
      setConfirmCp(r.cp)
      setSteps({ parse: 'done', search: 'done', confirm: 'done' })
      const clT = clAtAoa(best, g.targetAoa, obj.aoaGrid ?? [])
      setMsgs((m) => [...m, {
        role: 'assistant',
        text: `推荐 L=${best.L.toFixed(2)}、P=${best.P.toFixed(2)}、Q=${best.Q}、T=${best.T.toFixed(2)}，` +
          `${g.targetAoa.toFixed(0)} 度升力约 ${clT.toFixed(3)}` +
          `（八度 ${best.cl8.toFixed(3)}，数据库同工况最优为 ${obj.dbDesignBest}），` +
          `带内损失 ${best.delta.toFixed(3)}（参照 ${0.367}，越小越好）。` +
          (best.feasible
            ? '两条硬约束（局部马赫与分离代理）都满足。'
            : '注意有硬约束不满足，可以换备选看看。'),
        time: now(),
      }])
    } else {
      setSteps({ parse: 'done', search: 'done', confirm: 'fail' })
      setMsgs((m) => [...m, { role: 'assistant', text: '没有找到候选，换个说法试试。', time: now() }])
    }
    setBusy(false)
  }

  const pick = async (c: Candidate) => {
    setPicked(c)
    setConfirmCp(null)
    const r = await predictCp([c.L, c.P, c.Q, c.T], obj.uInf, goal?.targetAoa ?? 8, data.cpBasis, base)
    setConfirmCp(r.cp)
  }

  return (
    <section className="page page-in">
      <div className="page-head">
        <h2>反向设计与对话</h2>
        <span className="hint">
          说一句话就能得到一版设计。目标是八度攻角附近升力尽量高，四度和十度不要掉太多。
        </span>
        <span className={online ? 'badge online' : 'badge mock'}>
          {online ? '在线模式' : '本地规则模式'}
        </span>
        {data.mode === 'mock' && <span className="badge mock">示例数据</span>}
      </div>

      <div className="cols-2" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1.2fr)' }}>
        <div className="panel" style={{ display: 'grid', gridTemplateRows: 'auto 1fr auto', minHeight: 520 }}>
          <div className="panel-title">对话</div>
          <div ref={chatRef} style={{ overflowY: 'auto', maxHeight: 430, padding: '0 16px', display: 'grid', gap: 10, alignContent: 'start' }}>
            {msgs.map((m, i) => (
              <div key={i} style={{
                justifySelf: m.role === 'user' ? 'end' : 'start',
                maxWidth: '86%',
                background: m.role === 'user' ? 'var(--acc-dim)' : 'var(--bg-3)',
                border: `1px solid ${m.role === 'user' ? 'var(--acc-line)' : 'var(--line)'}`,
                borderRadius: 10,
                padding: '8px 12px',
              }}>
                <div style={{ fontSize: 13, lineHeight: 1.55, color: 'var(--fg-0)' }}>{m.text}</div>
                <div className="num" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 4, textAlign: 'right' }}>{m.time}</div>
              </div>
            ))}
            {busy && <div className="muted">正在处理…</div>}
          </div>
          <div style={{ padding: 12, borderTop: '1px solid var(--line)', display: 'flex', gap: 8 }}>
            <input
              type="text"
              style={{ flex: 1 }}
              placeholder="例如：我要一个八度附近升力高而且十度别掉的翼型"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void send() }}
            />
            <button onClick={() => void send()} disabled={busy || !input.trim()}
              style={{ background: 'var(--acc-dim)', borderColor: 'var(--acc-line)', color: 'var(--acc-strong)' }}>
              发送
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 'var(--sp-4)', alignContent: 'start' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 'var(--sp-3)' }}>
            <StepBlock title="第一步 解析需求" state={steps.parse}>
              {goal && (
                <>
                  目标攻角 <span className="num">{goal.targetAoa.toFixed(1)}°</span>，
                  关注带宽 <span className="num">{goal.bandAoas.map((a) => a.toFixed(1) + '°').join(' ')}</span>
                  {goal.preferHighCl ? '，升力优先' : ''}
                </>
              )}
            </StepBlock>
            <StepBlock title="第二步 检索候选" state={steps.search}>
              {cands.length > 0 && <>从 {data.results.fronts.length} 个前沿成员里挑出 {cands.length} 个候选</>}
            </StepBlock>
            <StepBlock title="第三步 形状与流场确认" state={steps.confirm}>
              {picked && confirmCp && <>已用模型复核压力曲线</>}
            </StepBlock>
          </div>

          {picked && (
            <>
              <div className="panel">
                <div className="panel-title">
                  <span>推荐参数与约束</span>
                  <span className="muted">{picked.source}</span>
                </div>
                <div className="panel-body" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span className="badge online">L 弯度系数 {picked.L.toFixed(3)}</span>
                  <span className="badge online">P 弯度位置 {picked.P.toFixed(3)}</span>
                  <span className="badge online">Q 弯型 {picked.Q === 1 ? '反弯' : '常规'}</span>
                  <span className="badge online">T 厚度系数 {picked.T.toFixed(3)}</span>
                </div>
                <div className="panel-body" style={{ display: 'grid', gap: 6 }}>
                  {/* 升力与带内损失是优化目标，不是硬门槛：只与参照值比较，
                      不给「达标/未达」。参照分别是数据库同工况最优真实翼型，
                      与数据库带内最优。 */}
                  <ConstraintRow
                    label={goal ? `目标攻角 ${goal.targetAoa.toFixed(1)}° 处的升力（越大越好）` : '升力（越大越好）'}
                    value={goal ? clAtAoa(picked, goal.targetAoa, obj.aoaGrid ?? []) : picked.cl8}
                    limit={obj.dbDesignBest} unit="" invert mode="ref" />
                  <ConstraintRow label="带内损失 Δ（固定 4/8/10 口径，越小越好）" value={picked.delta} limit={0.367} unit="" mode="ref" />
                  <ConstraintRow label="最大马赫数代理（4/8/10 度取最差）" value={picked.machMax} limit={obj.machLimit} unit="" />
                  <ConstraintRow label="分离代理（4/8/10 度取最差）" value={picked.sepMax} limit={obj.sepLimit} unit="" />
                </div>
              </div>
              <div className="cols-2">
                <div className="panel">
                  <div className="panel-title">推荐翼型形状</div>
                  <div className="panel-body">
                    <AirfoilShape params={[picked.L, picked.P, picked.Q, picked.T]} height={190} />
                  </div>
                </div>
                <div className="panel">
                  <div className="panel-title">压力曲线复核</div>
                  <div className="panel-body">
                    {confirmCp ? <CpCurve cp={confirmCp} height={190} /> : <div className="skeleton" style={{ height: 190 }} />}
                  </div>
                </div>
              </div>
              <div className="panel">
                <div className="panel-title">备选方案（点击切换）</div>
                <div className="panel-body" style={{ display: 'grid', gap: 4 }}>
                  {cands.map((c, i) => (
                    <button key={i} onClick={() => void pick(c)}
                      style={{
                        display: 'flex', justifyContent: 'space-between', fontSize: 12, textAlign: 'left',
                        ...(picked === c ? { background: 'var(--acc-dim)', borderColor: 'var(--acc-line)', color: 'var(--acc-strong)' } : {}),
                      }}>
                      <span className="num">L={c.L.toFixed(2)} P={c.P.toFixed(2)} Q={c.Q} T={c.T.toFixed(2)}</span>
                      <span className="num">
                        {goal ? `${goal.targetAoa.toFixed(0)}°升力=${clAtAoa(c, goal.targetAoa, obj.aoaGrid ?? []).toFixed(3)} ` : ''}
                        Δ={c.delta.toFixed(3)} {c.feasible ? '可行' : '不可行'}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
          {!picked && steps.confirm === 'wait' && (
            <div className="panel">
              <div className="panel-body muted" style={{ paddingTop: 12 }}>
                在左边对话框发一句设计需求，右侧会依次出现解析、检索、确认三步状态和推荐结果。
                没填接口信息时自动使用本地规则模式，不联网也能演示。
              </div>
            </div>
          )}
        </div>
      </div>

      <InverseCases
        cases={data.results.cases}
        machLimit={obj.machLimit}
        sepLimit={obj.sepLimit}
        dbBest={obj.dbDesignBest}
        deltaLimit={0.367}
      />
    </section>
  )
}

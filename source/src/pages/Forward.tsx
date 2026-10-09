import { useEffect, useMemo, useRef, useState } from 'react'
import './pages.css'
import type { DataState } from '../data/load.ts'
import { analyticField, type FieldName } from '../data/mock.ts'
import { predictCp, modelState } from '../lib/models.ts'
import type { NacaParams } from '../lib/naca.ts'
import { nacaOutline } from '../lib/naca.ts'
import { evaluateObjectives, type ObjectiveEval } from '../lib/objectives.ts'
import { predictField } from '../lib/fieldmodel.ts'
import AirfoilShape from '../components/AirfoilShape.tsx'
import CpCurve from '../components/CpCurve.tsx'
import FlowField from '../components/FlowField.tsx'

const base = import.meta.env.BASE_URL || './'

interface SliderSpec {
  key: string
  label: string
  min: number
  max: number
  step: number
  note: string
}

const SLIDERS_5: SliderSpec[] = [
  { key: 'L', label: '弯度系数 L', min: 0.004, max: 3.993, step: 0.001, note: '越大升力越高，巡航阻力也越大' },
  { key: 'P', label: '弯度位置 P', min: 3.005, max: 7.969, step: 0.001, note: '越大最大弯度越靠后，压心后移' },
  { key: 'T', label: '厚度系数 T', min: 5.008, max: 19.989, step: 0.001, note: '越大翼型越厚，结构空间大但阻力上升' },
]
const SLIDERS_4: SliderSpec[] = [
  { key: 'M', label: '弯度百分数 M', min: 0, max: 9, step: 0.1, note: '最大弯度占弦长的百分数，越大升力越高' },
  { key: 'P', label: '弯度位置 P', min: 1, max: 9, step: 0.1, note: '最大弯度位置占弦长的十分数' },
  { key: 'T', label: '厚度百分数 T', min: 6, max: 21, step: 0.1, note: '厚度占弦长的百分数' },
]

const FIELD_META: Record<FieldName, { label: string; unit: string; caption: string }> = {
  u: {
    label: '流向速度 u', unit: 'm/s',
    caption: '流向速度，沿来流方向的速度分量，单位米每秒。翼型上方气流加速，'
      + '下方减速，尾迹里出现速度亏损。',
  },
  v: {
    label: '横向速度 v', unit: 'm/s',
    caption: '横向速度，垂直于来流方向的速度分量，单位米每秒。绕流时气流被抬起或压下，'
      + '所以翼型上下两侧符号相反。',
  },
  p: {
    label: '压力 p', unit: 'm²/s²',
    caption: '压力，这里用的是运动学压力，即压强除以密度，单位平方米每秒平方。'
      + '翼型上方的负值区就是吸力区，是升力的主要来源。',
  },
  nu_t: {
    label: '涡粘 nu_t', unit: 'm²/s',
    caption: '涡黏系数，湍流模型里表示湍流掺混强度的量，单位平方米每秒。'
      + '它在近壁与尾迹里最大，也是四个场里最难预测的一个。',
  },
}

function fmt(v: number, n = 3): string {
  return v.toFixed(n)
}

// 四个指标的口径集中在 lib/objectives.ts 与 lib/proxies.ts，
// 与反向设计阶段用同一套定义，这里不再自己造近似公式。
// 早先版本在本文件里自拟过马赫代理与分离代理的换算，已作废。

interface Metric {
  label: string
  value: number | null
  limit: number
  lessIsBad: boolean
  unit: string
  source: string
}

function MetricCard({ m }: { m: Metric }) {
  const over = m.value !== null && (m.lessIsBad ? m.value > m.limit : m.value < m.limit)
  const ratio =
    m.value === null ? 0 : Math.min(1.25, Math.abs(m.value) / (Math.abs(m.limit) || 1))
  return (
    <div className={over ? 'card bad' : 'card ok'}>
      <span className="card-label">{m.label}</span>
      <span className="card-val num">
        {m.value === null ? '待接入' : m.value.toFixed(4)}
      </span>
      <div className="limit-bar">
        <div className="limit-fill" style={{ width: `${Math.min(100, ratio * 80)}%` }} />
      </div>
      <span className="card-note">
        约束 {m.lessIsBad ? '≤' : '≥'} {m.limit} {m.unit}
        {over && m.value !== null
          ? `，超出 ${(Math.abs(m.value - m.limit)).toFixed(4)}`
          : ''}
      </span>
      <span className="card-note">数值来源：{m.source}</span>
    </div>
  )
}

export default function Forward({ data }: { data: DataState }) {
  const [family, setFamily] = useState<'5digit' | '4digit'>('5digit')
  const [v5, setV5] = useState({ L: 3.67, P: 6.5, Q: 0, T: 13.5 })
  const [v4, setV4] = useState({ M: 2.0, P: 4.0, T: 12.0 })
  const [aoa, setAoa] = useState(8.0)
  const [uInf, setUInf] = useState(62.0)
  const [splitCp, setSplitCp] = useState(false)
  const [fieldName, setFieldName] = useState<FieldName>('p')
  const [cp, setCp] = useState<number[] | null>(null)
  const [usedModel, setUsedModel] = useState(false)
  const [flash, setFlash] = useState(0)
  const [objEval, setObjEval] = useState<ObjectiveEval | null>(null)
  const [field, setField] = useState<Float32Array | null>(null)
  const [fieldFallback, setFieldFallback] = useState('')
  const [modelReason, setModelReason] = useState('')

  // 探一次模型可用性，不可用时把原因显示出来，不默默降级。
  useEffect(() => {
    let alive = true
    modelState(base).then((s) => { if (alive) setModelReason(s.degradedReason) })
    return () => { alive = false }
  }, [])

  const params: NacaParams = useMemo(
    () => (family === '5digit' ? [v5.L, v5.P, v5.Q, v5.T] : [v4.M, v4.P, v4.T]),
    [family, v5, v4],
  )
  const outline = useMemo(() => nacaOutline(params).pts, [params])

  // 压力曲线预测：rAF 节流，参数变化后下一帧更新
  const reqRef = useRef(0)
  useEffect(() => {
    let alive = true
    cancelAnimationFrame(reqRef.current)
    reqRef.current = requestAnimationFrame(() => {
      predictCp(params, uInf, aoa, data.cpBasis, base).then((r) => {
        if (!alive) return
        setCp(r.cp)
        setUsedModel(r.usedModel)
        setFlash((f) => f + 1)
        // 推理失败时降级原因是在这一次调用里才写下的，所以这里再读一次
        modelState(base).then((s) => { if (alive) setModelReason(s.degradedReason) })
      })
    })
    return () => {
      alive = false
    }
  }, [params, uInf, aoa, data.cpBasis])

  // 四个设计指标：按设计带的四度、八度、十度三个攻角各算一次，
  // 来流速度固定取设计点，只跟形状参数走，与上面的攻角滑块无关。
  useEffect(() => {
    let alive = true
    evaluateObjectives(params, data.cpBasis, base)
      .then((r) => { if (alive) setObjEval(r) })
      .catch(() => { if (alive) setObjEval(null) })
    return () => { alive = false }
  }, [params, data.cpBasis])

  // 流场：优先用真实的 b_field.onnx 加场模态基还原，
  // 缺文件或推理失败时退回解析近似，并把原因写在图注里。
  useEffect(() => {
    let alive = true
    predictField(params, uInf, aoa, {
      fieldSmean: data.cpBasis.fieldSmean ?? data.cpBasis.smean,
      fieldSstd: data.cpBasis.fieldSstd ?? data.cpBasis.sstd,
      fieldGmean: data.cpBasis.fieldGmean ?? data.cpBasis.gmean,
      fieldGstd: data.cpBasis.fieldGstd ?? data.cpBasis.gstd,
    }, base, data.grid)
      .then((r) => {
        if (!alive) return
        setField(r.values[fieldName])
        setFieldFallback('')
      })
      .catch((e: unknown) => {
        if (!alive) return
        setField(analyticField(params, uInf, aoa, fieldName, data.grid))
        setFieldFallback(e instanceof Error ? e.message : String(e))
      })
    return () => { alive = false }
  }, [params, uInf, aoa, fieldName, data.grid, data.cpBasis])

  const obj = data.results.objective
  // 四个阈值的出处，与 results.json 的 objective 一致，逐条写在指标卡里。
  const metrics: Metric[] = useMemo(() => {
    const deltaRef = 0.367 // 数据库带内最优，与 objective 口径一致
    if (!objEval) {
      return [
        { label: '八度升力系数 CL8', value: null, limit: obj.dbDesignBest, lessIsBad: false, unit: '', source: '待接入' },
        { label: '带内损失 Δ', value: null, limit: deltaRef, lessIsBad: true, unit: '', source: '待接入' },
        { label: '最大马赫数代理', value: null, limit: obj.machLimit, lessIsBad: true, unit: '', source: '待接入' },
        { label: '分离代理', value: null, limit: obj.sepLimit, lessIsBad: true, unit: '', source: '待接入' },
      ]
    }
    const src = objEval.usedModel ? 'ONNX 三攻角' : '示例压力曲线'
    return [
      { label: '八度升力系数 CL8', value: objEval.cl8, limit: obj.dbDesignBest, lessIsBad: false, unit: '', source: src + '，压力项线积分。参照线 1.146 是官方数据库同口径最高的真实翼型，不是硬约束' },
      { label: '带内损失 Δ', value: objEval.delta, limit: deltaRef, lessIsBad: true, unit: '', source: src + '，四度与十度两个方向取大。它是优化目标不是硬约束，0.367 是数据库带内最优，搜索解落在 0.359 到 0.938 之间' },
      { label: '最大马赫数代理', value: objEval.machMax, limit: obj.machLimit, lessIsBad: true, unit: '', source: src + '，三攻角取最大。上限 0.6 是工程保守上限，超过后激波与阻力增长很快' },
      { label: '分离代理', value: objEval.sepMax, limit: obj.sepLimit, lessIsBad: true, unit: '', source: src + '，三攻角取最大。上限 0.007874 在 511 条五位族真实样本上标定，取九十分位，等于一个离散热度步长' },
    ]
  }, [objEval, obj])

  const sliderVal = (key: string) =>
    family === '5digit' ? v5[key as keyof typeof v5] : v4[key as keyof typeof v4]
  const setSlider = (key: string, val: number) => {
    if (family === '5digit') setV5((s) => ({ ...s, [key]: val }))
    else setV4((s) => ({ ...s, [key]: val }))
  }
  const sliders = family === '5digit' ? SLIDERS_5 : SLIDERS_4

  return (
    <section className="page page-in">
      <div className="page-head">
        <h2>前向预测</h2>
        <span className="hint">
          改一个形状参数，压力分布和流场马上跟着变。压力系数 Cp 是表面压力除以来流动压，
          负值表示吸力。
        </span>
        {data.mode === 'mock' && <span className="badge mock">示例数据</span>}
        {!usedModel && data.mode === 'real' && <span className="badge mock">示例模型</span>}
        {usedModel && <span className="badge real">ONNX 推理</span>}
      </div>

      <div className="pipeline">
        <div className="pipe-block">
          <span className="pipe-big num">14.0 GB</span>
          <span className="pipe-small">全量仿真数据</span>
          <span className="pipe-note">1000 条样本的网格与全场解，15033268712 字节（internal.vtu 加 aerofoil.vtp）</span>
        </div>
        <div className="pipe-arrow"><span>MOR 降维</span>→</div>
        <div className="pipe-block">
          <span className="pipe-big num">16.98 MB</span>
          <span className="pipe-small">模态基与坐标（压缩比约 885 比 1）</span>
          <span className="pipe-note">16984790 字节：场模态基 15.64 MiB，压力曲线基 37 KB，一千条样本的模态坐标 0.54 MB</span>
        </div>
        <div className="pipe-arrow"><span>代理模型训练</span>→</div>
        <div className="pipe-block">
          <span className="pipe-big num pipe-hero">2.55 MB</span>
          <span className="pipe-small">预测模型，纯浏览器可跑</span>
          <span className="pipe-note">两个 ONNX 合计 2552533 字节（2.43 MiB），全站资产 27.01 MB</span>
        </div>
      </div>

      {/* 单次计算速度对比。数字取自论文表 12：纯推理 1.91 秒 对 0.33 毫秒，快 5750 倍。 */}
      <p className="pipe-speed">
        单次计算：官方逐点图神经网络 <b>1.91 秒</b>，本方法 <b>0.33 毫秒</b>，
        <span className="pipe-speed-hero">快 5750 倍</span>；
        四个任务的训练总时长 <b>20.5 小时</b>降至 <b>107 秒</b>，快 <b>689 倍</b>。
      </p>

      <div className="fwd-grid">
        <div className="panel" style={{ gridColumn: '1', gridRow: '1 / span 2' }}>
          <div className="panel-title">形状参数</div>
          <div className="seg">
            <button className={family === '5digit' ? 'active' : ''} onClick={() => setFamily('5digit')}>
              五位族
            </button>
            <button className={family === '4digit' ? 'active' : ''} onClick={() => setFamily('4digit')}>
              四位族
            </button>
          </div>
          {sliders.map((s) => (
            <div className="ctl" key={s.key}>
              <div className="ctl-head">
                <span>{s.label}</span>
                <span className="ctl-val num" key={flash}>{fmt(sliderVal(s.key))}</span>
              </div>
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={sliderVal(s.key)}
                onChange={(e) => setSlider(s.key, Number(e.target.value))}
              />
              <span className="ctl-note">{s.note}</span>
            </div>
          ))}
          {family === '5digit' && (
            <div className="ctl">
              <div className="ctl-head">
                <span>弯型 Q</span>
                <span className="ctl-val">{v5.Q === 1 ? '反弯' : '常规'}</span>
              </div>
              <div className="seg" style={{ padding: 0 }}>
                <button className={v5.Q === 0 ? 'active' : ''} onClick={() => setV5((s) => ({ ...s, Q: 0 }))}>
                  常规弯度
                </button>
                <button className={v5.Q === 1 ? 'active' : ''} onClick={() => setV5((s) => ({ ...s, Q: 1 }))}>
                  反弯弯度
                </button>
              </div>
              <span className="ctl-note">反弯可降低俯仰力矩，层流区更长</span>
            </div>
          )}
          <div className="panel-title" style={{ marginTop: 8 }}>来流条件</div>
          <div className="ctl">
            <div className="ctl-head">
              <span>攻角（来流与弦线的夹角）</span>
              <span className="ctl-val num" key={flash}>{aoa.toFixed(2)}°</span>
            </div>
            <input type="range" min={-5} max={15} step={0.01} value={aoa}
              onChange={(e) => setAoa(Number(e.target.value))} />
            <span className="ctl-note">越大升力越高，过大引发分离</span>
          </div>
          <div className="ctl">
            <div className="ctl-head">
              <span>来流速度</span>
              <span className="ctl-val num" key={flash}>{uInf.toFixed(1)} m/s</span>
            </div>
            <input type="range" min={31.4} max={93.6} step={0.1} value={uInf}
              onChange={(e) => setUInf(Number(e.target.value))} />
            <span className="ctl-note">决定雷诺数，影响边界层状态</span>
          </div>
        </div>

        <div className="panel" style={{ gridColumn: '2', gridRow: '1' }}>
          <div className="panel-title">
            <span>翼型形状（按弦长归一化）</span>
            <span className="muted">弦线 · 弯度线</span>
          </div>
          <div className="panel-body">
            <AirfoilShape params={params} height={300} />
          </div>
          <div className="panel-body muted">
            轮廓由 NACA 四位/五位族公式生成，几何向量 128 维已在参数变化时缓存，
            供模型与形状视图共用。
          </div>
        </div>

        {/* 高度跟着左邻的翼型形状框撑满，下缘与它持平；图表用弹性高度占住中间，
            模型状态那行自然落到框底。 */}
        <div className="panel" style={{
          gridColumn: '3', gridRow: '1', alignSelf: 'stretch',
          display: 'flex', flexDirection: 'column',
        }}>
            <div className="panel-title">
              <span>压力系数曲线（闭环弧长序）</span>
              <button style={{ fontSize: 11, padding: '2px 8px' }} onClick={() => setSplitCp((v) => !v)}>
                {splitCp ? '看闭环曲线' : '拆成上下表面'}
              </button>
            </div>
            <div className="panel-body" style={{ flex: 1, minHeight: 0 }}>
              {cp ? (
                <CpCurve cp={cp} splitSurfaces={splitCp} height={330} />
              ) : (
                <div className="skeleton" style={{ height: 330 }} />
              )}
            </div>
            <div className="panel-body muted">
              模型状态：{modelReason || (usedModel ? '已启用，走 ONNX 推理' : '未启用，未记录原因')}
            </div>
          </div>
          <div className="cards-col" style={{ gridColumn: '4', gridRow: '1 / span 2' }}>
            {metrics.map((m) => (
              <MetricCard key={m.label} m={m} />
            ))}
          </div>

        {/* 流场云图横跨第二与第三列，把翼型形状下方那块空白也占满 */}
        <div className="panel" style={{ gridColumn: '2 / span 2', gridRow: '2' }}>
          <div className="panel-title">
            <span>流场云图（{FIELD_META[fieldName].label}）</span>
            <span style={{ display: 'inline-flex', gap: 4 }}>
              {(Object.keys(FIELD_META) as FieldName[]).map((f) => (
                <button
                  key={f}
                  style={{
                    fontSize: 11,
                    padding: '2px 8px',
                    ...(f === fieldName
                      ? { background: 'var(--acc-dim)', borderColor: 'var(--acc-line)', color: 'var(--acc-strong)' }
                      : {}),
                  }}
                  onClick={() => setFieldName(f)}
                >
                  {f}
                </button>
              ))}
            </span>
          </div>
          <div className="panel-body">
            {field ? (
              <FlowField
                grid={data.grid}
                field={field}
                unit={FIELD_META[fieldName].unit}
                outline={outline}
                height={382}
              />
            ) : (
              <div className="skeleton" style={{ height: 382 }} />
            )}
          </div>
          <div className="panel-body muted">
            {fieldFallback
              ? '流场模型不可用（' + fieldFallback + '），当前显示的是解析近似，只作示意。'
              : FIELD_META[fieldName].caption
                + ' 由 b_field.onnx 在浏览器内推理，再乘场模态基还原，单位与论文图一致。'}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">这个页面在算什么</div>
        <div className="panel-body" style={{ display: 'grid', gap: 10, fontSize: 14, lineHeight: 1.75 }}>
          <p style={{ margin: 0 }}>
            <b style={{ color: 'var(--fg-0)' }}>压力系数 Cp</b> 是表面静压除以来流动压得到的无量纲数，
            负值表示当地压力低于来流，也就是吸力。上表面前缘的大片负值区就是升力的主要来源，
            所以气动图惯例把纵轴反过来画，负值在上。
          </p>
          <p style={{ margin: 0 }}>
            <b style={{ color: 'var(--fg-0)' }}>四个指标</b>各自的意思是。
            八度升力系数是攻角八度时由压力分布积分出的升力，越大越好。
            带内损失是八度升力减四度升力与八度升力减十度升力中的大者，
            衡量偏离设计点后的性能掉落，越小越稳。
            最大马赫数代理估计翼面局部最快速度对应的马赫数，超过 0.6 容易出激波。
            分离代理估计边界层分离的严重程度，超过 0.007874 认为不可接受。
          </p>
          <p style={{ margin: 0 }}>
            <b style={{ color: 'var(--fg-0)' }}>前向预测的做法</b> 是把 133 维向量喂给模型。
            前 5 维是来流标量（速度、攻角、弯度、厚度、弯型），后 128 维是几何曲线。
            模型输出低维模态坐标，压力曲线 8 个、流场每个物理量 32 个，
            再乘模态基加均值还原成物理量。两个 ONNX 模型都在浏览器里跑，不依赖服务器。
          </p>
          <p style={{ margin: 0 }}>
            <b style={{ color: 'var(--fg-0)' }}>数字的可信边界</b> 要说明白。
            绝对升力值没有经过风洞或官方力系数验证，只能看相对排序与相对误差。
            形状还原不唯一，同样的压力分布可能对应多个翼型。
          </p>
        </div>
      </div>
    </section>
  )
}

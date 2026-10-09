// 反向设计的本地规则引擎与在线调用。
// 在线模式把用户的话发给 OpenAI 兼容接口；本地模式用关键词匹配加预计算结果。
// 密钥只在内存里，由 settings.ts 保管。

import { getSettings, isOnlineMode } from './settings.ts'
import type { FrontPoint, Results } from '../data/load.ts'

export interface DesignGoal {
  targetAoa: number
  bandAoas: number[]
  preferHighCl: boolean
  raw: string
}

/**
 * 意图分类结果。
 *
 * 为什么需要它：原来无论用户说什么，都会跑完「解析需求 -> 检索候选 -> 给推荐」
 * 整条链路。于是一句「你好」也会被硬解成设计目标并回车一个方案，
 * 看上去像预先编好的固定回复，这是真实的可用性问题。
 *
 * 现在先分类：只有 design 才进设计链路；其余只回一句话，不检索也不推荐。
 */
export type Intent = 'design' | 'greeting' | 'question' | 'other'

export interface Candidate {
  L: number
  P: number
  Q: number
  T: number
  cl4: number
  cl8: number
  cl10: number
  /** 0..14 度每 1 度的升力（15 个数）；由 b_cp.onnx 逐攻角实算得到。 */
  clByAoa?: number[]
  delta: number
  machMax: number
  sepMax: number
  feasible: boolean
  source: string
}

/** 本地规则：关键词解析用户意图。 */
export function parseGoalLocal(text: string): DesignGoal {
  const t = text.toLowerCase()
  const nums = [...t.matchAll(/(\d+(?:\.\d+)?)\s*(?:度|°)/g)].map((m) => Number(m[1]))
  let targetAoa = 8
  if (nums.length > 0) targetAoa = nums[0]
  const band = new Set<number>([targetAoa])
  for (const n of nums.slice(1)) band.add(n)
  if (t.includes('巡航') || t.includes('4度') || t.includes('4°')) band.add(4)
  if (t.includes('不要掉') || t.includes('别掉') || t.includes('10度') || t.includes('10°')) band.add(10)
  if (band.size === 1) {
    band.add(Math.max(0, targetAoa - 4))
    band.add(targetAoa + 2)
  }
  const preferHighCl =
    t.includes('升力') && (t.includes('高') || t.includes('大') || t.includes('尽量'))
  return {
    targetAoa,
    bandAoas: [...band].sort((a, b) => a - b),
    preferHighCl,
    raw: text,
  }
}

/** 在线模式：调用 OpenAI 兼容接口把一句话转成结构化目标。 */
export async function parseGoalOnline(text: string): Promise<DesignGoal> {
  const s = getSettings()
  // 归一化地址：去末尾斜杠，结尾不是 /chat/completions 就补上。
  // 这样用户填 https://api.agnes-ai.cn/v1 或填完整地址都能通。
  let url = s.endpoint.trim().replace(/\/+$/, '')
  if (!url.endsWith('/chat/completions')) url += '/chat/completions'
  const prompt =
    '把下面这句翼型设计需求转成 JSON，字段 targetAoa（数值）、bandAoas（数值数组）、' +
    'preferHighCl（布尔）。\n' +
    '重要：targetAoa 是【目标攻角，单位度】，取值范围 0 到 14。' +
    '句中提到的升力系数数值（如 1.2618、1.4225 这类带小数点的数）' +
    '【不是】攻角，不要把它当 targetAoa；那种情况应该从前面提到的角度（如“八度”）取，' +
    '并把它放进 bandAoas。若句中确实没提到角度，targetAoa 保留上一次的角度或默认 8。\n' +
    '只输出 JSON，不要解释。需求：' + text
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + s.apiKey,
    },
    body: JSON.stringify({
      model: s.model || 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    }),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error('接口返回 ' + res.status + '：' + body.slice(0, 200))
  }
  const j = await res.json()
  const content: string = j.choices?.[0]?.message?.content ?? ''
  const m = content.match(/\{[\s\S]*\}/)
  if (!m) throw new Error('接口没有返回 JSON')
  const parsed = JSON.parse(m[0])
  // 注意：不能用 `Number(x) || 8`——目标攻角 0 是合法值但 0 在 JS 里是假值，
  // 那样写会把用户明说的 0 度静默改成 8 度。只在“不是有限数”时兜底。
  const rawAoa = Number(parsed.targetAoa)
  let targetAoa = Number.isFinite(rawAoa) ? rawAoa : 8
  // 兜底校验（实测踩过）：用户说“让升力指标超出 1.4225 的阈值”时，
  // 模型会把 1.4225 当成目标攻角。攻角是有限范围内的度数，
  // 而本数据集的设计攻角只用到 0–14 度，因此超出范围的值一律不采信，
  // 改用本地关键词在原文里找“X 度”；再找不到就退回 8 度。
  if (!(targetAoa >= 0 && targetAoa <= 14)) {
    const local = parseGoalLocal(text)
    targetAoa = local.targetAoa
  }
  // 带宽：优先用模型给的；给空数组时回退到与本地规则同样的“目标±”构造，
  // 避免“目标 0 度、带宽却是空”这种自相矛盾的展示。
  let bandAoas: number[] = Array.isArray(parsed.bandAoas)
    ? parsed.bandAoas.map(Number).filter((n: number) => Number.isFinite(n) && n >= 0 && n <= 14)
    : []
  if (bandAoas.length === 0) {
    bandAoas = [Math.max(0, targetAoa - 4), targetAoa, targetAoa + 2]
  } else if (!bandAoas.includes(targetAoa)) {
    bandAoas = [targetAoa, ...bandAoas]
  }
  // 带宽只有一个点时补上邻角，否则“关注带宽”形同虚设（只有目标角）。
  if (bandAoas.length === 1) {
    bandAoas = [Math.max(0, targetAoa - 2), targetAoa, Math.min(14, targetAoa + 2)]
  }
  return {
    targetAoa,
    bandAoas: [...new Set(bandAoas)].sort((a, b) => a - b),
    preferHighCl: Boolean(parsed.preferHighCl),
    raw: text,
  }
}

/**
 * 把目标的攻角网格（0..14 度建每 1 度）线性插值取出指定攻角的升力。
 * 超出范围就夹到两端，不外推。
 */
export function clAtAoa(c: Candidate, aoa: number, grid: number[]): number {
  const ys = c.clByAoa
  if (!ys || ys.length !== grid.length) {
    // 没带逐攻角表时用 cl4/cl8/cl10 三点插值兼底（不应发生，仅作防护）
    const pts: Array<[number, number]> = [[4, c.cl4], [8, c.cl8], [10, c.cl10]]
    if (aoa <= pts[0][0]) return c.cl4
    if (aoa >= pts[2][0]) return c.cl10
    for (let i = 0; i + 1 < pts.length; i++) {
      const [x0, y0] = pts[i]
      const [x1, y1] = pts[i + 1]
      if (aoa >= x0 && aoa <= x1) return y0 + ((aoa - x0) / (x1 - x0)) * (y1 - y0)
    }
    return c.cl8
  }
  if (aoa <= grid[0]) return ys[0]
  const last = grid.length - 1
  if (aoa >= grid[last]) return ys[last]
  let i = 0
  while (i < last - 1 && grid[i + 1] < aoa) i++
  const t = (aoa - grid[i]) / (grid[i + 1] - grid[i])
  return ys[i] + t * (ys[i + 1] - ys[i])
}

/** 在线意图分类：把用户这句分到 design / greeting / question / other。 */
export async function classifyIntentOnline(text: string): Promise<Intent> {
  const s = getSettings()
  let url = s.endpoint.trim().replace(/\/+$/, '')
  if (!url.endsWith('/chat/completions')) url += '/chat/completions'
  const prompt =
    '判断下面这句话是不是一个翼型设计需求。只输出一个 JSON：' +
    '{"intent":"design"} 表示这是设计需求（例如指定了攻角、升力、厚度、巡航、失速等）；' +
    '{"intent":"greeting"} 表示问候或寒暄；' +
    '{"intent":"question"} 表示在问概念或提问但不是在提设计需求；' +
    '{"intent":"other"} 表示其它。只输出 JSON，不要解释。句子：' + text
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + s.apiKey,
    },
    body: JSON.stringify({
      model: s.model || 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    }),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error('接口返回 ' + res.status + '：' + body.slice(0, 200))
  }
  const j = await res.json()
  const content: string = j.choices?.[0]?.message?.content ?? ''
  const m = content.match(/\{[\s\S]*\}/)
  if (!m) throw new Error('接口没有返回 JSON')
  const got = String(JSON.parse(m[0]).intent || '').toLowerCase()
  if (got === 'design' || got === 'greeting' || got === 'question' || got === 'other') {
    return got as Intent
  }
  throw new Error('意图字段无法识别：' + got)
}

/** 在线闲聊回复：问候/提问时给一句得体的短回话。 */
export async function chatReplyOnline(text: string): Promise<string> {
  const s = getSettings()
  let url = s.endpoint.trim().replace(/\/+$/, '')
  if (!url.endsWith('/chat/completions')) url += '/chat/completions'
  const sys =
    '你是《数据驱动的翼型气动代理与反向设计》演示网站里的助手，该作品由 PCDLab 制作。' +
    '用中文，语气客气简短（两三句）。硬性限制：' +
    '不要透露或猜测底层模型名称、厂商或版本（被问到就只说你是本作品的翼型设计助手）；' +
    '不要编造任何气动数字；不要假装已经给出了设计结果；' +
    '只聊与翼型设计与本演示相关的内容，其它话题委婉拒绝。' +
    '回复末尾引导用户说出一句具体的设计需求，例如带目标攻角或升力期望。'
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + s.apiKey,
    },
    body: JSON.stringify({
      model: s.model || 'gpt-4o-mini',
      messages: [
        { role: 'system', content: sys },
        { role: 'user', content: text },
      ],
      temperature: 0.3,
    }),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error('接口返回 ' + res.status + '：' + body.slice(0, 200))
  }
  const j = await res.json()
  const content: string = j.choices?.[0]?.message?.content ?? ''
  if (!content.trim()) throw new Error('接口返回空内容')
  return content.trim()
}

/** 从前沿里检索候选，按**用户给的目标攻角**打分排序。
 *
 * 口径说明（口径 A）：
 *   - 排序用目标攻角处的升力；这使“我要 12 度”与“我要 4 度”给出不同推荐。
 *   - 带内损失 delta **仍用固定的 4/8/10 口径**（与报告里所有 delta 数字一致），
 *     不随用户输入变化；界面上会注明这一点。
 */
export function searchCandidates(goal: DesignGoal, results: Results): Candidate[] {
  const grid: number[] = results.objective.aoaGrid ?? []
  const pool: Candidate[] = results.fronts.map((f: FrontPoint) => ({
    L: f.L, P: f.P, Q: f.Q, T: f.T,
    cl4: f.cl4, cl8: f.cl8, cl10: f.cl10, clByAoa: f.clByAoa,
    delta: f.delta, machMax: f.machMax, sepMax: f.sepMax,
    feasible: f.feasible, source: `前沿 seed=${f.seed}`,
  }))
  const obj = results.objective
  const tAoa = goal.targetAoa
  const score = (c: Candidate) => {
    // 主项：目标攻角处的升力（越大越好）。
    // 次项：目标攻角附近的带宽里的最小升力，避免“只在单点高、一拍就跑”的解。
    const clTarget = clAtAoa(c, tAoa, grid)
    const band = goal.bandAoas.length ? goal.bandAoas : [tAoa]
    const bandMin = Math.min(...band.map((a) => clAtAoa(c, a, grid)))
    let s = goal.preferHighCl ? clTarget * 1.0 + bandMin * 0.5 : clTarget + bandMin * 0.25
    // 约束惩罚（与原来一致，口径不变）
    if (c.delta > 0.3) s -= (c.delta - 0.3) * 2
    if (c.machMax > obj.machLimit) s -= (c.machMax - obj.machLimit) * 3
    if (c.sepMax > obj.sepLimit) s -= (c.sepMax - obj.sepLimit) * 50
    if (!c.feasible) s -= 0.5
    return s
  }
  return pool.sort((a, b) => score(b) - score(a)).slice(0, 6)
}

export function onlineModeActive(): boolean {
  return isOnlineMode(getSettings())
}

/** 本地关键词分类（离线用；在线失败时也用它兼底）。 */
export function classifyIntentLocal(text: string): Intent {
  const t = text.toLowerCase().trim()
  if (!t) return 'other'
  const greet = ['你好', '您好', 'hi', 'hello', 'hey', '在吗', '在么', '早上好',
    '晚上好', '你是谁', '介绍一下你', '测试', 'test', '谢谢', 'thanks',
    'thank you', '哈哈']
  if (greet.some((k) => t === k || t.startsWith(k))) return 'greeting'
  const designKw = ['度', '°', '攻角', '迎角', '升力', '阻力', '翼型', '翼面',
    '气动', '设计', '弯度', '厚度', '巡航', '失速', '分离', '马赫', 'naca',
    '机翼', 'cl']
  const hasNumDeg = /\d+(?:\.\d+)?\s*(?:度|°)/.test(t)
  const hasDesignKw = designKw.some((k) => t.includes(k))
  if (hasNumDeg || hasDesignKw) {
    const askWords = ['什么是', '什么叫', '为什么', '怎么算', '是什么意思',
      '定义', '解释', '如何理解']
    if (askWords.some((k) => t.includes(k))) return 'question'
    return 'design'
  }
  const q = '?？吗什么怎么哪'
  for (const ch of q) if (t.includes(ch)) return 'question'
  return 'other'
}

/** 各意图的固定回话（不调模型也能给体面回复）。 */
export function cannedReply(intent: Intent): string {
  if (intent === 'greeting') {
    return '你好，我是这个翼型设计演示的助手。直接跟我说一句设计需求就行，比如：' +
      '我要一个八度附近升力高、十度别掉的翼型；或者：给我一个十二度大迎角不分离的翼型。' +
      '我会在 240 个候选里挑出合适的形状，并给出压力曲线和升力、带内损失、马赫、分离四项检查。'
  }
  if (intent === 'question') {
    return '这个问题我答不了——我是按设计需求检索候选的助手，不具备通用问答能力。' +
      '你直接说一个设计目标，例如「六度巡航、升力尽量高」，我就给出具体翼型与四项指标。'
  }
  return '我没读懂这句是不是一个设计需求。换个说法试试，带上目标攻角或期望，例如：' +
    '四度巡航的翼型，厚度不要太薄；或者：十二度大迎角不分离、升力高一点。'
}

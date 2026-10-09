// 模型加载与推理。真实 b_cp.onnx / b_field.onnx 存在时用 ONNX Runtime Web，
// 不存在或加载失败时退回解析近似（mock 层），并把降级原因交给界面显示。

import type { NacaParams } from './naca.ts'
import { fiveScalar, geomVector128 } from './naca.ts'
import type { CpBasis } from '../data/load.ts'
import { analyticCp } from '../data/mock.ts'
import { runExclusive } from './ortqueue.ts'

export interface ModelState {
  cpReady: boolean
  fieldReady: boolean
  /** 降级原因，空字符串表示无降级。 */
  degradedReason: string
}

type OrtSession = {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>>
}
type Ort = {
  InferenceSession: {
    create(url: string, opts?: Record<string, unknown>): Promise<OrtSession>
  }
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown
  env?: { wasm?: { numThreads?: number; simd?: boolean } }
}

let ortPromise: Promise<Ort | null> | null = null
async function getOrt(): Promise<Ort | null> {
  if (!ortPromise) {
    ortPromise = import('onnxruntime-web')
      .then((m) => {
        const ort = m as unknown as Ort
        // 单线程跑。多线程后端要 SharedArrayBuffer，静态服务器默认不发
        // COOP 与 COEP 响应头，浏览器不给，初始化就会失败并退化成示例模式。
        if (ort.env?.wasm) {
          ort.env.wasm.numThreads = 1
          ort.env.wasm.simd = true
        }
        return ort
      })
      .catch(() => null)
  }
  return ortPromise
}

let cpSession: OrtSession | null = null
let fieldSession: OrtSession | null = null
let sessionsPromise: Promise<void> | null = null
let degradedReason = ''

/**
 * 建立会话。这里缓存的是**进行中的 Promise**，不是一个布尔标志。
 * 早先用布尔标志时，第二个调用方会立刻返回，而那时第一个调用还没跑完，
 * 于是它读到的降级原因是空的，界面上就只显示"未记录原因"，把真正的错因吞掉了。
 */
function ensureSessions(base: string): Promise<void> {
  if (!sessionsPromise) sessionsPromise = createSessions(base)
  return sessionsPromise
}

async function createSessions(base: string): Promise<void> {
  const ort = await getOrt()
  if (!ort) {
    degradedReason = 'ONNX Runtime 加载失败'
    return
  }
  try {
    const res = await fetch(base + 'data/b_cp.onnx', { method: 'HEAD' })
    if (!res.ok) {
      degradedReason = '找不到压力曲线模型文件，请求状态 ' + res.status
    } else {
      cpSession = await ort.InferenceSession.create(base + 'data/b_cp.onnx', {
        executionProviders: ['wasm'],
      })
    }
  } catch (e) {
    degradedReason = '压力曲线模型加载失败：' + String(e)
  }
  try {
    const res = await fetch(base + 'data/b_field.onnx', { method: 'HEAD' })
    if (!res.ok) {
      degradedReason = (degradedReason ? degradedReason + '；' : '') +
        '找不到流场模型文件，请求状态 ' + res.status
    } else {
      fieldSession = await ort.InferenceSession.create(base + 'data/b_field.onnx', {
        executionProviders: ['wasm'],
      })
    }
  } catch (e) {
    degradedReason = (degradedReason ? degradedReason + '；' : '') +
      '流场模型加载失败：' + String(e)
  }
  if (!cpSession && !degradedReason) degradedReason = '压力曲线模型没有建立会话，原因未知'
  if (!fieldSession && !degradedReason) degradedReason = '流场模型没有建立会话，原因未知'
}

export async function modelState(base: string): Promise<ModelState> {
  await ensureSessions(base)
  return {
    cpReady: cpSession !== null,
    fieldReady: fieldSession !== null,
    degradedReason,
  }
}
function buildInput133(
  params: NacaParams,
  uInf: number,
  aoa: number,
  basis: CpBasis,
): Float32Array {
  const five = fiveScalar(params, uInf, aoa)
  const geom = geomVector128(params)
  const input = new Float32Array(133)
  for (let i = 0; i < 5; i++) {
    input[i] = (five[i] - basis.smean[i]) / basis.sstd[i]
  }
  for (let i = 0; i < 128; i++) {
    input[5 + i] = (geom[i] - basis.gmean[i]) / basis.gstd[i]
  }
  return input
}

/**
 * 从推理结果里取第一个输出张量。
 * onnxruntime-web 不同版本返回的可能是普通对象，也可能是 Map，
 * 早先只按普通对象取，遇到 Map 时会取到 undefined 然后抛错，
 * 表现就是"会话建好了但推理总是失败、退回示例曲线"。
 */
export function firstOutput(out: unknown): Float32Array {
  if (out instanceof Map) {
    const first = [...out.values()][0] as { data: Float32Array }
    return first.data
  }
  const rec = out as Record<string, { data: Float32Array }>
  const key = rec.alpha ? 'alpha' : Object.keys(rec)[0]
  return rec[key].data
}

/**
 * 前向预测压力曲线。
 * 有真实模型时走 ONNX，输出 8 个 POD 坐标，乘基加均值还原。
 * 没有模型或推理失败时退回解析近似曲线。
 */
export async function predictCp(
  params: NacaParams,
  uInf: number,
  aoa: number,
  basis: CpBasis,
  base: string,
): Promise<{ cp: number[]; usedModel: boolean }> {
  await ensureSessions(base)
  if (cpSession) {
    try {
      const ort = await getOrt()
      const input = buildInput133(params, uInf, aoa, basis)
      const tensor = new ort!.Tensor('float32', input, [1, 133])
      const out = await runExclusive(() => cpSession!.run({ x: tensor }))
      const alpha = firstOutput(out)
      const n = basis.nPts
      const cp = new Array<number>(n)
      for (let i = 0; i < n; i++) {
        let v = basis.mu[i]
        for (let k = 0; k < basis.podR; k++) v += alpha[k] * basis.phi[i][k]
        cp[i] = v
      }
      return { cp, usedModel: true }
    } catch (e) {
      degradedReason = '压力曲线推理失败，已降级：' + String(e)
    }
  }
  return { cp: analyticCp(params, uInf, aoa, basis.nPts), usedModel: false }
}

// 用真实的流场模型在浏览器里算流场。
//
// 口径来自 04_results/demo/01_DATA_CONTRACT.md 第 2.3 与 2.5 节：
//   输入是 133 维，前 5 维用来流标量做 z-score，后 128 维用几何曲线做 z-score；
//   模型输出 128 个模态坐标，按 u、v、p、nu_t 四场各 32 个排列；
//   每个场还原成 31066 个有效格上的值，再换算回物理单位；
//   物理值 = 归一化值 乘 stdOut 加 meanOut，系数在 field_basis.json 里。
//
// 这里自己开 onnxruntime 会话，与 lib/models.ts 的那一份相互独立。
// 两份会话会各自加载一次模型文件，代价是可以分别降级，页面不会因为一个模型坏了全黑。

import type { NacaParams } from './naca.ts'
import { fiveScalar, geomVector128 } from './naca.ts'
import type { Grid } from '../data/load.ts'
import { runExclusive } from './ortqueue.ts'

export type FieldName = 'u' | 'v' | 'p' | 'nu_t'

export const FIELD_ORDER: FieldName[] = ['u', 'v', 'p', 'nu_t']

export interface FieldBasisMeta {
  podR: number
  nGrid: number
  nFields: number
  order: FieldName[]
  phiOffset: number[]
  muOffset: number[]
  totalFloats: number
  meanOut: number[]
  stdOut: number[]
  unit: string[]
}

export interface FieldResult {
  /**
   * 每个场一个**整网格**数组，长度是 grid.nx 乘 grid.ny，下标 j*nx+i，
   * 掩膜外的格填 NaN。与 mock 的 analyticField 形状一致，画图那边直接按格点下标读。
   */
  values: Record<FieldName, Float32Array>
  usedModel: boolean
  reason: string
}

type OrtTensor = new (type: string, data: Float32Array, dims: number[]) => unknown
type OrtSession = { run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>> }
type Ort = {
  InferenceSession: { create(url: string, opts?: Record<string, unknown>): Promise<OrtSession> }
  Tensor: OrtTensor
  env?: { wasm?: { numThreads?: number; simd?: boolean } }
}

interface BasisBundle {
  meta: FieldBasisMeta
  data: Float32Array
}

let ortPromise: Promise<Ort | null> | null = null
let basisPromise: Promise<BasisBundle | null> | null = null
let sessionPromise: Promise<OrtSession | null> | null = null
let lastReason = ''

async function getOrt(): Promise<Ort | null> {
  if (!ortPromise) {
    ortPromise = import('onnxruntime-web')
      .then((m) => {
        const ort = m as unknown as Ort
        // 单线程跑。ONNX Runtime 的多线程后端要 SharedArrayBuffer，
        // 而静态服务器默认不发 COOP 与 COEP 响应头，浏览器就不给 SharedArrayBuffer，
        // 多线程初始化会失败，整个模型就退化成示例模式。我们的模型很小，单线程足够。
        if (ort.env?.wasm) {
          ort.env.wasm.numThreads = 1
          ort.env.wasm.simd = true
        }
        return ort
      })
      .catch(() => {
        lastReason = 'ONNX Runtime 加载失败'
        return null
      })
  }
  return ortPromise
}

async function getBasis(base: string): Promise<BasisBundle | null> {
  if (!basisPromise) {
    basisPromise = (async () => {
      try {
        const [metaRes, binRes] = await Promise.all([
          fetch(base + 'data/field_basis.json'),
          fetch(base + 'data/field_basis.bin'),
        ])
        if (!metaRes.ok || !binRes.ok) return null
        const meta = (await metaRes.json()) as FieldBasisMeta
        const buf = await binRes.arrayBuffer()
        const data = new Float32Array(buf)
        if (data.length !== meta.totalFloats) {
          lastReason = '场基长度与声明不符，声明 ' + meta.totalFloats + '，实际 ' + data.length
          return null
        }
        return { meta, data }
      } catch (e) {
        lastReason = '场基读取失败：' + String(e)
        return null
      }
    })()
  }
  return basisPromise
}

async function getSession(base: string): Promise<OrtSession | null> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await getOrt()
      if (!ort) return null
      try {
        const head = await fetch(base + 'data/b_field.onnx', { method: 'HEAD' })
        if (!head.ok) return null
        return await ort.InferenceSession.create(base + 'data/b_field.onnx', {
          executionProviders: ['wasm'],
        })
      } catch (e) {
        lastReason = '流场模型加载失败：' + String(e)
        return null
      }
    })()
  }
  return sessionPromise
}

export interface FieldModelStatus {
  ready: boolean
  reason: string
}

/** 探一下流场链路是否可用，界面据此显示真实模式还是近似模式。 */
export async function fieldModelStatus(base: string): Promise<FieldModelStatus> {
  const [basis, session] = await Promise.all([getBasis(base), getSession(base)])
  if (basis && session) return { ready: true, reason: '' }
  if (!basis) return { ready: false, reason: lastReason || '缺少场模态基' }
  return { ready: false, reason: lastReason || '缺少流场模型' }
}

/** 构造流场模型输入。统计量用 cp_basis.json 里的 field 那一组。 */
function buildInput133(
  params: NacaParams, uInf: number, aoa: number,
  fieldSmean: number[], fieldSstd: number[],
  fieldGmean: number[], fieldGstd: number[],
): Float32Array {
  const five = fiveScalar(params, uInf, aoa)
  const geom = geomVector128(params)
  const out = new Float32Array(133)
  for (let i = 0; i < 5; i++) out[i] = (five[i] - fieldSmean[i]) / fieldSstd[i]
  for (let i = 0; i < 128; i++) out[5 + i] = (geom[i] - fieldGmean[i]) / fieldGstd[i]
  return out
}

/**
 * 算四个场。给不出真实结果时抛错，让调用方决定退回近似还是显示待接入，
 * 不要在这里悄悄返回假数据。
 */
export async function predictField(
  params: NacaParams, uInf: number, aoa: number,
  stats: {
    fieldSmean: number[], fieldSstd: number[],
    fieldGmean: number[], fieldGstd: number[],
  },
  base: string,
  grid: Grid,
): Promise<FieldResult> {
  const [basis, session] = await Promise.all([getBasis(base), getSession(base)])
  if (!basis) throw new Error(lastReason || '场模态基不可用')
  if (!session) throw new Error(lastReason || '流场模型不可用')
  const ort = await getOrt()
  if (!ort) throw new Error(lastReason || 'ONNX Runtime 不可用')

  const { meta, data } = basis
  const input = buildInput133(params, uInf, aoa,
    stats.fieldSmean, stats.fieldSstd, stats.fieldGmean, stats.fieldGstd)
  const tensor = new ort.Tensor('float32', input, [1, 133])
  const out = await runExclusive(() => session.run({ x: tensor }))
  // onnxruntime-web 不同版本返回的可能是普通对象，也可能是 Map，两种都要认。
  const alpha = (out instanceof Map
    ? [...out.values()][0]
    : (out as Record<string, { data: Float32Array }>).alpha
      ?? Object.values(out as Record<string, { data: Float32Array }>)[0]).data
  if (alpha.length !== meta.nFields * meta.podR) {
    throw new Error('流场模型输出长度 ' + alpha.length + '，期望 ' + meta.nFields * meta.podR)
  }
  // 输出数组的形状必须与 mock 的 analyticField 一致：**整网格** nx*ny，
  // 下标是 j*nx+i，掩膜外的格填 NaN。
  // 这里踩过一个大坑：场模态基的格点顺序是"掩膜序"（第 n 个有效格），
  // 而画图那边是按整网格下标去读的。如果直接把掩膜序数组交给画图，
  // 整张流场图就会被错位铺开，看起来像"背景被放大且与翼型对不上"，
  // 而且不报错、不白屏，极难发现。所以在这里一次性散射回整网格。
  const total = grid.nx * grid.ny
  if (grid.mask.length !== total) {
    throw new Error('掩膜长度 ' + grid.mask.length + ' 与网格 ' + total + ' 不符')
  }
  const values = {} as Record<FieldName, Float32Array>
  for (let k = 0; k < meta.nFields; k++) {
    const phiOff = meta.phiOffset[k]
    const muOff = meta.muOffset[k]
    const mean = meta.meanOut[k]
    const std = meta.stdOut[k]
    const arr = new Float32Array(total).fill(NaN)
    let slot = 0
    for (let idx = 0; idx < total; idx++) {
      if (!grid.mask[idx]) continue
      let acc = 0
      for (let m = 0; m < meta.podR; m++) {
        acc += alpha[k * meta.podR + m] * data[phiOff + slot * meta.podR + m]
      }
      arr[idx] = (acc + data[muOff + slot]) * std + mean
      slot++
    }
    if (slot !== grid.nValid) {
      throw new Error('掩膜有效格 ' + slot + ' 与声明的 ' + grid.nValid + ' 不符')
    }
    values[meta.order[k]] = arr
  }
  return { values, usedModel: true, reason: '' }
}

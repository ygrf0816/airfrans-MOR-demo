// 案例流场数据：cases_field.bin 按 cases_field.json 的 blocks 切块。
// 前向预测页用它显示真实案例的流场；参数不匹配任何案例时返回 null 走解析近似。

import { assetUrl } from '../data/load.ts'
import type { Grid } from '../data/load.ts'
import type { FieldName } from '../data/mock.ts'

interface CaseMeta {
  code: string
  label: string
  L: number
  P: number
  Q: number
  T: number
  uInf: number
  aoas: number[]
}
interface BlockMeta {
  key: string
  field: string
  case: string
  aoa: number
  offset: number
  count: number
}
interface CasesFieldJson {
  cases: CaseMeta[]
  blocks: BlockMeta[]
  nGrid: number
  channelOrder: string[]
  unit?: Record<string, string> | string[]
}

let cached: Promise<{ meta: CasesFieldJson; bin: Float32Array } | null> | null = null

async function load(): Promise<{ meta: CasesFieldJson; bin: Float32Array } | null> {
  if (!cached) {
    cached = (async () => {
      try {
        const [metaRes, binRes] = await Promise.all([
          fetch(assetUrl('cases_field.json'), { cache: 'no-cache' }),
          fetch(assetUrl('cases_field.bin'), { cache: 'no-cache' }),
        ])
        if (!metaRes.ok || !binRes.ok) return null
        const meta = (await metaRes.json()) as CasesFieldJson
        const bin = new Float32Array(await binRes.arrayBuffer())
        if (bin.length === 0) return null
        return { meta, bin }
      } catch {
        return null
      }
    })()
  }
  return cached
}

/**
 * 找与参数最接近的案例和攻角，把 31066 个有效格填回 128×256 网格（掩膜外 NaN）。
 * 没有匹配或文件缺失时返回 null。
 */
export async function lookupCaseField(
  params: [number, number, number, number],
  aoa: number,
  field: FieldName,
  grid: Grid,
): Promise<{ field: Float32Array; caseLabel: string; caseAoa: number } | null> {
  const got = await load()
  if (!got) return null
  const { meta, bin } = got
  const [L, P, Q, T] = params
  // 参数距离：归一化欧氏距离
  let best: CaseMeta | null = null
  let bd = Infinity
  for (const c of meta.cases) {
    const d =
      ((c.L - L) / 4) ** 2 +
      ((c.P - P) / 5) ** 2 +
      (c.Q !== Q ? 1 : 0) ** 2 +
      ((c.T - T) / 15) ** 2
    if (d < bd) {
      bd = d
      best = c
    }
  }
  if (!best) return null
  // 匹配攻角取案例里最近的一档
  let bestAoa = best.aoas[0]
  for (const a of best.aoas) {
    if (Math.abs(a - aoa) < Math.abs(bestAoa - aoa)) bestAoa = a
  }
  const block = meta.blocks.find(
    (b) => b.case === best!.code && b.field === field && b.aoa === bestAoa,
  )
  if (!block) return null
  if (block.offset + block.count > bin.length) return null
  const out = new Float32Array(grid.nx * grid.ny).fill(NaN)
  const slice = bin.subarray(block.offset, block.offset + block.count)
  let k = 0
  for (let i = 0; i < out.length && k < slice.length; i++) {
    if (grid.mask[i]) out[i] = slice[k++]
  }
  return { field: out, caseLabel: best.label ?? best.code, caseAoa: bestAoa }
}

export function caseFieldAvailable(): Promise<boolean> {
  return load().then((v) => v !== null)
}

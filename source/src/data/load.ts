// 数据加载层：先试 fetch 真实文件，拿不到就退回内置假数据。
// 页面上用 DataSource 角标说明当前是示例数据还是真实数据。

import {
  mockCpBasis,
  mockGrid,
  mockManifest,
  mockResults,
  mockSamples,
  analyticCp,
  type AssetManifest,
  type CpBasis,
  type Grid,
  type Results,
  type SampleItem,
  type FrontPoint,
  type HistoryPoint,
  type CaseItem,
} from './mock.ts'
import type { NacaParams } from '../lib/naca.ts'

export type { AssetManifest, CpBasis, Grid, Results, SampleItem, FrontPoint, HistoryPoint, CaseItem }
export {
  analyticCp,
  analyticField,
  mockFieldBasis,
  type FieldBasisMeta,
  type FieldName,
} from './mock.ts'

const base = import.meta.env.BASE_URL || './'

export function assetUrl(name: string): string {
  return base + 'data/' + name
}

export type DataMode = 'real' | 'mock'

export interface DataState {
  mode: DataMode
  manifest: AssetManifest
  grid: Grid
  cpBasis: CpBasis
  samples: SampleItem[]
  results: Results
  /** 真实文件缺失的名字列表，用于界面提示。 */
  missing: string[]
}

let cached: Promise<DataState> | null = null
const listeners = new Set<(s: DataState) => void>()

async function tryFetchJson<T>(name: string): Promise<T | null> {
  try {
    const res = await fetch(assetUrl(name), { cache: 'no-cache' })
    if (!res.ok) return null
    return (await res.json()) as T
  } catch {
    return null
  }
}

async function tryFetchBin(name: string): Promise<Float32Array | null> {
  try {
    const res = await fetch(assetUrl(name), { cache: 'no-cache' })
    if (!res.ok) return null
    const buf = await res.arrayBuffer()
    return new Float32Array(buf)
  } catch {
    return null
  }
}

interface RawSample {
  i: number
  name: string
  uInf: number
  aoa: number
  series: '4digit' | '5digit'
  L: number | null
  M?: number | null
  P: number
  Q: number
  T: number
  clOfficial: number
  cdOfficial: number
  split: 'train' | 'test'
}

/** 把真实 samples.json 的条目与二进制行合成页面用的 SampleItem。 */
function adaptSamples(
  raw: RawSample[],
  cpBin: Float32Array | null,
  alphaBin: Float32Array | null,
  basis: CpBasis,
): SampleItem[] {
  return raw.map((s) => {
    const params: NacaParams =
      s.series === '5digit'
        ? [s.L ?? 0, s.P, s.Q ?? 0, s.T]
        : [s.M ?? 0, s.P, s.T]
    let cp: number[]
    if (cpBin && cpBin.length >= (s.i + 1) * 256) {
      cp = Array.from(cpBin.subarray(s.i * 256, s.i * 256 + 256))
    } else {
      cp = analyticCp(params, s.uInf, s.aoa)
    }
    let alpha: number[]
    if (alphaBin && alphaBin.length >= s.i * 136 + 8) {
      alpha = Array.from(alphaBin.subarray(s.i * 136, s.i * 136 + 8))
    } else {
      alpha = []
      for (let k = 0; k < basis.podR; k++) {
        let dot = 0
        for (let j = 0; j < basis.nPts; j++) dot += (cp[j] - basis.mu[j]) * basis.phi[j][k]
        alpha.push(dot)
      }
    }
    return { ...s, params, cp, alpha }
  })
}

async function loadAll(): Promise<DataState> {
  const missing: string[] = []
  let mode: DataMode = 'real'

  const manifest = await tryFetchJson<AssetManifest>('manifest.json')
  let finalManifest: AssetManifest
  if (manifest && Array.isArray(manifest.entries)) {
    finalManifest = manifest
  } else {
    finalManifest = mockManifest()
    mode = 'mock'
    missing.push('manifest.json')
  }

  // 依次探测关键文件；任何一个缺失就整页退回示例数据，保证一致性
  const probeNames = ['cp_basis.json', 'grid.json', 'samples.json', 'results.json']
  const probes = await Promise.all(probeNames.map((n) => tryFetchJson<unknown>(n)))
  const anyMissing = probes.some((p) => p === null)
  if (anyMissing) {
    mode = 'mock'
    probeNames.forEach((n, i) => {
      if (probes[i] === null) missing.push(n)
    })
  }

  if (mode === 'real') {
    const basis = probes[0] as CpBasis
    const gridRaw = probes[1] as {
      nx: number; ny: number; x0: number; x1: number; y0: number; y1: number
      x: number[]; y: number[]; maskRle: [number, number][]; nValid: number
    }
    const mask = new Array<boolean>(gridRaw.nx * gridRaw.ny)
    let p = 0
    for (const [val, cnt] of gridRaw.maskRle) {
      for (let k = 0; k < cnt; k++) mask[p++] = val === 1
    }
    const grid: Grid = {
      nx: gridRaw.nx, ny: gridRaw.ny,
      x0: gridRaw.x0, x1: gridRaw.x1, y0: gridRaw.y0, y1: gridRaw.y1,
      x: gridRaw.x, y: gridRaw.y, mask, nValid: gridRaw.nValid,
    }
    // 二进制附件缺失不致命：退回解析近似，但保持真实条目
    const [cpBin, alphaBin] = await Promise.all([
      tryFetchBin('samples_cp.bin'),
      tryFetchBin('samples_alpha.bin'),
    ])
    if (!cpBin) missing.push('samples_cp.bin')
    if (!alphaBin) missing.push('samples_alpha.bin')
    const samplesRaw = (probes[2] as { items: RawSample[] }).items
    const results = probes[3] as Results
    return {
      mode,
      manifest: finalManifest,
      grid,
      cpBasis: basis,
      samples: adaptSamples(samplesRaw, cpBin, alphaBin, basis),
      results,
      missing,
    }
  }

  return {
    mode: 'mock',
    manifest: finalManifest,
    grid: mockGrid(),
    cpBasis: mockCpBasis(),
    samples: mockSamples(1000),
    results: mockResults(),
    missing,
  }
}

export function loadData(): Promise<DataState> {
  if (!cached) {
    cached = loadAll().then((s) => {
      listeners.forEach((fn) => fn(s))
      return s
    })
  }
  return cached
}

/** 强制重新探测真实数据（不刷新页面），用于"重新检测数据"按钮。 */
export function reloadData(): Promise<DataState> {
  cached = null
  return loadData()
}

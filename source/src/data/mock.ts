// 确定性假数据层。真实数据文件没生成好之前，页面靠这一层完整演示。
// 全部用固定种子的线性同余伪随机，刷新画面不变，不用 Math.random()。
// 数据形状严格按 01_DATA_CONTRACT.md。

import { geomVector128, type NacaParams } from '../lib/naca.ts'

// ---- 固定种子伪随机（LCG，Numerical Recipes 常数）----
function makeRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
}

// ---- 类型定义（对齐契约）----

export interface ManifestEntry {
  name: string
  kind: string
  sizeMb: number
  required: boolean
  source: string
}
export interface AssetManifest {
  generatedAt: string
  version: number
  entries: ManifestEntry[]
  notes: string
}

export interface CpBasis {
  podR: number
  nPts: number
  s: number[]
  mu: number[]
  phi: number[][]
  smean: number[]
  sstd: number[]
  gmean: number[]
  gstd: number[]
  /** 流场模型的输入统计量，与压力曲线模型那一组不同，缺省时退回上面这一组。 */
  fieldSmean?: number[]
  fieldSstd?: number[]
  fieldGmean?: number[]
  fieldGstd?: number[]
  note: string
}

/** 样本可选的原始文件信息，由 tools/make_web_data.py 从原始数据测得。 */
export interface SampleFiles {
  /** 内部体积文件 internal.vtu 的字节数 */
  bytesVol?: number | null
  /** 该样本的网格点数 */
  nVol?: number | null
  /** 表面文件 aerofoil.vtp 的字节数 */
  bytesSurf?: number | null
  /** 表面点数 */
  nSurf?: number | null
}

export interface FieldBasisMeta {
  podR: number
  nGrid: number
  nFields: number
  order: string[]
  phiOffset: number[]
  muOffset: number[]
  totalFloats: number
  meanOut: number[]
  stdOut: number[]
  unit: string[]
  note: string
}

export interface Grid {
  nx: number
  ny: number
  x0: number
  x1: number
  y0: number
  y1: number
  x: number[]
  y: number[]
  mask: boolean[]
  nValid: number
}

export interface SampleItem extends SampleFiles {
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
  params: NacaParams
  cp: number[]
  alpha: number[]
}

export interface Results {
  dataset: {
    nTotal: number
    nTrain: number
    nTest: number
    nFourDigit: number
    nFiveDigit: number
    reRange: [number, number]
    aoaRange: [number, number]
    uInfRange: [number, number]
    aoaHist: { binEdges: number[]; counts: number[] }
    reHist: { binEdges: number[]; counts: number[] }
    splitCards: { name: string; train: number; test: number }[]
  }
  objective: {
    clTargetAoa: number
    bandAoas: number[]
    uInf: number
    machLimit: number
    sepLimit: number
    dbDesignBest: number
    bandUpperBound: number
    note: string
    /** 逐攻角升力表的攻角网格（0..14 度建每 1 度）。 */
    aoaGrid?: number[]
    clByAoaNote?: string
  }
  fronts: (FrontPoint & { hv?: number })[]
  history: HistoryPoint[]
  cases: CaseItem[]
  caveats: string[]
}

export interface FrontPoint {
  seed: number
  gen: number
  /** 原始种群行号（真实数据里有；mock 数据没有）。 */
  idx?: number
  L: number
  P: number
  Q: number
  T: number
  cl4: number
  cl8: number
  cl10: number
  /** 0..14 度每 1 度的升力（15 个数）。缺省时退回 cl4/cl8/cl10 线性插值。 */
  clByAoa?: number[]
  delta: number
  machMax: number
  sepMax: number
  feasible: boolean
  hv: number
}

export interface HistoryPoint {
  seed: number
  gen: number
  bestCl8: number
  hv: number
  nFeasible: number
}

export interface CaseItem {
  code: string
  label: string
  verdict: string
  L: number
  P: number
  Q: number
  T: number
  cl4: number
  cl8: number
  cl10: number
  delta: number
  machMax: number
  sepMax: number
  feasible: boolean
  reason: string
}

// ---- 压力曲线模型（解析近似，长得像真东西）----

/**
 * 用几条高斯凹坑拼出前缘吸力峰的压力曲线。
 * s 是归一化弧长 0..1，s=0 前缘，s=0.5 后缘，闭环。
 * 上表面前半段（s 0..0.5），下表面后半段（s 0.5..1）。
 */
export function analyticCp(
  params: NacaParams,
  uInf: number,
  aoa: number,
  nPts = 256,
): number[] {
  const five = params.length === 3
    ? { camber: params[0] / 100, thick: params[2] / 100 }
    : { camber: 0.15 * params[0], thick: params[3] / 100 }
  const aoaR = (aoa * Math.PI) / 180
  // 薄翼型理论：前缘吸力峰强度随攻角与弯度增大
  const peak = -(1.2 + 2.4 * aoaR + 5.0 * five.camber)
  const peakPos = 0.012 + 0.02 * Math.max(0, aoaR) // 吸力峰位置略随攻角后移
  const cp: number[] = []
  for (let i = 0; i < nPts; i++) {
    const s = i / (nPts - 1)
    let v: number
    if (s <= 0.5) {
      // 上表面，s 从 0 到 0.5 对应 x 从 0 到 1
      const x = s * 2
      const g1 = peak * Math.exp(-((x - peakPos) ** 2) / (2 * 0.012 ** 2))
      const g2 =
        -0.6 *
        (1 + 1.5 * aoaR) *
        Math.exp(-((x - 0.25) ** 2) / (2 * 0.09 ** 2))
      const rec = 1 - 0.85 * Math.exp(-((x - 0.85) ** 2) / (2 * 0.06 ** 2))
      v = (g1 + g2) * rec
    } else {
      // 下表面，压力侧，平缓正压
      const x = (1 - s) * 2
      v =
        0.35 *
        (0.4 + aoaR) *
        Math.exp(-((x - 0.3) ** 2) / (2 * 0.25 ** 2)) -
      0.15 * Math.exp(-((x - 0.9) ** 2) / (2 * 0.08 ** 2))
    }
    // 速度修正：动压不变（Cp 已归一化），只留一个小的 uInf 相关性
    v *= 1 + 0.0002 * (uInf - 62)
    cp.push(v)
  }
  return cp
}

// ---- 流场模型（绕圆柱势流加尾迹衰减）----

export type FieldName = 'u' | 'v' | 'p' | 'nu_t'

/** 在 256x128 公共网格上生成解析流场，掩膜外返回 NaN。 */
export function analyticField(
  params: NacaParams,
  uInf: number,
  aoa: number,
  field: FieldName,
  grid: Grid,
): Float32Array {
  const { nx, ny, x, y, mask } = grid
  const out = new Float32Array(nx * ny).fill(NaN)
  const aoaR = (aoa * Math.PI) / 180
  const thick = (params.length === 3 ? params[2] : params[3]) / 100
  const a = 0.08 + 0.5 * thick // 等效圆柱半径，随厚度增大
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i
      if (!mask[idx]) continue
      const xx = x[i] - 0.25 // 圆柱中心放在 x=0.25
      const yy = y[j]
      const r2 = xx * xx + yy * yy + 1e-8
      const a2 = a * a
      // 绕圆柱势流，叠加攻角旋转
      const ca = Math.cos(aoaR)
      const sa = Math.sin(aoaR)
      let u =
        uInf *
        (ca * (1 - (a2 * (xx * xx - yy * yy)) / (r2 * r2)) -
          sa * ((2 * a2 * xx * yy) / (r2 * r2)))
      let v =
        uInf *
        (sa * (1 + (a2 * (xx * xx - yy * yy)) / (r2 * r2)) +
          ca * ((2 * a2 * xx * yy) / (r2 * r2)))
      // 尾迹衰减：x>0.5 后方速度亏损
      if (xx > 0.3) {
        const wake = Math.exp(-((yy / (0.12 + 0.25 * (xx - 0.3))) ** 2))
        const deficit = 0.35 * uInf * wake * Math.exp(-(xx - 0.3) / 2.2)
        u -= deficit
        v -= deficit * 0.08 * Math.sign(yy)
      }
      let val: number
      if (field === 'u') val = u
      else if (field === 'v') val = v
      else if (field === 'p') {
        const q2 = u * u + v * v
        val = -0.5 * (q2 - uInf * uInf) // Cp 形式的压力（m^2/s^2）
      } else {
        // nu_t 涡粘：尾迹与近壁大
        const wall = Math.exp(-r2 / (2 * a2 * 4))
        const wake = xx > 0.3 ? Math.exp(-(yy * yy) / 0.02) * Math.exp(-(xx - 0.3) / 3) : 0
        val = 5e-4 * wall + 1.5e-3 * wake
      }
      out[idx] = val
    }
  }
  return out
}

// ---- 契约形状的假数据构造函数 ----

export function mockManifest(): AssetManifest {
  return {
    generatedAt: '示例数据（内置）',
    version: 1,
    entries: [
      { name: 'cp_basis.json', kind: 'basis', sizeMb: 0.06, required: true, source: 'mock' },
      { name: 'grid.json', kind: 'grid', sizeMb: 0.04, required: true, source: 'mock' },
      { name: 'samples.json', kind: 'samples', sizeMb: 0.6, required: true, source: 'mock' },
      { name: 'results.json', kind: 'results', sizeMb: 4.0, required: true, source: 'mock' },
    ],
    notes: '当前是内置示例数据，真实数据接入后此清单会被真实 manifest 替换',
  }
}

export function mockGrid(): Grid {
  const nx = 256
  const ny = 128
  const x0 = -2.16
  const x1 = 4.23
  const y0 = -1.65
  const y1 = 1.65
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < nx; i++) x.push(x0 + ((x1 - x0) * i) / (nx - 1))
  for (let j = 0; j < ny; j++) y.push(y0 + ((y1 - y0) * j) / (ny - 1))
  // 掩膜：剔除翼型内部（椭圆近似）与远场圆外
  const mask = new Array<boolean>(nx * ny)
  let nValid = 0
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const xx = x[i]
      const yy = y[j]
      // 翼型内部：以 (0.4, 0) 为中心的细长椭圆
      const inFoil =
        ((xx - 0.4) / 0.62) ** 2 + (yy / 0.09) ** 2 < 1 && xx > -0.05 && xx < 1.05
      // 远场剔除：半径 5.5 以外
      const far = (xx - 0.5) ** 2 + yy * yy > 30.25
      const ok = !inFoil && !far
      mask[j * nx + i] = ok
      if (ok) nValid++
    }
  }
  return { nx, ny, x0, x1, y0, y1, x, y, mask, nValid }
}

export function mockCpBasis(): CpBasis {
  const podR = 8
  const nPts = 256
  const s: number[] = []
  for (let i = 0; i < nPts; i++) s.push(i / (nPts - 1))
  // 均值曲线：一个典型的压力分布形状
  const mu = analyticCp([3.5, 6.5, 0, 13], 62, 5, nPts)
  // 8 条正交化过的模态形状（用勒让德样函数加符号变化，再 Gram-Schmidt）
  const raw: number[][] = []
  for (let k = 0; k < podR; k++) {
    const row: number[] = []
    for (let i = 0; i < nPts; i++) {
      const si = s[i]
      // 前缘集中的振型
      const base = Math.sin((k + 1) * Math.PI * si) * Math.exp(-si * (0.5 + k * 0.4))
      row.push(base)
    }
    raw.push(row)
  }
  // Gram-Schmidt 正交化
  for (let k = 0; k < podR; k++) {
    for (let j = 0; j < k; j++) {
      let dot = 0
      for (let i = 0; i < nPts; i++) dot += raw[k][i] * raw[j][i]
      for (let i = 0; i < nPts; i++) raw[k][i] -= dot * raw[j][i]
    }
    let norm = 0
    for (let i = 0; i < nPts; i++) norm += raw[k][i] * raw[k][i]
    norm = Math.sqrt(norm) || 1
    for (let i = 0; i < nPts; i++) raw[k][i] /= norm
  }
  const phi: number[][] = []
  for (let i = 0; i < nPts; i++) {
    const row: number[] = []
    for (let k = 0; k < podR; k++) row.push(raw[k][i] * 0.35) // 模态幅度量级
    phi.push(row)
  }
  // 归一化统计量：五标量与几何向量的均值方差（示例量级）
  const smean = [62, 5.5, 0.3, 0.12, 0.4]
  const sstd = [18, 5.5, 0.25, 0.045, 0.49]
  const gvec = geomVector128([3.5, 6.5, 0, 13])
  const gmean = gvec.map((v) => v * 0.9)
  const gstd = gvec.map(() => 0.03)
  return {
    podR,
    nPts,
    s,
    mu,
    phi,
    smean,
    sstd,
    gmean,
    gstd,
    note: '示例基：前缘锚定弧长序，s=0 前缘，s=0.5 后缘，闭环（示例数据）',
  }
}

export function mockSamples(n = 1000): SampleItem[] {
  const rng = makeRng(20261006)
  const items: SampleItem[] = []
  const basis = mockCpBasis()
  for (let i = 0; i < n; i++) {
    const isFive = rng() > 0.489
    const uInf = 31.4 + rng() * (93.6 - 31.4)
    const aoa = -4.94 + rng() * (14.93 + 4.94)
    let params: NacaParams
    let L: number | null = null
    let M: number | null = null
    let P: number
    let Q = 0
    let T: number
    if (isFive) {
      L = 0.004 + rng() * (3.993 - 0.004)
      P = 3.005 + rng() * (7.969 - 3.005)
      Q = rng() > 0.5 ? 1 : 0
      T = 5.008 + rng() * (19.989 - 5.008)
      params = [L, P, Q, T]
    } else {
      M = Math.round(rng() * 8)
      P = Math.round(1 + rng() * 8)
      T = 6 + Math.round(rng() * 15)
      Q = 0
      params = [M, P, T]
    }
    const cp = analyticCp(params, uInf, aoa)
    // 投影到示例基上得到 POD 坐标
    const alpha: number[] = []
    for (let k = 0; k < basis.podR; k++) {
      let dot = 0
      for (let j = 0; j < basis.nPts; j++) {
        dot += (cp[j] - basis.mu[j]) * basis.phi[j][k]
      }
      alpha.push(dot)
    }
    // 简单估算官方升力系数量级（只用于示例展示）
    const aoaR = (aoa * Math.PI) / 180
    const camber = params.length === 3 ? params[0] / 100 : 0.15 * params[0]
    const cl = 2 * Math.PI * (aoaR + 1.8 * camber) * 0.92
    const cd = 0.008 + 0.12 * aoaR * aoaR + 0.02 * camber
    const name = isFive
      ? `demo5_${uInf.toFixed(3)}_${aoa.toFixed(3)}_${L!.toFixed(3)}_${P.toFixed(2)}_${Q}_${T.toFixed(2)}`
      : `demo4_${uInf.toFixed(3)}_${aoa.toFixed(3)}_${M}_${P}_${T}`
    items.push({
      i,
      name,
      uInf,
      aoa,
      series: isFive ? '5digit' : '4digit',
      L,
      M,
      P,
      Q,
      T,
      clOfficial: cl,
      cdOfficial: cd,
      split: i % 5 === 4 ? 'test' : 'train',
      params,
      cp,
      alpha,
    })
  }
  return items
}

export function mockFieldBasis(): FieldBasisMeta {
  const nGrid = 31066
  const podR = 32
  const perField = nGrid * podR
  return {
    podR,
    nGrid,
    nFields: 4,
    order: ['u', 'v', 'p', 'nu_t'],
    phiOffset: [0, perField, 2 * perField, 3 * perField],
    muOffset: [4 * perField, 4 * perField + nGrid, 4 * perField + 2 * nGrid, 4 * perField + 3 * nGrid],
    totalFloats: 4 * perField + 4 * nGrid,
    meanOut: [42.44, 9.7, -452.4, 0.0008],
    stdOut: [29.64, 30.88, 2963.3, 0.0029],
    unit: ['m/s', 'm/s', 'm^2/s^2', 'm^2/s'],
    note: '示例元信息（示例数据）',
  }
}

export function mockResults(): Results {
  const fronts: FrontPoint[] = []
  const seeds = [42, 137, 2024]
  for (const seed of seeds) {
    const r2 = makeRng(seed)
    for (let k = 0; k < 40; k++) {
      const L = 0.5 + r2() * 3.4
      const P = 3.2 + r2() * 4.7
      const Q = r2() > 0.7 ? 1 : 0
      const T = 8 + r2() * 10
      const cl8 = 0.75 + 0.13 * L + 0.02 * (P - 5) + r2() * 0.08
      const delta = Math.max(0.02, 0.22 - 0.045 * L + r2() * 0.06)
      const cl4 = cl8 - delta * (0.7 + r2() * 0.5)
      const cl10 = cl8 - delta
      const machMax = 0.42 + 0.02 * L + 0.008 * T + r2() * 0.06
      const sepMax = 0.002 + 0.0016 * L + r2() * 0.004
      const feasible = machMax <= 0.6 && sepMax <= 0.007874 && delta <= 0.3
      fronts.push({
        seed, gen: 30 + Math.floor(r2() * 40), L, P, Q, T,
        cl4, cl8, cl10, delta, machMax, sepMax, feasible,
        hv: 0.3 + r2() * 0.25,
      })
    }
  }
  const history: HistoryPoint[] = []
  for (const seed of seeds) {
    const r2 = makeRng(seed * 7 + 1)
    let best = 0.85 + r2() * 0.1
    for (let genStep = 0; genStep <= 60; genStep += 2) {
      best = Math.min(1.32, best + 0.02 * r2() * (1 - genStep / 80))
      history.push({
        seed,
        gen: genStep,
        bestCl8: best,
        hv: 0.1 + 0.5 * (genStep / 60) * (0.8 + 0.2 * r2()),
        nFeasible: Math.floor(5 + 35 * (genStep / 60) * r2()),
      })
    }
  }
  const cases: CaseItem[] = [
    {
      code: 'case1_success', label: '成功案例', verdict: '通过',
      L: 3.67, P: 7.969, Q: 0, T: 15.08,
      cl4: 1.09, cl8: 1.24, cl10: 1.13, delta: 0.11,
      machMax: 0.55, sepMax: 0.0062, feasible: true,
      reason: '三项约束全部满足（示例数据）',
    },
    {
      code: 'case2_mach', label: '马赫超限', verdict: '不通过',
      L: 3.9, P: 7.5, Q: 0, T: 18.5,
      cl4: 1.15, cl8: 1.3, cl10: 1.16, delta: 0.14,
      machMax: 0.64, sepMax: 0.0055, feasible: false,
      reason: '最大马赫数代理超过 0.6 约束（示例数据）',
    },
    {
      code: 'case3_sep', label: '分离超限', verdict: '不通过',
      L: 3.95, P: 4.2, Q: 1, T: 12.3,
      cl4: 1.05, cl8: 1.18, cl10: 1.02, delta: 0.16,
      machMax: 0.52, sepMax: 0.0105, feasible: false,
      reason: '分离代理超过 0.007874 约束（示例数据）',
    },
  ]
  return {
    dataset: {
      nTotal: 1000, nTrain: 800, nTest: 200,
      nFourDigit: 489, nFiveDigit: 511,
      reRange: [2005000, 5999000],
      aoaRange: [-4.94, 14.93],
      uInfRange: [31.4, 93.6],
      aoaHist: {
        binEdges: [-5, -2, 1, 4, 7, 10, 13, 15],
        counts: [98, 152, 210, 236, 178, 96, 30],
      },
      reHist: {
        binEdges: [2e6, 2.6e6, 3.2e6, 3.8e6, 4.4e6, 5e6, 6e6],
        counts: [150, 190, 220, 180, 160, 100],
      },
      splitCards: [
        { name: 'full', train: 800, test: 200 },
        { name: 'scarce', train: 200, test: 200 },
        { name: 'sub10', train: 10, test: 200 },
        { name: 'sub30', train: 30, test: 200 },
        { name: 'sub50', train: 50, test: 200 },
        { name: 'sub100', train: 100, test: 200 },
        { name: 'sub300', train: 300, test: 200 },
        { name: 'sub500', train: 500, test: 200 },
      ],
    },
    objective: {
      clTargetAoa: 8,
      bandAoas: [4, 8, 10],
      uInf: 62,
      machLimit: 0.6,
      sepLimit: 0.007874,
      dbDesignBest: 1.146,
      bandUpperBound: 1.4225,
      note: '带内损失 = max(CL8-CL4, CL8-CL10)',
    },
    fronts,
    history,
    cases,
    caveats: [
      '当前页面展示的是内置示例数据，真实数据接入后会自动替换',
      '绝对升力值未经过风洞或官方力系数验证，只用相对排序与相对误差',
    ],
  }
}

export function mockCases(): CaseItem[] {
  return mockResults().cases
}

// NACA 四位族与五位族翼型几何生成器。
// 从 01_initial/AirfRANS-main/naca_generator.py 逐行移植，口径必须与 Python 完全一致。
// 几何向量 128 维口径与 03_code/prediction/geometry_input.py 的 geometric_vector 一致。

/** 四位族参数 [M, P, T]，五位族参数 [L, P, Q, T]。 */
export type NacaParams4 = [number, number, number]
export type NacaParams5 = [number, number, number, number]
export type NacaParams = NacaParams4 | NacaParams5

export function isFiveDigit(p: NacaParams): p is NacaParams5 {
  return p.length === 4
}

/** 厚度分布。CTE=true 表示闭合后缘，四次项系数 -0.1036。 */
export function thicknessDist(t: number, x: number, CTE = true): number {
  const a = CTE ? -0.1036 : -0.1015
  return (
    5 *
    t *
    (0.2969 * Math.sqrt(x) -
      0.126 * x -
      0.3516 * x * x +
      0.2843 * x * x * x +
      a * x * x * x * x)
  )
}

/**
 * 弯度线与弯度线斜率。
 * params 长度 2 是四位族 [M, P]，m = M/100，p = P/10。
 * params 长度 3 是五位族 [L, P, Q]，c_l = 0.15 L，x_f = P/20，
 * 中间变量 m 用牛顿迭代解 x*(1-sqrt(x/3)) - x_f = 0，容差 1e-15。
 */
export function camberLine(
  params: [number, number] | [number, number, number],
  x: number[],
): { yc: number[]; dyc: number[] } {
  const n = x.length
  const yc = new Array<number>(n).fill(0)
  const dyc = new Array<number>(n).fill(0)

  if (params.length === 2) {
    const m = params[0] / 100
    const p = params[1] / 10
    if (p === 0) {
      for (let i = 0; i < n; i++) dyc[i] = -2 * m * x[i]
      return { yc, dyc }
    }
    if (p === 1) {
      for (let i = 0; i < n; i++) dyc[i] = 2 * m * (1 - x[i])
      return { yc, dyc }
    }
    for (let i = 0; i < n; i++) {
      const xi = x[i]
      if (xi < p) {
        yc[i] = (m / (p * p)) * (2 * p * xi - xi * xi)
        dyc[i] = ((2 * m) / (p * p)) * (p - xi)
      } else {
        yc[i] =
          (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * xi - xi * xi)
        dyc[i] = ((2 * m) / ((1 - p) * (1 - p))) * (p - xi)
      }
    }
    return { yc, dyc }
  }

  if (params.length === 3) {
    const [l, p, q] = params
    const c_l = 0.15 * l
    const x_f = p / 20

    const f = (v: number) => v * (1 - Math.sqrt(v / 3)) - x_f
    const df = (v: number) => 1 - (3 * Math.sqrt(v / 3)) / 2
    let m = 0.5
    // 牛顿迭代，容差与 Python 一样是 1e-15
    for (let iter = 0; iter < 1000; iter++) {
      const nm = Math.max(m - f(m) / df(m), 0)
      if (Math.abs(m - nm) <= 1e-15) {
        m = nm
        break
      }
      m = nm
    }
    const r =
      (3 * m - 7 * m * m + 8 * m * m * m - 4 * m * m * m * m) /
        Math.sqrt(m * (1 - m)) -
      1.5 * (1 - 2 * m) * (Math.PI / 2 - Math.asin(1 - 2 * m))
    const k1 = c_l / r

    if (q === 0) {
      for (let i = 0; i < n; i++) {
        const xi = x[i]
        if (xi <= m) {
          yc[i] = k1 * (xi ** 3 - 3 * m * xi * xi + m * m * (3 - m) * xi)
          dyc[i] = k1 * (3 * xi * xi - 6 * m * xi + m * m * (3 - m))
        } else {
          yc[i] = k1 * m ** 3 * (1 - xi)
          dyc[i] = -k1 * m ** 3
        }
      }
    } else if (q === 1) {
      const k = (3 * (m - x_f) ** 2 - m ** 3) / (1 - m) ** 3
      for (let i = 0; i < n; i++) {
        const xi = x[i]
        if (xi <= m) {
          yc[i] =
            k1 *
            ((xi - m) ** 3 - k * (1 - m) ** 3 * xi - m ** 3 * xi + m ** 3)
          dyc[i] = k1 * (3 * (xi - m) ** 2 - k * (1 - m) ** 3 - m ** 3)
        } else {
          yc[i] =
            k1 *
            (k * (xi - m) ** 3 -
              k * (1 - m) ** 3 * xi -
              m ** 3 * xi +
              m ** 3)
          dyc[i] =
            k1 * (3 * k * (xi - m) ** 2 - k * (1 - m) ** 3 - m ** 3)
        }
      }
    } else {
      throw new Error('Q must be 0 for normal camber or 1 for reflex camber.')
    }
    return { yc, dyc }
  }

  throw new Error('camber params must have length 2 or 3')
}

export interface Outline {
  /** 轮廓点，顺序与 Python 一致：上表面 TE->LE，再下表面 LE->TE（去末点倒序）。 */
  pts: [number, number][]
  /** 弦长 c = xmax - xmin（旋转后轮廓的横轴跨度）。 */
  chord: number
  xMin: number
  xMax: number
}

/**
 * 生成翼型轮廓。
 * 采样用余弦站位 beta = pi * linspace(1, 0, N+1)，x = (1-cos(beta))/2，
 * x 从 1 递减到 0，即点序是后缘到前缘。
 * 拼接顺序：上表面全部点，接去掉末点的下表面倒序，首尾两点强制 (1, 0)。
 */
export function nacaOutline(params: NacaParams, nbSamples = 400): Outline {
  let camberParams: [number, number] | [number, number, number]
  let t: number
  if (params.length === 3) {
    camberParams = [params[0], params[1]]
    t = params[2] / 100
  } else if (params.length === 4) {
    camberParams = [params[0], params[1], params[2]]
    t = params[3] / 100
  } else {
    throw new Error('params must have length 3 (4-digit) or 4 (5-digit)')
  }

  const x: number[] = []
  for (let i = 0; i <= nbSamples; i++) {
    const beta = Math.PI * (1 - i / nbSamples)
    x.push((1 - Math.cos(beta)) / 2)
  }

  const { dyc } = camberLine(camberParams, x)
  const { yc } = camberLine(camberParams, x)
  const yt = x.map((xi) => thicknessDist(t, xi))

  const pts: [number, number][] = []
  for (let i = 0; i <= nbSamples; i++) {
    const theta = Math.atan(dyc[i])
    pts.push([
      x[i] - yt[i] * Math.sin(theta),
      yc[i] + yt[i] * Math.cos(theta),
    ])
  }
  for (let i = nbSamples - 1; i >= 0; i--) {
    const theta = Math.atan(dyc[i])
    pts.push([
      x[i] + yt[i] * Math.sin(theta),
      yc[i] - yt[i] * Math.cos(theta),
    ])
  }
  pts[0] = [1, 0]
  pts[pts.length - 1] = [1, 0]

  let xMin = Infinity
  let xMax = -Infinity
  for (const [px] of pts) {
    if (px < xMin) xMin = px
    if (px > xMax) xMax = px
  }
  return { pts, chord: xMax - xMin, xMin, xMax }
}

/** 64 个余弦站位：beta = pi * linspace(1, 0, 64, endpoint=False)，x = (1-cos)/2。 */
export function chordNodes64(): number[] {
  const n = 64
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const beta = Math.PI * (1 - i / n)
    out[i] = (1 - Math.cos(beta)) / 2
  }
  return out
}

/**
 * 128 维几何向量。
 * 先按轮廓的弦长 c 归一化，站位 xs = chordNodes64 * c + xMin，
 * 每个站位取带宽 (c/64)*0.5 内的轮廓点，最大 y/c 当上表面，最小 y/c 当下表面。
 * 前 64 个数是上表面，后 64 个数是下表面。与 geometry_input.py 口径一致。
 */
export function geomVector128(params: NacaParams, nbSamples = 400): number[] {
  const { pts, chord: c, xMin } = nacaOutline(params, nbSamples)
  return geomVector128FromOutline(pts, c, xMin)
}

export function geomVector128FromOutline(
  pts: [number, number][],
  c: number,
  xMin: number,
): number[] {
  const n = 64
  const stations = chordNodes64()
  const tol = (c / n) * 0.5
  const vec = new Array<number>(2 * n).fill(0)
  for (let i = 0; i < n; i++) {
    const xx = stations[i] * c + xMin
    let yMax = -Infinity
    let yMin = Infinity
    let count = 0
    for (const [px, py] of pts) {
      if (Math.abs(px - xx) < tol) {
        if (py > yMax) yMax = py
        if (py < yMin) yMin = py
        count++
      }
    }
    if (count >= 2) {
      vec[i] = yMax / c
      vec[n + i] = yMin / c
    } else if (count === 1) {
      vec[i] = yMax / c
      vec[n + i] = yMax / c
    }
  }
  return vec
}

/**
 * 五标量输入。
 * 四位族 [M, P, T] -> [uInf, aoa, M/100, T/100, 0]
 * 五位族 [L, P, Q, T] -> [uInf, aoa, 0.15*L, T/100, round(Q)]
 */
export function fiveScalar(
  params: NacaParams,
  uInf: number,
  aoa: number,
): number[] {
  if (params.length === 3) {
    const [M, , T] = params
    return [uInf, aoa, M / 100, T / 100, 0]
  }
  const [L, , Q, T] = params
  return [uInf, aoa, 0.15 * L, T / 100, Math.round(Q)]
}

/** 弯度线采样点，用于形状视图里的弯度线叠加显示。 */
export function camberCurve(
  params: NacaParams,
  nbSamples = 200,
): [number, number][] {
  const camberParams: [number, number] | [number, number, number] =
    params.length === 3
      ? [params[0], params[1]]
      : [params[0], params[1], params[2]]
  const x: number[] = []
  for (let i = 0; i <= nbSamples; i++) x.push(i / nbSamples)
  const { yc } = camberLine(camberParams, x)
  return x.map((xi, i) => [xi, yc[i]])
}

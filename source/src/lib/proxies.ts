// 两条约束代理，以及由压力曲线算升力的线性权重。
//
// 这三个函数都是从 03_code/inversion/ga_design.py 逐行移植的，不是另起一套。
// 演示页面上的指标必须与论文、报告、以及反向设计页面里的约束判定用同一套口径，
// 否则同一版设计在两处显示不同的数，评审一看就穿。
//
// 核对方见 app/scripts/check-proxies.mjs，拿 results.json 里已经算好的
// 七个案例的 machMax、sepMax、三个攻角的升力反过来验这三个函数。

import { nacaOutline, type NacaParams } from './naca.ts'

/** 来流速度，设计点固定每秒 62 米。 */
export const U_INF = 62
/** 声速，每秒 343 米。 */
export const SOUND_SPEED = 343
/** 局部马赫代理的上限。 */
export const MACH_LIMIT = 0.6
/** 分离代理的上限，等于一个离散步长。 */
export const SEP_LIMIT = 0.007874
/** 分离代理里判定平坦的梯度阈值。 */
const FLAT_THRESH = 0.05
/** 压力曲线点数。 */
const N_PTS = 256

/**
 * 局部马赫数代理，取吸力峰处的估计值。
 * 公式是来流速度除以声速，再乘根号下 1 减最小 Cp，最小 Cp 大于 1 时根号里取 0。
 * 这是代理量，不是求解出来的马赫数。
 */
export function machProxy(cp: ArrayLike<number>): number {
  let minCp = Infinity
  for (let i = 0; i < cp.length; i++) if (cp[i] < minCp) minCp = cp[i]
  return (U_INF / SOUND_SPEED) * Math.sqrt(Math.max(1 - minCp, 0))
}

/**
 * 分离代理，量的是吸力面上压力恢复平坦段的最长连续长度。
 *
 * 口径有三处怪癖，都是原实现已经核过、并且与已跑结果绑定的，不要改。
 * 第一，压力曲线是闭环，下标 0 与 255 都在前缘，约 127 在后缘。
 *   上半段 cp 的 0 到 127 要反转过来，才与下半段 128 到 255 方向一致，都是前缘到后缘。
 * 第二，配对以后用 0 到 1 均分 128 点去配，掩码取落在 0.3 到 0.95 的那一段，
 *   这一步实际对应物理弦向 0.05 到 0.70，不是名字暗示的后段。
 * 第三，取平均 Cp 更负的那一侧当吸力面，梯度小于 0.05 算平坦，
 *   返回值是平坦段长度乘一个格宽，所以取值离散，一格约 0.007874。
 */
export function sepProxy(cp: ArrayLike<number>): number {
  if (cp.length !== N_PTS) {
    throw new Error('压力曲线点数应为 ' + N_PTS + '，收到 ' + cp.length)
  }
  const half0 = new Float64Array(128)
  const half1 = new Float64Array(128)
  for (let i = 0; i < 128; i++) {
    half0[i] = cp[127 - i]      // cp 的 0 到 127 反转，变成前缘到后缘
    half1[i] = cp[128 + i]      // cp 的 128 到 255 本来就是前缘到后缘
  }
  let sum0 = 0
  let sum1 = 0
  for (let i = 0; i < 128; i++) { sum0 += half0[i]; sum1 += half1[i] }
  const side = sum0 / 128 < sum1 / 128 ? half0 : half1
  const n = 128
  const step = 1 / (n - 1)
  const idx0 = Math.ceil(0.3 * (n - 1))
  const idx1 = Math.floor(0.95 * (n - 1))
  let best = 0
  let cur = 0
  for (let i = idx0; i < idx1; i++) {
    const grad = Math.abs((side[i + 1] - side[i]) / step)
    if (grad < FLAT_THRESH) {
      cur += 1
      if (cur > best) best = cur
    } else {
      cur = 0
    }
  }
  return best * step
}

/**
 * 由 256 点压力曲线算升力系数的线性权重。
 *
 * 口径与 03_code/inversion/ga_design.py 的 build_weight_256 相同。
 * 先用官方生成器以 994 个采样点生成轮廓，去掉末尾重复点，把前缘转到下标 0，
 * 再由切线转 90 度得法向，并用"指向质心还是背离质心"定朝向，这里取内法向，
 * 与原实现一致，取外法向会让升力符号相反。
 * 然后把轮廓点的弧长位置映射到 256 个等分点上，累加成 256 个权重。
 *
 * 用法是点积，升力等于权重与压力曲线逐点相乘再求和。
 */
export function liftWeight256(params: NacaParams, aoaDeg: number): Float64Array {
  const outline = nacaOutline(params, 994)
  let pts = outline.pts
  // 生成器首尾都是后缘点，去掉末尾那个重复点
  const first = pts[0]
  const last = pts[pts.length - 1]
  if (Math.abs(first[0] - last[0]) < 1e-12 && Math.abs(first[1] - last[1]) < 1e-12) {
    pts = pts.slice(0, -1)
  }
  const n = pts.length
  let le = 0
  for (let i = 1; i < n; i++) if (pts[i][0] < pts[le][0]) le = i
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const src = pts[(le + i) % n]
    x[i] = src[0]
    y[i] = src[1]
  }
  let cx = 0
  let cy = 0
  for (let i = 0; i < n; i++) { cx += x[i]; cy += y[i] }
  cx /= n
  cy /= n
  const nrmX = new Float64Array(n)
  const nrmY = new Float64Array(n)
  const dl = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const ip = (i + 1) % n
    const im = (i - 1 + n) % n
    let tx = x[ip] - x[im]
    let ty = y[ip] - y[im]
    const norm = Math.hypot(tx, ty) + 1e-15
    tx /= norm
    ty /= norm
    let nx = ty
    let ny = -tx
    const outward = Math.sign((x[i] - cx) * nx + (y[i] - cy) * ny)
    nx *= -outward          // 反过来取内法向，与原实现一致
    ny *= -outward
    nrmX[i] = nx
    nrmY[i] = ny
    const jp = (i + 1) % n
    dl[i] = Math.hypot(x[jp] - x[i], y[jp] - y[i])
  }
  let total = 0
  for (let i = 0; i < n; i++) total += dl[i]
  const rad = (aoaDeg * Math.PI) / 180
  const sinA = Math.sin(rad)
  const cosA = Math.cos(rad)
  const w = new Float64Array(N_PTS)
  const step = 1 / (N_PTS - 1)
  let cum = 0
  for (let j = 0; j < n; j++) {
    const coef = -sinA * nrmX[j] + cosA * nrmY[j]
    const sj = cum / total
    const contrib = coef * dl[j]
    if (sj <= 0) {
      w[0] += contrib
    } else if (sj >= 1) {
      w[N_PTS - 1] += contrib
    } else {
      const i = Math.min(Math.floor(sj / step), N_PTS - 2)
      const frac = (sj - i * step) / step
      w[i] += contrib * (1 - frac)
      w[i + 1] += contrib * frac
    }
    cum += dl[j]
  }
  return w
}

/** 用权重与压力曲线点积得到升力系数。 */
export function liftFromCp(cp: ArrayLike<number>, w: Float64Array): number {
  let sum = 0
  for (let i = 0; i < w.length; i++) sum += w[i] * cp[i]
  return sum
}

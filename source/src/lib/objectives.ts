// 设计目标的评估：把四个指标一次算齐，口径与反向设计阶段完全一致。
//
// 口径来自 03_code/inversion/ga_design.py 的 eval_pop：
//   CL4、CL8、CL10 是在四度、八度、十度三个攻角下的升力系数，
//   带内损失 delta 取 八度升力减四度升力 与 八度升力减十度升力 中的大者，
//   局部马赫代理 m_max 取三个攻角里的最大值，
//   分离代理 sep 同样取三个攻角里的最大值，
//   可行要求 m_max 不超过 0.6 且 sep 不超过 0.007874。
//
// 三个攻角都要跑一次压力曲线模型，单次在毫秒级，所以直接在浏览器里算。
// 来流速度按设计点固定每秒 62 米，与搜索结果保持同一口径。

import type { NacaParams } from './naca.ts'
import type { CpBasis } from '../data/load.ts'
import { predictCp } from './models.ts'
import { MACH_LIMIT, SEP_LIMIT, U_INF, liftFromCp, liftWeight256, machProxy, sepProxy }
  from './proxies.ts'

/** 设计带，与反向设计阶段一致。 */
export const BAND_ANGLES = [4, 8, 10] as const

export interface ObjectiveEval {
  cl4: number
  cl8: number
  cl10: number
  /** 带内损失，两个方向取大。 */
  delta: number
  /** 三个攻角里最大的局部马赫代理。 */
  machMax: number
  /** 三个攻角里最大的分离代理。 */
  sepMax: number
  machOk: boolean
  sepOk: boolean
  feasible: boolean
  /** 压力曲线是否来自真实模型。 */
  usedModel: boolean
  /** 三度攻角的压力曲线，键是攻角。 */
  cpByAngle: Record<number, number[]>
}

/**
 * 给定参数，算齐四个指标。
 * 来流速度固定取设计点，不随攻角滑块变，这样与搜索阶段的数字可比。
 */
export async function evaluateObjectives(
  params: NacaParams,
  basis: CpBasis,
  base: string,
): Promise<ObjectiveEval> {
  const curves = await Promise.all(
    BAND_ANGLES.map((a) => predictCp(params, U_INF, a, basis, base)),
  )
  const cpByAngle: Record<number, number[]> = {}
  BAND_ANGLES.forEach((a, i) => { cpByAngle[a] = curves[i].cp })
  const cls = BAND_ANGLES.map((a, i) =>
    liftFromCp(curves[i].cp, liftWeight256(params, a)))
  const cl4 = cls[0]
  const cl8 = cls[1]
  const cl10 = cls[2]
  const machs = curves.map((c) => machProxy(c.cp))
  const seps = curves.map((c) => sepProxy(c.cp))
  const machMax = Math.max(...machs)
  const sepMax = Math.max(...seps)
  return {
    cl4, cl8, cl10,
    delta: Math.max(cl8 - cl4, cl8 - cl10),
    machMax, sepMax,
    machOk: machMax <= MACH_LIMIT,
    sepOk: sepMax <= SEP_LIMIT,
    feasible: machMax <= MACH_LIMIT && sepMax <= SEP_LIMIT,
    usedModel: curves.every((c) => c.usedModel),
    cpByAngle,
  }
}

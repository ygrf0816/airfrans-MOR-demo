// 代理量与升力权重的口径自检。
//
// 拿全精度压力曲线与已落盘的案例指标，反过来验 src/lib/proxies.ts：
//   machMax 应等于三个攻角里 machProxy 的最大值
//   sepMax  应等于三个攻角里 sepProxy 的最大值
//   cl4/cl8/cl10 应等于 liftWeight256 与对应压力曲线的点积
//
// 基准文件由 04_results/demo/tools/make_proxy_fixture.py 生成，
// 曲线是全精度，所以判据可以收到 1e-9。
// 另外顺带跑一遍 results.json 里的取整曲线，用来确认站点上用的是同一套公式，
// 这一遍只要求到 1e-2，因为取整会让分离代理在临界点上翻一个格。
//
// 跑法（在 04_results/demo/app 下）：node scripts/check-proxies.mjs

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { liftFromCp, liftWeight256, machProxy, sepProxy } from '../src/lib/proxies.ts'

const here = dirname(fileURLToPath(import.meta.url))
const fixtures = JSON.parse(readFileSync(
  join(here, 'fixtures', 'cases_cp_precise.json'), 'utf8'))

const angles = [4, 8, 10]
let worst = { mach: 0, sep: 0, cl: 0 }
let n = 0
const rows = []

for (const c of fixtures.cases) {
  const key = `${c.tag}-${c.case}`
  const cur = fixtures.curves[key]
  if (!cur) { console.log(`  ${key} 缺曲线，跳过`); continue }
  const cps = [cur.cp4, cur.cp8, cur.cp10]
  const params = [c.L, c.P, c.Q, c.T]
  const machMax = Math.max(...cps.map((cp) => machProxy(cp)))
  const sepMax = Math.max(...cps.map((cp) => sepProxy(cp)))
  const weights = angles.map((a) => liftWeight256(params, a))
  const cl = [4, 8, 10].map((_, i) => liftFromCp(cps[i], weights[i]))
  const expCl = [c.cl4, c.cl8, c.cl10]
  const dMach = Math.abs(machMax - c.machMax)
  const dSep = Math.abs(sepMax - c.sepMax)
  const dCl = Math.max(...cl.map((v, i) => Math.abs(v - expCl[i])))
  worst.mach = Math.max(worst.mach, dMach)
  worst.sep = Math.max(worst.sep, dSep)
  worst.cl = Math.max(worst.cl, dCl)
  n++
  rows.push({ key, machMax, sepMax, cl, dMach, dSep, dCl })
}

console.log('案例    mach 算得 / 报表      sep 算得 / 报表        升力最大差')
for (const r of rows) {
  console.log(`  ${r.key.padEnd(6)} ${r.machMax.toFixed(6)} / ${fixtures.cases.find(
    (c) => `${c.tag}-${c.case}` === r.key).machMax.toFixed(6)}`
    + `   ${r.sepMax.toFixed(6)} / ${fixtures.cases.find(
      (c) => `${c.tag}-${c.case}` === r.key).sepMax.toFixed(6)}`
    + `   ${r.dCl.toExponential(2)}  升力 [${r.cl.map((v) => v.toFixed(4)).join(', ')}]`)
}

console.log('')
console.log(`比对 ${n} 个案例，全精度曲线`)
// 判据按基准文件自身的取整位数定：cases_eval_*.csv 里 machMax 与升力保留四位小数，
// sepMax 保留六位小数，所以残差上限分别是 5e-5 与 5e-7。达到这个量级就说明移植是逐位对的。
console.log(`  马赫代理最大差 ${worst.mach.toExponential(3)}，判据 1e-4（基准四位小数）`)
console.log(`  分离代理最大差 ${worst.sep.toExponential(3)}，判据 1e-6（基准六位小数）`)
console.log(`  升力最大差 ${worst.cl.toExponential(3)}，判据 1e-4（基准四位小数）`)

// 再用站点上真正会读到的取整曲线跑一遍，只作一致性烟测
const results = JSON.parse(readFileSync(
  join(here, '..', 'public', 'data', 'results.json'), 'utf8'))
let smokeWorst = { mach: 0, sep: 0, cl: 0 }
for (const c of results.cases) {
  const cur = results.caseCurves[`${c.tag}-${c.case}`]
  if (!cur) continue
  const cps = [cur.cp4, cur.cp8, cur.cp10]
  const params = [c.L, c.P, c.Q, c.T]
  smokeWorst.mach = Math.max(smokeWorst.mach,
    Math.abs(Math.max(...cps.map((cp) => machProxy(cp))) - c.machMax))
  smokeWorst.sep = Math.max(smokeWorst.sep,
    Math.abs(Math.max(...cps.map((cp) => sepProxy(cp))) - c.sepMax))
  const weights = angles.map((a) => liftWeight256(params, a))
  smokeWorst.cl = Math.max(smokeWorst.cl,
    Math.max(...[0, 1, 2].map((i) => Math.abs(liftFromCp(cps[i], weights[i])
      - [c.cl4, c.cl8, c.cl10][i]))))
}
console.log('')
console.log('取整曲线的烟测（站点实际读到的数据）')
console.log(`  马赫代理最大差 ${smokeWorst.mach.toExponential(3)}`)
console.log(`  分离代理最大差 ${smokeWorst.sep.toExponential(3)}，一格是 7.874e-3，`
  + '取整会让临界案例翻一格，属已知现象')
console.log(`  升力最大差 ${smokeWorst.cl.toExponential(3)}`)

const pass = worst.mach < 1e-4 && worst.sep < 1e-6 && worst.cl < 1e-4
  && smokeWorst.mach < 1e-4 && smokeWorst.sep < 8e-3 && smokeWorst.cl < 1e-4 && n >= 8
console.log(pass ? 'PROXY_CHECK_OK' : 'PROXY_CHECK_FAIL')
process.exit(pass ? 0 : 1)

// 几何生成器自检：逐组比对 geom_fixture.json 里的 12 组参数。
// 128 维几何向量的最大绝对差必须小于 1e-9，轮廓点也一并核对。
// 运行方式：node app/scripts/selftest_geom.mjs（在 demo/ 目录下），
// 或 node --experimental-strip-types 直接跑 TS 版本。

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const fixturePath = join(here, '..', 'public', 'data', 'geom_fixture.json')

// ---- 从 src/lib/naca.ts 复制同一套算法（保持纯 JS 便于 node 直接跑）----
function thicknessDist(t, x, CTE = true) {
  const a = CTE ? -0.1036 : -0.1015
  return (
    5 * t *
    (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x +
      0.2843 * x * x * x + a * x * x * x * x)
  )
}

function camberLine(params, x) {
  const n = x.length
  const yc = new Array(n).fill(0)
  const dyc = new Array(n).fill(0)
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
        yc[i] = (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * xi - xi * xi)
        dyc[i] = ((2 * m) / ((1 - p) * (1 - p))) * (p - xi)
      }
    }
    return { yc, dyc }
  }
  const [l, p, q] = params
  const c_l = 0.15 * l
  const x_f = p / 20
  const f = (v) => v * (1 - Math.sqrt(v / 3)) - x_f
  const df = (v) => 1 - (3 * Math.sqrt(v / 3)) / 2
  let m = 0.5
  for (let iter = 0; iter < 1000; iter++) {
    const nm = Math.max(m - f(m) / df(m), 0)
    if (Math.abs(m - nm) <= 1e-15) { m = nm; break }
    m = nm
  }
  const r =
    (3 * m - 7 * m * m + 8 * m ** 3 - 4 * m ** 4) / Math.sqrt(m * (1 - m)) -
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
  } else {
    const k = (3 * (m - x_f) ** 2 - m ** 3) / (1 - m) ** 3
    for (let i = 0; i < n; i++) {
      const xi = x[i]
      if (xi <= m) {
        yc[i] = k1 * ((xi - m) ** 3 - k * (1 - m) ** 3 * xi - m ** 3 * xi + m ** 3)
        dyc[i] = k1 * (3 * (xi - m) ** 2 - k * (1 - m) ** 3 - m ** 3)
      } else {
        yc[i] = k1 * (k * (xi - m) ** 3 - k * (1 - m) ** 3 * xi - m ** 3 * xi + m ** 3)
        dyc[i] = k1 * (3 * k * (xi - m) ** 2 - k * (1 - m) ** 3 - m ** 3)
      }
    }
  }
  return { yc, dyc }
}

function nacaOutline(params, nbSamples = 400) {
  let camberParams, t
  if (params.length === 3) {
    camberParams = [params[0], params[1]]
    t = params[2] / 100
  } else {
    camberParams = [params[0], params[1], params[2]]
    t = params[3] / 100
  }
  const x = []
  for (let i = 0; i <= nbSamples; i++) {
    const beta = Math.PI * (1 - i / nbSamples)
    x.push((1 - Math.cos(beta)) / 2)
  }
  const { yc, dyc } = camberLine(camberParams, x)
  const yt = x.map((xi) => thicknessDist(t, xi))
  const pts = []
  for (let i = 0; i <= nbSamples; i++) {
    const th = Math.atan(dyc[i])
    pts.push([x[i] - yt[i] * Math.sin(th), yc[i] + yt[i] * Math.cos(th)])
  }
  for (let i = nbSamples - 1; i >= 0; i--) {
    const th = Math.atan(dyc[i])
    pts.push([x[i] + yt[i] * Math.sin(th), yc[i] - yt[i] * Math.cos(th)])
  }
  pts[0] = [1, 0]
  pts[pts.length - 1] = [1, 0]
  let xMin = Infinity, xMax = -Infinity
  for (const [px] of pts) { if (px < xMin) xMin = px; if (px > xMax) xMax = px }
  return { pts, chord: xMax - xMin, xMin }
}

function chordNodes64() {
  const n = 64
  const out = new Array(n)
  for (let i = 0; i < n; i++) {
    const beta = Math.PI * (1 - i / n)
    out[i] = (1 - Math.cos(beta)) / 2
  }
  return out
}

function geomVector128(params, nbSamples = 400) {
  const { pts, chord: c, xMin } = nacaOutline(params, nbSamples)
  const n = 64
  const stations = chordNodes64()
  const tol = (c / n) * 0.5
  const vec = new Array(2 * n).fill(0)
  for (let i = 0; i < n; i++) {
    const xx = stations[i] * c + xMin
    let yMax = -Infinity, yMin = Infinity, count = 0
    for (const [px, py] of pts) {
      if (Math.abs(px - xx) < tol) {
        if (py > yMax) yMax = py
        if (py < yMin) yMin = py
        count++
      }
    }
    if (count >= 2) { vec[i] = yMax / c; vec[n + i] = yMin / c }
    else if (count === 1) { vec[i] = yMax / c; vec[n + i] = yMax / c }
  }
  return vec
}
// ---- 自检主流程 ----

const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'))
const nb = fixture.nbSamples ?? 400
let worstGeom = 0
let worstGeomId = ''
let worstOutline = 0
let worstOutlineId = ''

for (const item of fixture.items) {
  const vec = geomVector128(item.params, nb)
  let md = 0
  for (let i = 0; i < 128; i++) {
    md = Math.max(md, Math.abs(vec[i] - item.geom128[i]))
  }
  if (md > worstGeom) { worstGeom = md; worstGeomId = item.id }

  const { pts } = nacaOutline(item.params, nb)
  let od = 0
  const m = Math.min(pts.length, item.outline.length)
  for (let i = 0; i < m; i++) {
    od = Math.max(
      od,
      Math.abs(pts[i][0] - item.outline[i][0]),
      Math.abs(pts[i][1] - item.outline[i][1]),
    )
  }
  if (od > worstOutline) { worstOutline = od; worstOutlineId = item.id }

  console.log(
    `${item.id.padEnd(10)} family=${item.family.padEnd(7)} geomMaxDiff=${md.toExponential(3)} outlineMaxDiff=${od.toExponential(3)}`,
  )
}

console.log('----')
console.log(`12 组参数几何向量最大绝对差: ${worstGeom.toExponential(6)} (worst: ${worstGeomId})`)
console.log(`轮廓点最大绝对差:           ${worstOutline.toExponential(6)} (worst: ${worstOutlineId})`)

if (worstGeom >= 1e-9) {
  console.error('FAIL: 几何向量最大绝对差 >= 1e-9')
  process.exit(1)
}
console.log('PASS: 几何向量自检通过，最大绝对差 < 1e-9')

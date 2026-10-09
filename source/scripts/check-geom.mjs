// 几何自检脚本：读 public/data/geom_fixture.json，对 12 条逐条按
// geom128Recipe 用 outline 重算 geom128，打印最大绝对差，判据 1e-9。
// 输出追加到 04_results/demo/_geom_check.txt。
//
// 用法：cd 04_results/demo/app && node scripts/check-geom.mjs

import { readFileSync, appendFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const fixturePath = path.resolve(__dirname, "..", "public", "data", "geom_fixture.json")
const fixture = JSON.parse(readFileSync(fixturePath, "utf-8"))

function geom128FromOutline(outline) {
  // step 2: c = maxX - minX, x0 = minX
  let xMin = Infinity, xMax = -Infinity
  for (const [px] of outline) {
    if (px < xMin) xMin = px
    if (px > xMax) xMax = px
  }
  const c = xMax - xMin
  // step 3: 余弦站位，xs[i] = (1 - cos(pi*(1 - i/64)))/2 * c + x0, i=0..63
  const n = 64
  const tol = (c / n) * 0.5
  const vec = new Array(2 * n).fill(0)
  for (let i = 0; i < n; i++) {
    const xs = ((1 - Math.cos(Math.PI * (1 - i / n))) / 2) * c + xMin
    // step 4/5/6
    let yMax = -Infinity, yMin = Infinity, count = 0
    for (const [px, py] of outline) {
      if (Math.abs(px - xs) < tol) {
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

const TOL = 1e-9
const lines = []
let allOk = true

for (const item of fixture.items) {
  const vec = geom128FromOutline(item.outline)
  const target = item.geom128
  let maxDiff = 0
  for (let i = 0; i < 128; i++) {
    const d = Math.abs(vec[i] - target[i])
    if (d > maxDiff) maxDiff = d
  }
  const ok = maxDiff < TOL
  if (!ok) allOk = false
  lines.push(
    `${ok ? "OK " : "BAD"} ${item.id.padEnd(10)}  maxDiff=${maxDiff.toExponential(4)}  (tol ${TOL})`,
  )
}

lines.push("")
lines.push(`verdict: ${allOk ? "PASS all 12 items < " + TOL : "FAIL, see BAD lines above"}`)
lines.push(`fixture: ${fixturePath}`)
lines.push(`run at: ${new Date().toISOString()}`)

const out = lines.join("\n") + "\n"
appendFileSync(
  path.resolve(__dirname, "..", "..", "_geom_check.txt"),
  "=== geom check run ===\n" + out,
)
process.stdout.write(out)

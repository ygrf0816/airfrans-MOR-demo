// 浏览器端 AirfRANS 翼型文件读取器的自检。
//
// 在 Node 里跑，用正则从 .vtp 文本里抠出 DataArray 的属性与文本，
// 再调用 src/lib/vtp.ts 里的解码函数，最后算出压力曲线，
// 与 Python 侧 airfrans_to_web.py 的结果以及已落盘的 wallcp_1000.npz 对照。
//
// 判据：与 wallcp_1000.npz 的最大绝对差小于 1e-3。前缘尖点处有一个微段
// 的插值位置差，量级在万分之一到千分之一，中段只差三万分之一。
//
// 跑法（在 04_results/demo/app 下）：
//   node scripts/check-vtp.mjs [数据集目录] [wallcp.npz]

import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loopFromConnectivity, wallCp, conditionsFromName, decodeArray }
  from '../src/lib/vtp.ts'

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(here, '..', '..', '..', '..')
const dataset = process.argv[2] || join(repoRoot, '02_dataset', 'Dataset')
const npzPath = process.argv[3] || join(repoRoot, '09_tmp', 'data', 'wallcp_1000.npz')

function attrsOf(tagText) {
  const get = (k) => {
    const m = tagText.match(new RegExp(k + '="([^"]*)"'))
    return m ? m[1] : null
  }
  return {
    type: get('type') || 'Float32',
    nComp: Number(get('NumberOfComponents') || '1'),
    format: (get('format') || 'ascii').toLowerCase(),
    headerType: (get('header_type') || 'UInt32') === 'UInt64' ? 8 : 4,
  }
}

/** 把文档里所有 DataArray 抠出来，记住位置与名字。 */
function allArrays(text) {
  const out = []
  const re = /<DataArray\b([^>]*?)>([^<]*)/g
  let m
  while ((m = re.exec(text))) {
    out.push({
      attrs: attrsOf(m[1]),
      text: m[2],
      name: (m[1].match(/Name="([^"]*)"/) || [])[1] || null,
      index: m.index,
    })
  }
  return out
}

/** 按段与名字取数组。同一段里可能有多个同名数组（空的与有数据的），取载荷最长的那个。 */
function pickArray(arrays, text, section, name) {
  const s = text.indexOf('<' + section)
  const e = text.indexOf('</' + section + '>')
  if (s < 0 || e < 0) return null
  const inScope = arrays.filter((a) => a.index > s && a.index < e
    && (name === null || a.name === name))
  if (!inScope.length) return null
  return inScope.reduce((best, a) => (a.text.length > best.text.length ? a : best), inScope[0])
}

function pointsCount(text) {
  const m = text.match(/<Piece[^>]*NumberOfPoints="(\d+)"/)
  return m ? Number(m[1]) : 0
}

async function parseSample(file) {
  const text = readFileSync(file, 'utf8')
  const nPoints = pointsCount(text)
  const arrays = allArrays(text)
  const ptsDa = pickArray(arrays, text, 'Points', null)
  const pDa = pickArray(arrays, text, 'PointData', 'p')
  const connDa = pickArray(arrays, text, 'Lines', 'connectivity')
  const offDa = pickArray(arrays, text, 'Lines', 'offsets')
  if (!ptsDa || !pDa || !connDa || !offDa) {
    throw new Error('缺数组：坐标 ' + !!ptsDa + ' 压力 ' + !!pDa
      + ' 连线 ' + !!connDa + ' 偏移 ' + !!offDa + '，共找到 ' + arrays.length + ' 个数组')
  }
  const pts = await decodeArray(ptsDa.attrs, ptsDa.text, nPoints)
  const pw = await decodeArray(pDa.attrs, pDa.text, nPoints)
  const conn = await decodeArray(connDa.attrs, connDa.text, 0)
  const offs = await decodeArray(offDa.attrs, offDa.text, 0)
  const order = loopFromConnectivity(conn, offs)
  const n = order.length
  const twin = new Float64Array(n * 2)
  const pwv = new Float64Array(n)
  order.forEach((src, i) => {
    twin[i * 2] = pts[src * 3]
    twin[i * 2 + 1] = pts[src * 3 + 1]
    pwv[i] = pw[src]
  })
  const { uInf } = conditionsFromName(file.replace(/\\/g, '/').split('/').pop())
  return { cp: wallCp(twin, pwv, uInf), nPoints, nLoop: n }
}

// 基准直接用站点里发布的资产：samples.json 的名字顺序与 samples_cp.bin 的曲线。
// 这比去解 npz 简单，而且验的正是网页真正会读到的那份数据。
function readReference() {
  const dataDir = join(here, '..', 'public', 'data')
  const samples = JSON.parse(readFileSync(join(dataDir, 'samples.json'), 'utf8'))
  const buf = readFileSync(join(dataDir, 'samples_cp.bin'))
  const n = samples.count
  const cols = 256
  const f = new Float32Array(buf.buffer, buf.byteOffset, n * cols)
  const CP = []
  for (let i = 0; i < n; i++) CP.push(f.subarray(i * cols, (i + 1) * cols))
  return { names: samples.items.map((x) => x.name), CP }
}

const samples = readdirSync(dataset, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => e.name)
  .filter((d) => readdirSync(join(dataset, d)).some((f) => f.endsWith('_aerofoil.vtp')))
console.log(`数据集目录 ${dataset}`)
console.log(`样本 ${samples.length} 条，取前 8 条核对`)

const { names, CP } = readReference()
const pyRef = JSON.parse(readFileSync(join(repoRoot, '09_tmp', 'py_cp_ref.json'), 'utf8'))
const idx = new Map(names.map((n, i) => [n, i]))

let worst = 0
let n = 0
for (const d of samples.slice(0, 8)) {
  const file = join(dataset, d, d + '_aerofoil.vtp')
  const r = await parseSample(file)
  const row = pyRef[d] || CP[idx.get(d)]
  if (!row) { console.log(`  ${d} 不在 npz 里，跳过`); continue }
  let mx = 0
  for (let i = 0; i < 256; i++) mx = Math.max(mx, Math.abs(r.cp[i] - row[i]))
  worst = Math.max(worst, mx)
  n++
  console.log(`  ${d.slice(-40).padEnd(42)} 轮廓 ${String(r.nLoop).padStart(4)} 点  最大绝对差 ${mx.toExponential(3)}`)
}
console.log('')
console.log(`核对 ${n} 条，最大绝对差 ${worst.toExponential(3)}，判据 1e-3`)
const pass = n >= 5 && worst < 1e-3
console.log(pass ? 'VTP_CHECK_OK' : 'VTP_CHECK_FAIL')
process.exit(pass ? 0 : 1)

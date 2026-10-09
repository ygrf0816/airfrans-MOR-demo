// 在浏览器里直接读 AirfRANS 的翼型文件（*_aerofoil.vtp）。
//
// 有了它，"上传自己的数据"就不需要先跑 Python 脚本，用户把数据集里的原始
// 翼型文件拖进来即可，页面自己算出压力曲线，再走同一套模态投影。
//
// 文件格式是 VTK 的 XML PolyData，要点是二进制载荷的编码方式很特别：
// 载荷被切成若干块，每块单独用 zlib 压缩，再各自 base64 编码后首尾相接写在
// 同一个文本节点里。整段一次性 base64 解码会在第一处补位等于号那里断掉，
// 后半截数据全丢。正确做法是按四个字符一组逐组解码再拼起来。
//
// 这一点是从 03_code/probe/vtkxml2.py 核出来的，那份代码在 1000 条样本上跑通过。
//
// 口径与 04_results/demo/tools/airfrans_to_web.py 完全一致，
// 包括前缘锚点取最小 x 的下一个点这个约定。

export interface VtpSample {
  /** 从文件名解析出的样本名 */
  name: string
  /** 来流速度，米每秒 */
  uInf: number
  /** 攻角，度 */
  aoa: number
  /** 轮廓点数 */
  nPoints: number
  /** 压力曲线，256 点，前缘锚定弧长序 */
  cp256: number[]
  /** 轮廓点，按闭环顺序，用于画形状 */
  outline: Array<[number, number]>
}

const DTYPE: Record<string, { bytes: number, get: (dv: DataView, o: number) => number }> = {
  Float32: { bytes: 4, get: (dv, o) => dv.getFloat32(o, true) },
  Float64: { bytes: 8, get: (dv, o) => dv.getFloat64(o, true) },
  Int32: { bytes: 4, get: (dv, o) => dv.getInt32(o, true) },
  Int64: { bytes: 8, get: (dv, o) => Number(dv.getBigInt64(o, true)) },
  UInt32: { bytes: 4, get: (dv, o) => dv.getUint32(o, true) },
  UInt64: { bytes: 8, get: (dv, o) => Number(dv.getBigUint64(o, true)) },
  Int16: { bytes: 2, get: (dv, o) => dv.getInt16(o, true) },
  UInt8: { bytes: 1, get: (dv, o) => dv.getUint8(o) },
  Int8: { bytes: 1, get: (dv, o) => dv.getInt8(o) },
}

/** 按四个字符一组解码 base64，拼回真正的字节流。 */
function base64Stream(text: string): Uint8Array {
  const s = text.replace(/\s+/g, '')
  const groups = Math.ceil(s.length / 4)
  const chunks: Uint8Array[] = []
  let total = 0
  for (let i = 0; i < groups; i++) {
    const g = s.slice(i * 4, i * 4 + 4)
    const bin = atob(g.padEnd(4, '='))
    const u = new Uint8Array(bin.length)
    for (let k = 0; k < bin.length; k++) u[k] = bin.charCodeAt(k)
    chunks.push(u)
    total += u.length
  }
  const out = new Uint8Array(total)
  let p = 0
  for (const c of chunks) { out.set(c, p); p += c.length }
  return out
}

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream('deflate')
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(ds)
  const buf = await new Response(stream).arrayBuffer()
  return new Uint8Array(buf)
}

export interface ArrayAttrs {
  type: string
  nComp: number
  format: string
  headerType: number
}

/**
 * 解一个 DataArray。参数与文本分开传，这样在浏览器外用 DOM 之外的途径
 * 也能调用它做核对（见 app/scripts/check-vtp.mjs）。
 */
export async function decodeArray(
  attrs: ArrayAttrs, text: string, nTuples: number,
): Promise<number[]> {
  const spec = DTYPE[attrs.type] || DTYPE.Float32
  if (attrs.format !== 'binary') {
    const vals = text.trim().split(/\s+/).map(Number)
    return vals.slice(0, nTuples * attrs.nComp)
  }
  const raw = base64Stream(text)
  // 空数组只写了头没有数据块，这种情况直接返回空，不要越界读。
  if (raw.byteLength < attrs.headerType * 3 + attrs.headerType) return []
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
  const readU = (o: number) => attrs.headerType === 8
    ? Number(dv.getBigUint64(o, true)) : dv.getUint32(o, true)
  const nBlocks = readU(0)
  let pos = attrs.headerType * 3
  const parts: Uint8Array[] = []
  let total = 0
  for (let b = 0; b < Math.max(1, nBlocks); b++) {
    const clen = readU(pos)
    pos += attrs.headerType
    const piece = await inflate(raw.subarray(pos, pos + clen))
    pos += clen
    parts.push(piece)
    total += piece.length
  }
  const bytes = new Uint8Array(total)
  let q = 0
  for (const p of parts) { bytes.set(p, q); q += p.length }
  const out = new DataView(bytes.buffer)
  const need = Math.min(nTuples > 0 ? nTuples * attrs.nComp : Math.floor(total / spec.bytes),
    Math.floor(total / spec.bytes))
  const vals = new Array<number>(need)
  for (let i = 0; i < need; i++) vals[i] = spec.get(out, i * spec.bytes)
  return vals
}

function attrsOf(da: Element): ArrayAttrs {
  return {
    type: da.getAttribute('type') || 'Float32',
    nComp: Number(da.getAttribute('NumberOfComponents') || '1'),
    format: (da.getAttribute('format') || 'ascii').toLowerCase(),
    headerType: (da.getAttribute('header_type') || 'UInt32') === 'UInt64' ? 8 : 4,
  }
}

/** 解一个 DataArray 元素。 */
async function decodeDataArray(da: Element, nTuples: number): Promise<number[]> {
  // 只取直接子文本节点。有些 DataArray 里还嵌着 InformationKey 子元素，
  // 直接读 textContent 会把那段标记也当成载荷，base64 解不动。
  let payload = ''
  for (const node of Array.from(da.childNodes)) {
    if (node.nodeType === 3) payload += node.nodeValue || ''
  }
  if (!payload.trim()) throw new Error('DataArray 里没有载荷文本')
  return decodeArray(attrsOf(da), payload, nTuples)
}

/** 从 Lines 的 connectivity 与 offsets 还原一条闭环的点序。 */
export function loopFromConnectivity(conn: number[], offs: number[]): number[] {
  const n = conn.length ? Math.max(...conn) + 1 : 0
  const adj: number[][] = Array.from({ length: n }, () => [])
  let start = 0
  for (const end of offs) {
    const cell = conn.slice(start, end)
    for (let k = 0; k + 1 < cell.length; k++) {
      adj[cell[k]].push(cell[k + 1])
      adj[cell[k + 1]].push(cell[k])
    }
    start = end
  }
  const bad = adj.findIndex((a) => a.length !== 2)
  if (bad >= 0) throw new Error('第 ' + bad + ' 个点的度数不是 2，闭环拼不出来')
  const order = [0]
  let prev = -1
  let cur = 0
  for (let guard = 0; guard < n + 2; guard++) {
    const nxt = adj[cur][0] === prev ? adj[cur][1] : adj[cur][0]
    if (nxt === 0) break
    order.push(nxt)
    prev = cur
    cur = nxt
  }
  if (order.length !== n) {
    throw new Error('闭环只走到 ' + order.length + ' 个点，总数是 ' + n)
  }
  return order
}

/**
 * 弧长归一化、前缘锚定的压力曲线，口径与 airfrans_to_web.py 逐行一致。
 *
 * 有两处容易写错的细节，必须照抄原实现，不能"顺手改对"：
 *   第一，坐标数组要先把首点再接一份到末尾，长度变成 N+1，滚动与取半都用这个长度；
 *     而压力数组只有 N 个，两者长度不同是原实现如此。
 *   第二，弧长参数 s 是把"首项为零的累加和"去掉最后一项得到的，
 *     也就是闭环的最后一段不参与参数化，s 的末项正好等于 1。
 * 这两点决定了前缘尖点附近少数几个点的取值，改了就与 Python 侧对不上。
 */
export function wallCp(
  pts: ArrayLike<number>, pw: ArrayLike<number>, uInf: number, n = 256, leShift = 1,
): number[] {
  const np = pw.length
  // 坐标：首点再接一份，长度 N+1
  const X = new Float64Array(np + 1)
  const Y = new Float64Array(np + 1)
  for (let i = 0; i < np; i++) {
    X[i] = pts[i * 2]
    Y[i] = pts[i * 2 + 1]
  }
  X[np] = pts[0]
  Y[np] = pts[1]
  let le = 0
  for (let i = 1; i <= np; i++) if (X[i] < X[le]) le = i
  le = (le + leShift) % np
  const nr = np + 1
  const Xr = new Float64Array(nr)
  const Yr = new Float64Array(nr)
  for (let i = 0; i < nr; i++) {
    const s = (i + le) % nr
    Xr[i] = X[s]
    Yr[i] = Y[s]
  }
  const P = new Float64Array(np)
  for (let i = 0; i < np; i++) P[i] = pw[(i + le) % np]
  const reverse = Yr[Math.floor(nr / 2)] < Yr[Math.floor(nr / 2) - 1]
  if (reverse) { Xr.reverse(); Yr.reverse(); P.reverse() }
  // 取前 N 个点当闭环
  const seg = new Float64Array(np)
  for (let i = 0; i < np; i++) {
    const j = (i + 1) % np
    seg[i] = Math.hypot(Xr[j] - Xr[i], Yr[j] - Yr[i])
  }
  const sArr = new Float64Array(np)
  let acc = 0
  for (let i = 0; i < np; i++) { sArr[i] = acc; acc += seg[i] }
  const last = sArr[np - 1] || 1
  const q = 0.5 * uInf * uInf
  const cp = new Float64Array(np)
  for (let i = 0; i < np; i++) {
    sArr[i] /= last
    cp[i] = P[i] / q
  }
  const out = new Array<number>(n)
  for (let k = 0; k < n; k++) {
    const t = k / (n - 1)
    let i = 0
    while (i < np - 2 && sArr[i + 1] < t) i++
    const t0 = sArr[i]
    const t1 = sArr[i + 1] ?? t0 + 1e-12
    const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0
    out[k] = cp[i] + f * ((cp[i + 1] ?? cp[i]) - cp[i])
  }
  return out
}

/** 从文件名解析来流速度与攻角。 */
export function conditionsFromName(name: string): { uInf: number, aoa: number, key: string } {
  const key = name.replace(/_aerofoil\.vtp$/i, '').replace(/\.vtp$/i, '')
  const parts = key.replace(/^airFoil2D_SST_/, '').split('_')
  return { uInf: Number(parts[0]) || 62, aoa: Number(parts[1]) || 0, key }
}

/**
 * 解析一个 AirfRANS 翼型文件。
 * fileName 用于取名与来流条件，buf 是文件内容。
 */
export async function parseAerofoilVtp(fileName: string, buf: ArrayBuffer): Promise<VtpSample> {
  const text = new TextDecoder('utf-8').decode(buf)
  const doc = new DOMParser().parseFromString(text, 'application/xml')
  if (doc.getElementsByTagName('parsererror').length) {
    throw new Error('不是合法的 XML 文件，可能不是 AirfRANS 的翼型文件')
  }
  const piece = doc.getElementsByTagName('Piece')[0]
  if (!piece) throw new Error('文件里没有 Piece 节点，可能不是 PolyData')
  const nPoints = Number(piece.getAttribute('NumberOfPoints') || '0')
  if (!nPoints) throw new Error('文件里没有点数')

  let pts: number[] | null = null
  let pw: number[] | null = null
  let conn: number[] | null = null
  let offs: number[] | null = null
  for (const child of Array.from(piece.children)) {
    const tag = child.tagName
    if (tag === 'Points') {
      const da = child.getElementsByTagName('DataArray')[0]
      if (da) pts = await decodeDataArray(da, nPoints)
    } else if (tag === 'PointData') {
      for (const da of Array.from(child.getElementsByTagName('DataArray'))) {
        if (da.getAttribute('Name') === 'p') pw = await decodeDataArray(da, nPoints)
      }
    } else if (tag === 'Lines' || tag === 'Polys' || tag === 'Strips' || tag === 'Verts') {
      // 同一个 Piece 里 Verts / Lines / Strips / Polys 四段都会带 connectivity 与 offsets，
      // 多数段是空的（本数据集 1000/1000 都是 Lines 有数据、其余三段是空的）。
      // 早期版本只判断标签名，后面出现的空 Polys 会把已经解出的 Lines 覆盖成空数组，
      // 于是闭环拼不出来。这里只接受非空的那一份，空的不许覆盖。
      for (const da of Array.from(child.getElementsByTagName('DataArray'))) {
        const nm = da.getAttribute('Name')
        if (nm === 'connectivity') {
          const got = await decodeDataArray(da, 0)
          if (got.length && (!conn || got.length > conn.length)) conn = got
        } else if (nm === 'offsets') {
          const got = await decodeDataArray(da, 0)
          if (got.length && (!offs || got.length > offs.length)) offs = got
        }
      }
    }
  }
  if (!pts) throw new Error('文件里没有坐标数组')
  if (!pw) throw new Error('文件里没有压力数组 p，这不是翼型表面文件')
  if (!conn || !offs) throw new Error('文件里没有轮廓连线，拼不出闭环')

  const order = loopFromConnectivity(conn, offs)
  const twin = new Float64Array(order.length * 2)
  const pwv = new Float64Array(order.length)
  order.forEach((src, i) => {
    twin[i * 2] = pts![src * 3]
    twin[i * 2 + 1] = pts![src * 3 + 1]
    pwv[i] = pw![src]
  })
  const { uInf, aoa, key } = conditionsFromName(fileName)
  if (!(uInf > 0)) throw new Error('文件名里没读到来流速度，请按 airFoil2D_SST_风速_攻角_... 命名')
  const cp256 = wallCp(twin, pwv, uInf)
  const outline: Array<[number, number]> = []
  for (let i = 0; i < order.length; i++) outline.push([twin[i * 2], twin[i * 2 + 1]])
  return { name: key, uInf, aoa, nPoints, cp256, outline }
}

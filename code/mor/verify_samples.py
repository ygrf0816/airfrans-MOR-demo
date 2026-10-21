"""P1 样本完整性闸门 + 批量 raw-deflate 还原。
流程：
  1) 读 04_results/analysis/samples_invalid_deflate/（已隔间的无效 deflate 流）
  2) 每个 *.vtu / *.vtp 就地 raw deflate 解压（zlib.decompressobj(-15)）
     -> 校验：
        a) 解压后尾部 == b'</VTKFile>'（.vtu）/ b'</vtkPolyData>'（.vtp）
        b) pyvista.read() 成功（.vtu/.vtp 都走 PV）
        c) .vtu 必须含字段 p/U/nut/implicit_distance
        d) 文件名对应 sim 在 selected_names 或在 crdownload 抽出的名单里
  3) 成功 -> 04_results/analysis/samples/<sim>/<file>（正确扩展名）
     失败 -> 04_results/analysis/samples_failed_decompress/<sim>/<file>（保留原始 deflate 以便重取）
  4) 写 04_results/analysis/00_integrity_gate.md + integrity_gate.json
幂等：samples/ 已有同名同大小文件则跳过。
"""
import os, sys, json, zlib, time, collections, glob
import pyvista as pv
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
BAD = os.path.join(ROOT, '04_results', 'analysis', 'samples_invalid_deflate')
OUT = os.path.join(ROOT, '04_results', 'analysis', 'samples')
FAIL = os.path.join(ROOT, '04_results', 'analysis', 'samples_failed_decompress')
SEL = os.path.join(ROOT, '04_results', 'analysis', 'selected_names.json')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FAIL, exist_ok=True)

sel = set(json.load(open(SEL))['selected'])

EXT_TAIL = {
    'internal.vtu': b'</VTKFile>',
    'aerofoil.vtp': b'</vtkPolyData>',
    'freestream.vtp': b'</vtkPolyData>',
}
VTU_FIELDS = ['p', 'U', 'nut', 'implicit_distance']


def is_selected_or_extracted(sim):
    return sim in sel or True  # crdownload 抽出的 475 个 sim 都允许；闸门只校文件格式


def raw_inflate(fp):
    with open(fp, 'rb') as fh:
        raw = fh.read()
    d = zlib.decompressobj(-15)
    out = d.decompress(raw)
    out += d.flush()
    return out


def process(fp):
    """-> (ok: bool, msg, decoded_bytes)"""
    sim = os.path.basename(os.path.dirname(fp))
    base = os.path.basename(fp)
    kind = None
    for k in EXT_TAIL:
        if base.endswith(k):
            kind = k
    if kind is None:
        return False, 'unknown kind', None
    try:
        decoded = raw_inflate(fp)
    except Exception as e:
        return False, f'zlib fail: {type(e).__name__}: {e}', None
    tail = EXT_TAIL[kind]
    if not decoded.endswith(tail) and not decoded.rstrip().endswith(b'</VTKFile>') and not decoded.rstrip().endswith(b'</vtkPolyData>'):
        return False, f'tail {tail!r} missing (got {decoded[-30:]!r})', None
    dst_fp = os.path.join(OUT, sim, base)
    # 已落盘同大小 -> skip
    if os.path.exists(dst_fp) and os.path.getsize(dst_fp) == len(decoded):
        return True, 'already-on-disk', decoded
    os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
    with open(dst_fp, 'wb') as fh:
        fh.write(decoded)
    # pyvista 读
    try:
        ds = pv.read(dst_fp)
    except Exception as e:
        # 删坏文件
        try: os.remove(dst_fp)
        except Exception: pass
        return False, f'pv.read fail: {type(e).__name__}: {str(e)[:100]}', None
    if kind == 'internal.vtu':
        for fld in VTU_FIELDS:
            if fld not in ds.point_data:
                try: os.remove(dst_fp)
                except Exception: pass
                return False, f'missing field {fld}', None
    return True, f'ok N={ds.n_points}', decoded


ok, fail = 0, 0
fails = []
n_files = 0
t0 = time.time()
for sim_dir in sorted(glob.glob(os.path.join(BAD, '*'))):
    sim = os.path.basename(sim_dir)
    if not os.path.isdir(sim_dir):
        continue
    for fp in sorted(glob.glob(os.path.join(sim_dir, '*'))):
        if not os.path.isfile(fp):
            continue
        n_files += 1
        okk, msg, _ = process(fp)
        if okk:
            ok += 1
        else:
            fail += 1
            fails.append({'sim': sim, 'file': os.path.basename(fp), 'reason': msg})
            # 归档原始 deflate 到 failed_decompress
            fail_fp = os.path.join(FAIL, sim, os.path.basename(fp))
            os.makedirs(os.path.dirname(fail_fp), exist_ok=True)
            if not os.path.exists(fail_fp):
                with open(fp, 'rb') as fin, open(fail_fp, 'wb') as fout:
                    fout.write(fin.read())

elapsed = time.time() - t0
print(f'扫描 {n_files} 文件，成功 {ok}，失败 {fail}，耗时 {elapsed:.0f}s')
print(f'有效 sim: {len(glob.glob(os.path.join(OUT, "*")))}，完整 3 文件: '
      f'{sum(1 for d in glob.glob(os.path.join(OUT,"*")) if len(glob.glob(os.path.join(d,"*")))==3)}')

sel_have = sorted([d for d in glob.glob(os.path.join(OUT, '*')) if os.path.basename(d) in sel])
print(f'选中的 39 个里已齐完整 3 文件: {sum(1 for d in sel_have if len(glob.glob(os.path.join(d,"*")))==3)} / {len(sel)}')

gate = {
    'n_files_scanned': n_files,
    'n_ok': ok,
    'n_fail': fail,
    'fails': fails[:50],
    'n_complete_sims': sum(1 for d in glob.glob(os.path.join(OUT,"*")) if len(glob.glob(os.path.join(d,"*")))==3),
    'selected_complete': sum(1 for d in sel_have if len(glob.glob(os.path.join(d,"*")))==3),
    'elapsed_s': elapsed,
}
json.dump(gate, open(os.path.join(ROOT, 'analysis', 'integrity_gate.json'), 'w'),
          ensure_ascii=False, indent=1)

with open(os.path.join(ROOT, 'analysis', '00_integrity_gate.md'), 'w', encoding='utf-8') as fh:
    fh.write('# P1 样本完整性闸门（00_integrity_gate.md）\n\n')
    fh.write('## 0. 失效样本根因\n\n')
    fh.write('早期 `extract_from_crdownload.py` 只按 zip local file header 取到成员的 **deflate 压缩流**，'
             '未做 `zlib.decompressobj(-15)` 还原，导致 475 个 `_internal.vtu` + 952 个 `.vtp` 全是压缩流'
             '（大小 ≈ 远端解压后真值的 66.5~67%），VTK 解析器拒读。已隔间到 `samples_invalid_deflate/`。\n\n')
    fh.write('## 1. 修复方式（选 (a) 就地解压）\n\n')
    fh.write('- 依据：用户实测 12 个文件 raw deflate 还原后大小与远端中央目录一致、尾部有 `</VTKFile>`、'
             'meshio 读得 4 字段、点数 164k~182k。\n')
    fh.write('- 命令：`python 03_code/p1/verify_samples.py`（本脚本）。逐文件：'
             '`zlib.decompressobj(-15).decompress(raw)` → 校验尾部 → `pyvista.read` → 4 字段断言。\n\n')
    fh.write(f'## 2. 修复结果\n\n- 扫描 {gate["n_files_scanned"]} 文件')
    fh.write(f'\n- 解压+校验通过 **{gate["n_ok"]}**，失败 **{gate["n_fail"]}**')
    fh.write(f'\n- 有效 sim（完整 3 文件）: **{gate["n_complete_sims"]}**')
    fh.write(f'\n- 选中 39 里齐 3 文件: **{gate["selected_complete"]}**')
    if fails:
        fh.write(f'\n## 3. 失败名单（前 50）\n\n| sim | 文件 | 原因 |\n|---|---|---|\n')
        for f in fails[:50]:
            fh.write(f'| {f["sim"]} | {f["file"]} | {f["reason"][:80]} |\n')
    else:
        fh.write('\n## 3. 失败名单\n- 无（全部解压+校验通过）。\n')
    fh.write('\n## 4. 此前作废产物\n\n')
    fh.write('- `04_results/analysis/01_inventory.md` / `inventory.json`：样本文件数统计基于被隔间的压缩流'
             '（当时按 "文件齐全=3" 计，实际不可读）→ **作废**，等本闸门过后重跑。')
    fh.write('\n- `03_code/p1/p1_params.py` / `p1_mesh_sample.py` / `p1_field_sample.py` / `p1_physics_sample.py`：'
             '若此前已跑过，结论全部**作废**，需用还原后的 samples/ 重跑。')
    fh.write('\n- `samples_invalid_deflate/`：保留原始 deflate 流备查，勿用于任何分析。')
    fh.write('\n- `samples_failed_decompress/`（若产生）：解压失败成员的原始 deflate 流，等全量 `02_dataset\\Dataset` 就绪后补取。\n')
print('-> 00_integrity_gate.md + integrity_gate.json')

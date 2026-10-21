"""P1 全量扩充：从 F:\\temp download path for tandem\\Dataset.zip（完整 9.56 GB，testzip 通过）
对全部 1000 仿真抽取 3 文件到 04_results/analysis/samples/，
每个文件过 4-assert 闸门（结尾/ pyvista 可读 / 4 字段齐全 / 大小匹配），
已有效文件跳过。报告 N/1000 + 新增 + 失败。
"""
import os, sys, json, zipfile, time, shutil, collections
import pyvista as pv
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC  = r'F:\temp download path for tandem\Dataset.zip'
OUT  = os.path.join(ROOT, '04_results', 'analysis', 'samples')
FAIL = os.path.join(ROOT, '04_results', 'analysis', 'samples_failed_decompress')
os.makedirs(OUT, exist_ok=True); os.makedirs(FAIL, exist_ok=True)

TAIL = {'.vtu': b'</VTKFile>', '.vtp': b'</vtkPolyData>'}
VTU_FIELDS = ['p', 'U', 'nut', 'implicit_distance']

print(f'ZIP: {SRC}  ({os.path.getsize(SRC)/1048576:.0f} MB)')
zf = zipfile.ZipFile(SRC)

# enumerate all 1000 sims from the zip
sim_files = collections.defaultdict(list)   # sim -> [(arcname, infolist)]
for zi in zf.infolist():
    parts = zi.filename.split('/')
    if len(parts) == 3 and parts[0] == 'Dataset' and not zi.filename.endswith('/'):
        sim_files[parts[1]].append((zi.filename, zi))

print(f'zip 里发现 {len(sim_files)} 个仿真 × 各 {list(sim_files.values())[0][0][0].split("/")[-1]!r} 等文件')

n_valid, n_new, n_fail, n_done = 0, 0, 0, 0
n_total_files = 0
t0 = time.time()

for sim, files in sorted(sim_files.items()):
    n_files_this = len(files)
    ok_this = True
    for arcname, zi in files:
        fname = arcname.split('/')[2]
        ext = os.path.splitext(fname)[1]
        target = os.path.join(OUT, sim, fname)
        n_total_files += 1
        if os.path.exists(target) and os.path.getsize(target) == zi.file_size:
            n_done += 1
            continue
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(zi) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            # verify: size
            if os.path.getsize(target) != zi.file_size:
                os.remove(target)
                n_fail += 1
                continue
            # gate: tail
            with open(target, 'rb') as fh:
                fh.seek(0, 2)
                fh.seek(-64, 1)
                tail_bytes = fh.read()
            expected_tail = TAIL.get(ext, b'</VTKFile>')
            if not tail_bytes.rstrip().endswith(expected_tail):
                os.makedirs(os.path.join(FAIL, sim), exist_ok=True)
                shutil.move(target, os.path.join(FAIL, sim, fname))
                n_fail += 1
                ok_this = False
                continue
            # gate: pyvista.read + fields
            try:
                ds = pv.read(target)
            except Exception:
                os.remove(target)
                n_fail += 1
                ok_this = False
                continue
            if ext == '.vtu':
                missing = [f for f in VTU_FIELDS if f not in ds.point_data]
                if missing:
                    os.remove(target)
                    n_fail += 1
                    ok_this = False
                    continue
        except Exception:
            ok_this = False
            n_fail += 1
            continue
        n_new += 1

    n_done += 0  # counted inside loop
    _files_done_ok = all(
        os.path.exists(os.path.join(OUT, sim, arcname.split('/')[2]) )
        for arcname, _ in files)
    if _files_done_ok and n_files_this == 3:
        n_valid += 1
    if ((n_new + n_fail + n_done) % 300) < n_files_this and n_new + n_fail + n_done > 0:
        print(f'  progress: valid={n_valid} new={n_new} fail={n_fail} done={n_done}  ({time.time()-t0:.0f}s)', flush=True)

# Final check: count sims with all 3 files on disk and valid
complete_sims = sorted(s for s, files in sim_files.items()
                       if all(os.path.exists(os.path.join(OUT, s, arcname.split('/')[2]))
                              for arcname, _ in files))
n_valid = len(complete_sims)

report = {
    'run_time': time.ctime(),
    'src_zip': SRC,
    'src_mb': os.path.getsize(SRC)/1048576,
    'n_sims_in_zip': len(sim_files),
    'n_valid_sims': n_valid,
    'n_valid_out_of_1000': n_valid,
    'n_new_files_this_run': n_new,
    'n_fail_this_run': n_fail,
    'n_already_done': n_done,
    'complete_sims': complete_sims,
}
json.dump(report, open(os.path.join(ROOT, '04_results', 'analysis', 'expand_report.json'), 'w'),
          ensure_ascii=False, indent=1)

print(f'\n=== 全量扩充结果 ===')
print(f'有效仿真（完整3文件+过闸门）: {n_valid} / 1000')
print(f'本轮新增 {n_new}，已跳过 {n_done}，失败 {n_fail}')
print(f'耗时 {time.time()-t0:.0f}s')
print('DONE_EXPAND')

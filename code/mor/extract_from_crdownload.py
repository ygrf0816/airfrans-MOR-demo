"""从浏览器正在下的 .crdownload（= 已下载 zip 前缀）里按 local file header 顺序扫描，
只抽取 selected_names.json 命中的仿真（每个 3 文件）。已下载的才抽得到。
不依赖网络、不依赖中央目录。幂等、断点（跳过已落盘）。
"""
import os, sys, json, struct, zlib, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = r'F:\temp download path for tandem\未确认 157643.crdownload'
SEL = os.path.join(ROOT, '04_results', 'analysis', 'selected_names.json')
OUT = os.path.join(ROOT, '04_results', 'analysis', 'samples')
os.makedirs(OUT, exist_ok=True)
sel = set(json.load(open(SEL))['selected'])

sig_lfh = b'PK\x03\x04'
size = os.path.getsize(SRC)
print(f'SRC {size/1048576:.0f} MB, 选中 {len(sel)} 个仿真')

wanted = set(sel)
found = {}
t0 = time.time()
with open(SRC, 'rb') as fh:
    fh.seek(0)
    # 顺序扫描 local file headers；每次读到 LFH 签名就解析
    buf_start = 0
    CHUNK = 4 << 20
    # 用 mmap 更稳
import mmap
try:
    mm = mmap.mmap(-1, 0)  # 占位
except Exception:
    mm = None
try:
    with open(SRC, 'rb') as _f:
        mm = mmap.mmap(_f.fileno(), 0, access=mmap.ACCESS_READ)
except Exception as e:
    print('mmap failed:', e); mm = None

if mm:
    import re
    pos = 0
    n_match = 0
    n_lfh = 0
    while True:
        i = mm.find(b'PK\x03\x04', pos)
        if i < 0:
            break
        hdr = mm[i:i+30]
        if len(hdr) < 30:
            pos = i + 4; continue
        n_lfh += 1
        ver, flags, method, mtime, mdate, crc, csize, usize, nlen, elen = struct.unpack('<HHHHHIIIHH', hdr[4:30])
        if not (20 <= ver < 1000 and method in (0, 8) and csize < (1 << 31)):
            pos = i + 4; continue
        if nlen >= 255 or elen >= 255:
            pos = i + 4; continue
        name = mm[i+30:i+30+nlen]
        if not (0 < nlen < 512 and all(0x20 <= b < 0x80 for b in name)):
            pos = i + 4; continue
        fn = name.decode('utf-8')
        data_off = i + 30 + nlen + elen
        data_end = data_off + csize
        pos = i + 4
        n_match += 1
        parts = fn.split('/')
        if len(parts) == 3 and parts[0] == 'Dataset':
            sim = parts[1]
            target_dir = os.path.join(OUT, sim)
            target_fp = os.path.join(target_dir, parts[2])
            complete = data_end <= len(mm)
            if complete and not (os.path.exists(target_fp) and os.path.getsize(target_fp) == csize):
                os.makedirs(target_dir, exist_ok=True)
                fp = target_fp + '.part'
                with open(fp, 'wb') as out:
                    out.write(mm[data_off:data_end])
                os.replace(fp, target_fp)
                found.setdefault(sim, []).append(parts[2])
                if sim in wanted:
                    print(f'  [选中] {fn[-55:]:55s} {csize/1048576:5.2f} MB', flush=True)
    # 落盘：所有已抽 sim + 选中命中统计
    all_extracted = sorted(k for k, v in found.items() if len(v) == 3)
    n_wanted_have = sum(1 for s in wanted if s in set(all_extracted))
    print(f'\nLFH {n_lfh}，命中完整 3 文件 sim {len(all_extracted)}；其中选中 39 里有 {n_wanted_have} 个')
    json.dump({'extracted_complete': all_extracted,
               'selected_in_extracted': sorted(wanted & set(all_extracted)),
               'src_mb': size / 1048576, 'n_lfh_seen': n_lfh},
              open(os.path.join(ROOT, 'analysis', 'extracted_from_crdownload.json'), 'w'),
              ensure_ascii=False, indent=1)
else:
    print('无 mmap，跳过')

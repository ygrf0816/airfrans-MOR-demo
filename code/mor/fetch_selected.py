"""用 RangeFile + zipfile 从远端 9.5GB zip 抽 39 个仿真（每个 3 文件）到 04_results/analysis/samples/。
- 中央目录只读末尾（~12s），之后按文件定位
- 幂等：已存在文件跳过
- 网络失败重试 3 轮 x 4 次（RangeFile 已加强）
- 单文件 >500s 即视为卡死，重新 RangeFile + zipfile
"""
import os, sys, json, time, zipfile, collections
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROBE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'probe'))
sys.path.insert(0, PROBE)
from rangefile import RangeFile, URL, PROXY

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SEL = os.path.join(ROOT, '04_results', 'analysis', 'selected_names.json')
OUT = os.path.join(ROOT, '04_results', 'analysis', 'samples')
os.makedirs(OUT, exist_ok=True)
sel = json.load(open(SEL))['selected']
print(f'要抽 {len(sel)} 个仿真 x 3 文件', flush=True)


def open_zip():
    t0 = time.time()
    f = RangeFile(URL, proxies=PROXY)
    z = zipfile.ZipFile(f)
    names = z.namelist()
    print(f'zip 成员 {len(names)}，建索引 {time.time()-t0:.1f}s', flush=True)
    bydir = collections.defaultdict(list)
    for n in names:
        p = n.split('/')
        if len(p) == 3 and p[0] == 'Dataset':
            bydir[p[1]].append(n)
    return z, bydir


z, bydir = open_zip()
t_start = time.time()
ok_files = 0
t0 = time.time()
for sim in sel:
    members = bydir.get(sim, [])
    if not members:
        print(f'[!] 缺失 {sim}', flush=True)
        continue
    dst = os.path.join(OUT, sim)
    os.makedirs(dst, exist_ok=True)
    for mem in members:
        base = mem.split('/')[-1]
        fp = os.path.join(dst, base)
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            ok_files += 1
            continue
        t_file = time.time()
        try:
            data = z.read(z.getinfo(mem))
            tmp = fp + '.part'
            with open(tmp, 'wb') as fh:
                fh.write(data)
            os.replace(tmp, fp)
            ok_files += 1
            dt = time.time() - t_file
            print(f'  {sim[-38:]:38s} {base[-28:]:28s} {len(data)/1048576:6.2f} MB {dt:5.1f}s', flush=True)
            if dt > 500:
                print(f'[!] {sim} {base} stalled {dt}s, 重启会话', flush=True)
                z, bydir = open_zip()
        except Exception as e:
            print(f'[!] {sim} {base} FAILED {type(e).__name__}: {str(e)[:80]}，重启会话', flush=True)
            z, bydir = open_zip()
print(f'\n抽取完成 {ok_files} 个文件，耗时 {time.time()-t_start:.0f}s', flush=True)

"""抽 39 个选中仿真（3 文件/个）到 04_results/analysis/samples/。
优先读本地 02_dataset/Dataset.zip（若存在）；否则远端 Range。幂等。
"""
import os, sys, json, time, zipfile, collections
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SEL = os.path.join(ROOT, '04_results', 'analysis', 'selected_names.json')
OUT = os.path.join(ROOT, '04_results', 'analysis', 'samples')
ZIPLOCAL = os.path.join(ROOT, '02_dataset', 'Dataset.zip')
sel = json.load(open(SEL))['selected']
os.makedirs(OUT, exist_ok=True)

if os.path.exists(ZIPLOCAL):
    z = zipfile.ZipFile(ZIPLOCAL)
    names = z.namelist()
    print(f'本地 zip，成员 {len(names)}')
    readfn = lambda info: z.read(info)
else:
    sys.path.insert(0, os.path.join(ROOT, '03_code', 'probe'))
    from fetch_samples import RangeFile, URL, PROXY
    import zipfile as zf
    f = RangeFile(URL, proxies=PROXY)
    z = zf.ZipFile(f)
    names = z.namelist()
    print(f'远端 Range，成员 {len(names)}')
    readfn = lambda info: z.read(info)

bydir = collections.defaultdict(list)
for n in names:
    p = n.split('/')
    if len(p) == 3 and p[0] == 'Dataset':
        bydir[p[1]].append(n)

ok = miss = 0
for sim in sel:
    members = bydir.get(sim, [])
    if not members:
        miss += 1
        print(f'[!] 缺失 {sim}')
        continue
    dst = os.path.join(OUT, sim)
    os.makedirs(dst, exist_ok=True)
    for mem in members:
        fp = os.path.join(dst, mem.split('/')[-1])
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            continue
        t0 = time.time()
        data = readfn(z.getinfo(mem))
        with open(fp, 'wb') as fh:
            fh.write(data)
        ok += 1
        print(f'{sim[-36:]:36s} {mem.split("/")[-1][-30:]:30s} {len(data)/1048576:6.2f} MB {time.time()-t0:4.1f}s')
print(f'抽取 {ok} 文件，目录缺失 {miss}')

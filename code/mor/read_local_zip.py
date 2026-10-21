"""从 F:\temp download path for tandem\未确认 157643.crdownload 增量读取 zip 中央目录。
浏览器合并成 Dataset.zip 后即可用本地文件直接抽（更快更稳）。
用法：python 03_code/p1/read_local_zip.py <选中名列表json>
"""
import os, sys, json, zipfile, collections, shutil

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = r'F:\temp download path for tandem\未确认 157643.crdownload'
SEL = os.path.join(ROOT, '04_results', 'analysis', 'selected_names.json')
OUT = os.path.join(ROOT, '04_results', 'analysis', 'samples')
os.makedirs(OUT, exist_ok=True)
sel = json.load(open(SEL))['selected']
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print(f'SRC 大小 {os.path.getsize(SRC)/1048576:.0f} MB / 9534 MB')
z = zipfile.ZipFile(SRC)
names = z.namelist()
print(f'中央目录可读，成员 {len(names)}')
bydir = collections.defaultdict(list)
for n in names:
    p = n.split('/')
    if len(p) == 3 and p[0] == 'Dataset':
        bydir[p[1]].append(n)
for sim in sel:
    members = bydir.get(sim)
    if not members:
        print(f'[!] {sim} 不在 zip'); continue
    dst = os.path.join(OUT, sim); os.makedirs(dst, exist_ok=True)
    for mem in members:
        fp = os.path.join(dst, mem.split('/')[-1])
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            continue
        data = z.read(mem)
        open(fp, 'wb').write(data)
        print(f'  {sim[-38:]:38s} {len(data)/1048576:6.2f} MB')
print('done')

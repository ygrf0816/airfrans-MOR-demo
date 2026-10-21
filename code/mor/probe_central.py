"""验证 RangeFile 读远端 zip 中央目录的耗时（不抽全量）。"""
import sys, time, zipfile
sys.path.insert(0, r'E:\workspace\agnes-hackathon\10_airfrans\03_code\probe')
from rangefile import RangeFile, URL, PROXY
import requests
s = requests.Session()
r = s.head(URL, timeout=60, proxies=PROXY, allow_redirects=True)
print('HEAD status:', r.status_code, 'size:', r.headers.get('Content-Length'), flush=True)
t0 = time.time()
f = RangeFile(URL, proxies=PROXY)
print('HEAD via RangeFile:', f.size, f'{time.time()-t0:.1f}s', flush=True)
z = zipfile.ZipFile(f)
names = z.namelist()
print('namelist len:', len(names), f'{time.time()-t0:.1f}s total', flush=True)
print('first 3:', names[:3], flush=True)

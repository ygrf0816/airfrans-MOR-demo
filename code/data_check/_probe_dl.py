import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
try:
    import requests
    s = requests.Session()
    proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    URL = 'https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip'
    h = s.head(URL, timeout=30, proxies=proxies, allow_redirects=True)
    print('head', h.status_code, h.headers.get('Content-Length'))
    r = s.get(URL, headers={'Range': 'bytes=0-2097151'}, timeout=30, proxies=proxies, stream=True)
    print('get', r.status_code)
    total = 0
    for c in r.iter_content(1 << 20):
        total += len(c)
    print('downloaded bytes:', total)
except Exception as e:
    print('EXC', type(e).__name__, e)

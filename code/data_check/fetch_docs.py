"""拉取 airfrans 官方文档全部页面，保存为 HTML + 转纯文本，供通读。
输出：04_results/airfrans_docs/<name>.html / .txt
"""
import os, re, sys
import requests
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROXY = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
BASE = 'https://airfrans.readthedocs.io/en/latest/'
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '04_results', 'airfrans_docs'))
os.makedirs(OUT, exist_ok=True)

pages = [
    'index.html',
    'notes/introduction.html', 'notes/installation.html', 'notes/dataset.html',
    'notes/simulation.html', 'notes/visualization.html',
    'modules/root.html', 'modules/dataset.html', 'modules/sampling.html',
    'modules/naca.html',
    '_modules/airfrans/reorganize.html', '_modules/airfrans/simulation.html',
    '_modules/airfrans/dataset.html', '_modules/airfrans/sampling.html',
    '_modules/airfrans/naca_generator.html',
]


def strip_tags(html: str) -> str:
    html = re.sub(r'<script.*?</script>', '', html, flags=re.S | re.I)
    html = re.sub(r'<style.*?</style>', '', html, flags=re.S | re.I)
    # 标记代码块
    html = re.sub(r'<pre.*?>', '\n```\n', html, flags=re.S | re.I)
    html = re.sub(r'</pre>', '\n```\n', html, flags=re.S | re.I)
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
    html = re.sub(r'</p>', '\n', html, flags=re.I)
    html = re.sub(r'<li[^>]*>', '\n  - ', html, flags=re.I)
    html = re.sub(r'<h[1-6]>', '\n## ', html, flags=re.I)
    html = re.sub(r'<[^>]+>', '', html)
    html = re.sub(r'&lt;', '<', html).replace('&gt;', '>').replace('&amp;', '&')
    html = re.sub(r'&quot;', '"', html).replace('&nbsp;', ' ')
    html = re.sub(r'\n{3,}', '\n\n', html)
    return html.strip()


s = requests.Session()
for rel in pages:
    name = rel.replace('/', '_')
    u = BASE + rel
    for attempt in range(3):
        try:
            r = s.get(u, proxies=PROXY, timeout=30)
            if r.status_code == 404:
                print(f'404  {rel}  (跳过)')
                break
            r.raise_for_status()
            with open(os.path.join(OUT, name + '.html'), 'w', encoding='utf-8') as f:
                f.write(r.text)
            with open(os.path.join(OUT, name + '.txt'), 'w', encoding='utf-8') as f:
                f.write(strip_tags(r.text))
            print(f'OK   {rel:42s} html={len(r.text):7d}  txt={os.path.getsize(os.path.join(OUT, name+".txt"))}')
            break
        except Exception as e:
            print(f'retry{attempt} {rel}: {type(e).__name__} {e}')
print('done ->', OUT)

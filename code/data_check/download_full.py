"""全量下载 AirfRANS 9.5GB -> data/Dataset.zip，走 proxy，断点续传 + 卡死自动重连。
用法：python 03_code/p0/download_full.py
完成后自动解压到 data/Dataset/。下载中断后重跑会断点续传。
"""
import os, sys, time, hashlib, zipfile
import requests

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

URL = 'https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip'
PROXY = 'http://127.0.0.1:7890'
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(ROOT, 'data'); os.makedirs(OUT, exist_ok=True)
ZIP = os.path.join(OUT, 'Dataset.zip')
RESUME = ZIP + '.part'
BLOCK = 4 << 20          # 4 MB 分块，降低单包卡死概率
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 60        # 每次读 4MB 的超时


def log(*a):
    print(*a, flush=True)


def get_total():
    s = requests.Session()
    h = s.head(URL, timeout=CONNECT_TIMEOUT,
               proxies={'http': PROXY, 'https': PROXY}, allow_redirects=True)
    return int(h.headers['Content-Length'])


def download():
    total = get_total()
    log(f'目标大小 {total/1048576:.0f} MB')
    start = os.path.getsize(RESUME) if os.path.exists(RESUME) else 0
    log(f'断点 {start/1048576:.1f} MB / {total/1048576:.0f} MB')
    s = requests.Session()
    proxies = {'http': PROXY, 'https': PROXY}
    done = start
    stall = 0
    t_last = time.time()
    while done < total:
        try:
            with s.get(URL, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                       proxies=proxies,
                       headers={'Range': f'bytes={done}-{total-1}'}) as r:
                r.raise_for_status()
                for chunk in r.iter_content(BLOCK):
                    if not chunk:
                        continue
                    with open(RESUME, 'ab') as f:
                        f.write(chunk)
                    done += len(chunk)
                    t_last = time.time(); stall = 0
                    if done % (64 * 1048576) < BLOCK:
                        el = time.time()
                        spd = (done - start) / max(el - t_last, 1e-3)
                        log(f'  {done/1048576:9.1f} / {total/1048576:.0f} MB')
                        t_last = time.time()
        except Exception as e:
            log(f'  异常 {type(e).__name__}: {e} @ {done/1048576:.1f} MB，2s 后重连')
            time.sleep(2)
            continue
    os.replace(RESUME, ZIP)
    log(f'下载完成 {os.path.getsize(ZIP)/1048576:.0f} MB')
    h = hashlib.sha256()
    with open(ZIP, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    log(f'sha256: {h.hexdigest()}')


def extract():
    DEST = os.path.join(OUT, 'Dataset')
    if os.path.exists(DEST):
        log(f'解压目录已存在 {DEST}')
        return
    t0 = time.time()
    log('解压 ...')
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DEST)
    log(f'解压完成 {time.time()-t0:.0f}s -> {DEST}')


if __name__ == '__main__':
    download()
    extract()
    log('ALL DONE')

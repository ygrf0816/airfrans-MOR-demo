"""诊断为什么 os.replace(tmp, dst) 在 bridge prefetch 里 WinError 183 失败。
模拟：建临时 .part，再 os.replace 到目标位置，测目标是否已存在。
"""
import os, sys, tempfile, pathlib, time
A = r'E:\workspace\agnes-hackathon\10_airfrans\04_results\analysis\samples'
name = 'airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32'
dst = os.path.join(A, name, name + '_internal.vtu')
tmp = dst + '.part'
print('dst:', dst)
print('dst exists:', os.path.exists(dst))
print('tmp exists:', os.path.exists(tmp))
print('dir exists:', os.path.isdir(os.path.dirname(dst)))
# 列 samples 目录
entries = []
if os.path.isdir(A):
    for d in os.listdir(A):
        fp = os.path.join(A, d)
        if os.path.isdir(fp):
            entries.append((d, os.listdir(fp)))
for d, fs in entries:
    print(f'  {d}: {fs}')
# 再尝试 replace 一次
try:
    with open(tmp, 'wb') as fh:
        fh.write(b'test')
    os.replace(tmp, dst)
    print('replace OK, dst now exists:', os.path.exists(dst))
except Exception as e:
    print('replace FAILED:', type(e).__name__, e)

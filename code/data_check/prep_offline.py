"""生成工作站 pytorch env (python 3.10, manylinux) 的离线 wheel 包。
输出到 04_results/whl/，供 scp 上传 + pip --no-index 安装。
本机 py 3.12 下 pip download 的 manylinux wheel 对 3.10 兼容（cp310/abi3/py3-none-any）。
vtk 用 py3-none-any 不行（vtk 是二进制），需按 py 版本选 wheel；pip download -r 指定目标 py。
"""
import os, sys, subprocess

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
WHL = os.path.join(ROOT, '04_results', 'whl'); os.makedirs(WHL, exist_ok=True)
REQ = os.path.join(WHL, 'requirements.txt')
with open(REQ, 'w') as f:
    f.write("airfrans\n")
    f.write("pyvista\n")
    f.write("vtk\n")
    f.write("numpy\n")
    f.write("tqdm\n")
    f.write("matplotlib\n")
# 用 --python-version 3.10 为目标环境下载（避免本机 3.12 装不上的二进制）
cmd = [sys.executable, '-m', 'pip', 'download', '-r', REQ, '-d', WHL,
       '--python-version', '310', '--only-binary=:all:',
       '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple']
print(' '.join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-3000:])
print('STDERR:', r.stderr[-1500:])
print('exit', r.returncode)
import glob
tot = 0
for f in sorted(glob.glob(os.path.join(WHL, '*.whl'))):
    mb = os.path.getsize(f)//1048576
    tot += mb
    print(f'  {os.path.basename(f):60s} {mb} MB')
print(f'合计 {tot} MB')

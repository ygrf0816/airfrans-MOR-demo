#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工作站（10.10.20.87 / zch）远程调用封装 —— 绕开 AGH shell 白名单环境导致的 ssh 255 问题。

背景（2026-10-01 定位）
  Agnes Harness 的 shell 工具只传白名单环境变量（PATH/HOME/LANG/LC_ALL/TMPDIR/TEMP/TMP/
  SystemRoot/COMSPEC/PATHEXT 等），**不含 `ProgramData`**。
  Windows 版 OpenSSH 在启动早期会读 `getenv("ProgramData")`，取到 NULL 就直接退出 255、
  且不打印任何信息（连 `ssh -V` 都零输出）。这不是网络/密钥/known_hosts/杀软问题：
  只要把 `ProgramData` 放回子进程环境，`ssh -V` 与真实连接立刻恢复正常。

  关键实验（同一 shell 内控制变量）：
    whitelist env                -> exit 255, 无输出
    whitelist env + ProgramData  -> exit 0,  打印版本 / 连上工作站
    ProgramData=C:\\Nonexistent   -> exit 0  （变量"存在"即可，值不校验）
    full env - ProgramData       -> exit 255 （从完整环境里删掉它即复现故障）

用法
    python 03_code/station/remote.py "hostname; nproc"
    python 03_code/station/remote.py --script 09_tmp/job.sh     # 上传并执行
    python 03_code/station/remote.py --put local.txt /home/zch/airfrans/data/

远端工作区
    /home/zch/airfrans          ← 当前项目（AirfRANS），所有操作在此进行
    /home/zch/heat3d_hackathon  ← 旧 Heat3D 项目，已归档（见其 ARCHIVED.md），勿用
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HOST = "10.10.20.87"
USER = "zch"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
SSH_DIR = r"C:\Users\Administrator\.ssh"
KEY = os.path.join(SSH_DIR, "id_rsa")
KNOWN_HOSTS = os.path.join(SSH_DIR, "known_hosts")
REMOTE_TMP = "/home/zch/airfrans"


def _base_env() -> dict:
    """复制当前环境，并强制补齐 OpenSSH(Windows) 启动所需的 ProgramData。"""
    env = dict(os.environ)
    if not env.get("ProgramData"):
        env["ProgramData"] = r"C:\ProgramData"
    if not env.get("HOME"):
        env["HOME"] = SSH_DIR.rsplit("\\", 2)[0]
    env.setdefault("SystemRoot", r"C:\Windows")
    env.setdefault("COMSPEC", r"C:\Windows\system32\cmd.exe")
    return env


def _ssh_opts() -> list:
    return [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-i", KEY,
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
    ]


def run(command: str, capture: bool = True) -> subprocess.CompletedProcess:
    """在工作站上执行一条命令（command 会被远端 shell 解析）。"""
    argv = [SSH, *_ssh_opts(), f"{USER}@{HOST}", command]
    return subprocess.run(
        argv, env=_base_env(), capture_output=capture, text=True,
        encoding="utf-8", errors="replace",
    )


def put(local: str, remote_dir: str = REMOTE_TMP) -> subprocess.CompletedProcess:
    argv = [SCP, *_ssh_opts(), local, f"{USER}@{HOST}:{remote_dir}"]
    return subprocess.run(
        argv, env=_base_env(), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def get(remote: str, local_dir: str = ".") -> subprocess.CompletedProcess:
    argv = [SCP, *_ssh_opts(), f"{USER}@{HOST}:{remote}", local_dir]
    return subprocess.run(
        argv, env=_base_env(), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def run_script(local_script: str) -> subprocess.CompletedProcess:
    """先 scp 上传，再远端 bash 执行（避免多层引号被 PowerShell/bash 互吞）。"""
    name = os.path.basename(local_script)
    remote = f"{REMOTE_TMP}/{name}"
    cp = put(local_script)
    if cp.returncode != 0:
        return cp
    return run(f"bash {remote}")


def main() -> int:
    ap = argparse.ArgumentParser(description="远程调用工作站（自动修复 ProgramData 环境问题）")
    ap.add_argument("command", nargs="?", help="要在工作站执行的命令")
    ap.add_argument("--script", help="本地 .sh 脚本：上传后远端 bash 执行")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE_DIR"), help="上传文件")
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL_DIR"), help="下载文件", default=None)
    args = ap.parse_args()

    if args.script:
        r = run_script(args.script)
    elif args.put:
        r = put(args.put[0], args.put[1])
    elif getattr(args, 'get', None):
        r = get(args.get[0], args.get[1] if len(args.get)>1 else '.')
    elif args.command:
        r = run(args.command)
    else:
        r = run("hostname; nproc; nvidia-smi -L; ls -d " + REMOTE_TMP)

    # Windows 控制台默认 GBK，远端返回的 UTF-8/BOM 可能编不出来 → 强制 utf-8 + replace
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if r.stdout:
        sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())

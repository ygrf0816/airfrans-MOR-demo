#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""演示作品的静态服务器。

为什么不用 `python -m http.server`：Python 3.9 之前的标准库里没有
`application/wasm` 这条 MIME 记录，.wasm 会被发成 application/octet-stream。
ONNX Runtime Web 在这种情况下不会退回用 ArrayBuffer 加载，而是直接失败，
表现是模型面板全部显示「待接入」、压力曲线与流场云图全空，且不给出原因。
这个脚本自己把 MIME 写死，因此不挑 Python 版本。

用法：
    python serve.py            # 默认 http://127.0.0.1:8080
    python serve.py 9000       # 指定端口
    python serve.py 9000 0.0.0.0   # 指定端口与监听地址
"""
import http.server
import os
import socketserver
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 站点根目录按可能的摆放位置依次探测：
#   本目录/site     —— 提交目录里的摆法
#   本目录/dist     —— 交付包里 01_demo_site/ 的摆法
#   本目录          —— 脚本被直接放进站点里
ROOT = HERE
for cand in ("site", "dist"):
    p = os.path.join(HERE, cand)
    if os.path.isfile(os.path.join(p, "index.html")):
        ROOT = p
        break

# 关键：这几条不能靠系统 MIME 表，写死。
EXTRA = {
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
    ".onnx": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def guess_type(self, path):
        ext = os.path.splitext(str(path))[1].lower()
        if ext in EXTRA:
            return EXTRA[ext]
        return super().guess_type(path)

    def end_headers(self):
        # 不缓存，改完文件刷新就能看到
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), Handler) as httpd:
        print("演示作品已启动： http://%s:%d/" % (host, port))
        print("站点目录： %s" % ROOT)
        print("按 Ctrl+C 停止。")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

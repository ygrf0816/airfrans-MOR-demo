# -*- coding: utf-8 -*-
"""把阶段报告转成**图内嵌的 HTML**（图片转 base64），任何浏览器打开都能看到图。

为什么需要它：Markdown 里的相对路径在不同查看器（Typora / VS Code / 各类 Web 预览 /
直接点开文件）行为不一致，容易"看不到图"。HTML 把图片直接塞进文件，**零外部依赖**。

用法：python 03_code/inversion/report_to_html.py
产物：04_results/inversion/EXPERIMENT_REPORT.html
"""
import base64
import os
import re

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DIR = os.path.join(ROOT, "04_results", "inversion")
SRC = os.path.join(DIR, "EXPERIMENT_REPORT.md")
OUT = os.path.join(DIR, "EXPERIMENT_REPORT.html")

CSS = """
<style>
  html { -webkit-text-size-adjust: 100%; }
  body { margin: 0 auto; max-width: 980px; padding: 32px 28px 80px;
         font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
         font-size: 16px; line-height: 1.85; color: #1b1b1b; background: #fff; }
  h1 { font-size: 26px; line-height: 1.5; border-bottom: 2px solid #d0d7de; padding-bottom: 12px; }
  h2 { font-size: 22px; margin-top: 40px; border-bottom: 1px solid #e3e3e3; padding-bottom: 6px; }
  h3 { font-size: 18px; margin-top: 28px; }
  h4 { font-size: 16px; margin-top: 22px; color: #333; }
  table { border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 14.5px; }
  th, td { border: 1px solid #d8dde3; padding: 7px 10px; text-align: left; vertical-align: top; }
  th { background: #f3f6fa; }
  img { max-width: 100%; height: auto; display: block; margin: 18px auto 6px;
        border: 1px solid #e6e6e6; border-radius: 4px; }
  code { background: #f4f6f8; padding: 1px 5px; border-radius: 3px;
         font-family: Consolas, "Courier New", monospace; font-size: 14px; }
  pre { background: #f6f8fa; padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
  blockquote { border-left: 4px solid #cfd8e3; margin: 12px 0; padding: 4px 14px;
               background: #fafbfc; color: #3b3b3b; }
  hr { border: none; border-top: 1px solid #e3e3e3; margin: 32px 0; }
  strong { color: #0b3d91; }
</style>
"""


def embed(md_text):
    """把 ![alt](file.png) 换成 base64 内嵌的 <img>。"""
    miss = []

    def repl(m):
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return m.group(0)
        full = os.path.join(DIR, path)
        if not os.path.isfile(full):
            miss.append(path)
            return m.group(0)
        b64 = base64.b64encode(open(full, "rb").read()).decode("ascii")
        return ('<img alt="%s" src="data:image/png;base64,%s">' % (alt, b64))

    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl, md_text)
    return md_text, miss


def main():
    text = open(SRC, encoding="utf-8").read()
    text, miss = embed(text)
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    doc = ("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
           "<title>反向设计阶段报告（图内嵌）</title>%s</head><body>%s</body></html>"
           % (CSS, html))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    n_img = doc.count("data:image/png;base64,")
    print("->", OUT, "（%.1f MB，内嵌图片 %d 张）" % (os.path.getsize(OUT) / 1e6, n_img))
    if miss:
        print("!! 以下图片没找到，未内嵌：", miss)


if __name__ == "__main__":
    main()

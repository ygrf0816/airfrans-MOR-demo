#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付包组装与容量核对。

两种用法：

  # 1 只看体积构成，不复制任何文件
  python 04_results/demo/tools/pack_check.py --audit

  # 2 组装交付包并核对
  python 04_results/demo/tools/pack_check.py --build --dest 07_submission/package

  # 3 只核对一个已经存在的目录或压缩包
  python 04_results/demo/tools/pack_check.py --dir 07_submission/package
  python 04_results/demo/tools/pack_check.py --zip 07_submission/package.zip

硬约束：总体积不超过 200 MB；不得出现原始数据集与中间缓存。
演示站点是自带的，dist 里已经带了 data 目录，所以不再单独复制一份资产，
避免同样的文件在包里出现两次。
"""
import argparse
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

LIMIT_MB = 200.0

# 出现这些词就报错。都是原始数据或中间缓存，不该进交付包。
# 注意不要把样本名 airFoil2D_SST_ 当排除词：反演的图与数据文件名里本来就带样本名，
# 那是要交的证据。数据集本身由 02_dataset 这个词拦下就够了。
FORBIDDEN = ["node_modules", "02_dataset", "field_cache/", "field_cache_475subsample",
             "redraw_data", "__pycache__", "vol_fields_1000", "bfield_basis.npz",
             ".vtu", ".vtp", ".pyc", ".pyo"]
# 单个文件上限。交付包里最大的文件是 ONNX Runtime 的 wasm，约 27 MB。
SINGLE_FILE_WARN_MB = 40.0
SINGLE_FILE_FAIL_MB = 60.0

# 交付包结构。每项是（目标相对路径，来源路径或通配，可选说明）
INCLUDE = [
    ("01_demo_site", "04_results/demo/app/dist", "可离线运行的演示站点，data 已内含"),
    ("02_paper", "04_results/paper/paper_zh.md", "中文论文源文件"),
    ("02_paper", "04_results/paper/paper_en.md", "英文论文源文件"),
    ("02_paper", "04_results/paper/paper_zh.docx", "中文论文 Word 版"),
    ("02_paper", "04_results/paper/paper_en.docx", "英文论文 Word 版"),
    ("02_paper", "04_results/paper/REFS.md", "参考文献"),
    ("02_paper", "04_results/paper/figs", "论文插图"),
    ("03_reports", "agent.md", "项目说明"),
    ("03_reports", "00_docs", "过程文档"),
    ("03_reports", "04_results/demo/00_PLAN.md", "演示作品计划"),
    ("03_reports", "04_results/demo/01_DATA_CONTRACT.md", "演示数据与模型契约"),
    ("03_reports", "04_results/demo/02_UI_DESIGN.md", "界面设计说明"),
    ("03_reports", "04_results/inversion/EXPERIMENT_REPORT.md", "阶段报告"),
    ("03_reports", "04_results/inversion/EXPERIMENT_LOG.md", "阶段执行记录"),
    ("03_reports", "04_results/inversion/REPORT_FIG_INDEX.md", "报告插图索引"),
    ("03_reports", "04_results/inversion/FIG_DATA_INDEX.md", "过程图索引"),
    ("03_reports", "04_results/inversion/00_PLAN_STAGE2.md", "阶段计划"),
    ("03_reports", "04_results/inversion/01_reachability.md", "可达性"),
    ("03_reports", "04_results/inversion/02_ga_results.md", "搜索解集"),
    ("03_reports", "04_results/inversion/03_weight_verify.md", "权重与口径核对"),
    ("03_reports", "04_results/inversion/05_anchor.md", "锚定核对"),
    ("03_reports", "04_results/inversion/06_inversion.md", "反演"),
    ("03_reports", "04_results/inversion/07_cases.md", "案例"),
    ("03_reports", "04_results/inversion/grad_check.md", "梯度可导性核对"),
    ("03_reports", "04_results/prediction/EXPERIMENT_REPORT.md", "预测阶段报告"),
    ("03_reports", "04_results/explain/EXPERIMENT_REPORT.md", "可解释性分析报告"),
    ("04_code", "03_code", "全部代码，探针依赖与缓存已排除"),
    ("05_evidence", "04_results/explain", "可解释性分析的落盘数据与配图"),
    ("05_evidence", "04_results/inversion/results", "全部原始结果数据"),
    ("05_evidence", "04_results/inversion/figs", "过程图"),
    ("05_evidence", "04_results/prediction/runs_full_B_official_eval_metrics.json", "曲线模型指标"),
    ("05_evidence", "04_results/prediction/runs_full_B_field_eval_metrics.json", "场模型指标"),
    ("05_evidence", "04_results/prediction/runs_full_A_eval_metrics.json", "图网络指标"),
    ("05_evidence", "04_results/prediction/benchmark_official.json", "官方基线"),
    ("05_evidence", "04_results/demo/data_remote/export_check.json", "模型导出一致性核对"),
    ("05_evidence", "04_results/demo/data_remote/trim_check.json", "基与网格核对"),
    ("05_evidence", "04_results/demo/data_remote/local_check.json", "本机资产核对"),
    ("05_evidence", "04_results/demo/data_remote/verify_demo_assets.json", "演示资产总核对"),
    ("05_evidence", "04_results/demo/screenshots", "五个页面的实际渲染截图"),
    ("06_reproduction", "04_results/demo/tools", "演示资产导出脚本"),
    ("06_reproduction", "04_results/demo/run_export.sh", "远端导出驱动脚本"),
    ("07_records", "07_submission/records", "对话记录与工具调用记录"),
]

# 目录下要排除的子路径（相对 ROOT 或相对源目录的片段）
EXCLUDE_PARTS = ["__pycache__", "node_modules", ".pytest_cache", ".mypy_cache",
                 "probe/libs", "probe/libs_meshio"]
EXCLUDE_SUFFIX = [".pyc", ".pyo"]
# 03_code 下额外排除的探针残留
EXCLUDE_PREFIX_REL = ["03_code/probe/airFoil2D_SST_"]


def is_excluded(rel):
    rel = rel.replace("\\", "/")
    for p in EXCLUDE_PARTS:
        if "/" + p + "/" in "/" + rel + "/":
            return p
    for pre in EXCLUDE_PREFIX_REL:
        if rel.startswith(pre):
            return pre
    for s in EXCLUDE_SUFFIX:
        if rel.endswith(s):
            return s
    return None


def walk_sizes(src):
    """返回 (字节数, 文件数, 最大文件, 文件列表)。src 可以是文件或目录。"""
    total, count, biggest, files = 0, 0, ("", 0), []
    src = src.replace("\\", "/")
    if os.path.isfile(src):
        sz = os.path.getsize(src)
        return sz, 1, (os.path.basename(src), sz), [(src, os.path.basename(src))]
    for r, ds, fs in os.walk(src):
        ds[:] = [d for d in ds if not is_excluded(os.path.relpath(os.path.join(r, d), ROOT))]
        for f in fs:
            p = os.path.join(r, f)
            rel = os.path.relpath(p, ROOT)
            if is_excluded(rel):
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            total += sz
            count += 1
            files.append((p, os.path.relpath(p, src)))
            if sz > biggest[1]:
                biggest = (rel, sz)
    return total, count, biggest, files


def audit():
    print("=" * 78)
    print("交付包体积构成（只统计，不复制）")
    print("=" * 78)
    rows = []
    for cat, src, note in INCLUDE:
        p = os.path.join(ROOT, src)
        if not os.path.exists(p):
            print("  %-16s %-58s 缺席" % (cat, src))
            continue
        total, count, biggest, _ = walk_sizes(p)
        rows.append((cat, src, total, count, biggest, note))
        print("  %-16s %10.2f MB  %5d 个文件  最大 %-34s %s"
              % (cat, total / 1048576, count,
                 os.path.basename(biggest[0])[:34], note))
    # 按类别汇总
    agg = {}
    for cat, src, total, count, _, _ in rows:
        a = agg.setdefault(cat, [0, 0])
        a[0] += total
        a[1] += count
    print("-" * 78)
    grand, grand_n = 0, 0
    for cat in sorted(agg):
        print("  %-16s %10.2f MB  %5d 个文件" % (cat, agg[cat][0] / 1048576, agg[cat][1]))
        grand += agg[cat][0]
        grand_n += agg[cat][1]
    print("-" * 78)
    print("  合计 %10.2f MB  %5d 个文件，上限 %.0f MB，余量 %.2f MB"
          % (grand / 1048576, grand_n, LIMIT_MB, LIMIT_MB - grand / 1048576))
    print()
    print("参考体积（不打包的部分）")
    for label, p in (("核心代码（含探针残留）", "03_code"),
                     ("探针依赖库（应排除）", "03_code/probe/libs"),
                     ("原始数据集（应排除）", "02_dataset"),
                     ("全部结果目录（含大缓存）", "04_results")):
        fp = os.path.join(ROOT, p)
        if os.path.isdir(fp):
            t = sum(os.path.getsize(os.path.join(r, f))
                    for r, ds, fs in os.walk(fp) for f in fs
                    if os.path.exists(os.path.join(r, f)))
            print("  %-24s %10.2f MB" % (label, t / 1048576))
    return grand


def build(dest):
    if os.path.exists(dest):
        print("目标目录已存在，先清空：%s" % dest)
        shutil.rmtree(dest)
    os.makedirs(dest)
    print("组装到 %s" % dest)
    for cat, src, note in INCLUDE:
        p = os.path.join(ROOT, src)
        if not os.path.exists(p):
            print("  跳过（缺席）%s" % src)
            continue
        target = os.path.join(dest, cat, os.path.basename(src.rstrip("/")))
        if os.path.isdir(p):
            shutil.copytree(p, target,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", "node_modules", "*.pyc", "*.pyo",
                                "libs", "libs_meshio", "airFoil2D_SST_*"))
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(p, target)
        print("  复制 %-16s %s" % (cat, src))
    return check_dir(dest)


def check_dir(dest):
    total, count, biggest, files = walk_sizes(dest)
    problems = []
    for p, rel in files:
        base = os.path.basename(p)
        for bad in FORBIDDEN:
            if bad in rel.replace("\\", "/") or bad in base:
                problems.append("命中排除词 %s：%s" % (bad, rel))
    if total / 1048576 > LIMIT_MB:
        problems.append("总体积 %.2f MB 超过上限 %.0f MB" % (total / 1048576, LIMIT_MB))
    if biggest[1] / 1048576 > SINGLE_FILE_FAIL_MB:
        problems.append("最大单文件 %.2f MB 超过 %.0f MB，确认它是不是原始数据"
                        % (biggest[1] / 1048576, SINGLE_FILE_FAIL_MB))
    print("=" * 78)
    print("交付包核对：%s" % dest)
    print("  文件数 %d" % count)
    print("  总体积 %.2f MB（上限 %.0f MB，余量 %.2f MB）"
          % (total / 1048576, LIMIT_MB, LIMIT_MB - total / 1048576))
    print("  最大单文件 %.2f MB  %s" % (biggest[1] / 1048576, biggest[0]))
    if biggest[1] / 1048576 > SINGLE_FILE_WARN_MB:
        print("  注意：最大单文件超过 %.0f MB，确认它是必要的资产" % SINGLE_FILE_WARN_MB)
    if problems:
        print("  发现 %d 个问题：" % len(problems))
        for x in problems[:20]:
            print("    %s" % x)
    else:
        print("  没有发现排除词命中，体积达标")
    print("PACK_CHECK_%s" % ("FAIL" if problems else "OK"))
    rep = {"dest": dest, "files": count, "bytes": int(total),
           "totalMb": round(total / 1048576, 2), "limitMb": LIMIT_MB,
           "largest": {"path": biggest[0], "bytes": int(biggest[1])},
           "problems": problems, "passed": not problems}
    out = os.path.join(dest, "_pack_check.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rep


def check_zip(zp):
    z = zipfile.ZipFile(zp)
    total = sum(i.file_size for i in z.infolist())
    count = len([i for i in z.infolist() if not i.is_dir()])
    biggest = max(z.infolist(), key=lambda i: i.file_size)
    problems = []
    for i in z.infolist():
        for bad in FORBIDDEN:
            if bad in i.filename:
                problems.append("命中排除词 %s：%s" % (bad, i.filename))
    if total / 1048576 > LIMIT_MB:
        problems.append("解压后总体积 %.2f MB 超过上限" % (total / 1048576))
    print("压缩包 %s" % zp)
    print("  文件数 %d，解压后 %.2f MB，压缩包本身 %.2f MB"
          % (count, total / 1048576, os.path.getsize(zp) / 1048576))
    print("  最大单文件 %.2f MB  %s" % (biggest.file_size / 1048576, biggest.filename))
    print("  问题 %d 个" % len(problems))
    for x in problems[:10]:
        print("    %s" % x)
    print("PACK_CHECK_%s" % ("FAIL" if problems else "OK"))
    return {"zip": zp, "files": count, "bytes": total, "problems": problems,
            "passed": not problems}


def main():
    ap = argparse.ArgumentParser(description="交付包组装与容量核对")
    ap.add_argument("--audit", action="store_true", help="只看体积构成")
    ap.add_argument("--build", action="store_true", help="组装交付包")
    ap.add_argument("--dest", default="07_submission/package", help="组装目标目录")
    ap.add_argument("--dir", help="核对一个已存在的目录")
    ap.add_argument("--zip", help="核对一个已存在的压缩包")
    args = ap.parse_args()
    if args.zip:
        rep = check_zip(args.zip)
    elif args.dir:
        rep = check_dir(os.path.abspath(args.dir))
    elif args.build:
        rep = build(os.path.abspath(args.dest))
    else:
        audit()
        return 0
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

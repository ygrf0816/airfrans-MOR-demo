# 任务：补跑一次一致性自检，再把三个权重取回来（第 3 轮）

上一轮的结果我已经核对过，事实如下，不用再验证。

## 已经核实的事实

- 两个 ONNX 模型**已经导出成功并取回本机**，形状正确。
  `b_cp.onnx` 1214583 字节，输入 `x` 形状 1 乘 133，输出 `alpha` 形状 1 乘 8。
  `b_field.onnx` 1337950 字节，输入 `x` 形状 1 乘 133，输出 `alpha` 形状 1 乘 128。
  这两个文件这一轮**不要重新导出**。
- 场基那一半也全部正确，`trim_check.json` 里四条对照的斜率都落在 1 附近。
- 上一轮崩溃的原因在我写的脚本里。导出的图批次维固定成 1（浏览器一次只算一条，
  这是我们要的），但自检一次喂了 8 条，所以 onnxruntime 报
  `Got invalid dimensions for input: x ... Got: 8 Expected: 1`。
  **这个 bug 我已经在本地修好了**，现在改成逐条喂入。
- 崩溃发生在写出两个 ONNX 之后、写出元信息之前，所以少了
  `cp_basis_meta.json` 与 `export_check.json` 这两个文件。这一轮就是来补它们的。

## 这一轮要做的事

### 第 1 步 传两个文件到远端

- 本地 `04_results/demo/tools/export_onnx.py` 传到远端 `04_results/demo/tools/`
- 本地 `04_results/demo/run_export_check.sh` 传到远端 `04_results/demo/`

`trim_field_basis.py` 这一轮不用传也不用跑，它已经对了。

### 第 2 步 起后台任务

远端执行下面这一条，命令会立刻返回。

```
cd /home/zch/airfrans && nohup bash 04_results/demo/run_export_check.sh > 04_results/demo/run_export_check.out 2>&1 &
```

### 第 3 步 等一会再查，只查一次

```
tail -n 60 /home/zch/airfrans/04_results/demo/run_export_check.out ; ls -l /home/zch/airfrans/04_results/demo/data_remote ; ls -l /home/zch/airfrans/04_results/demo/data_remote/weights
```

看到 `ALL_DONE` 就是完成了。看到 `Traceback` 就抄报错原文报告，**不要改脚本**。
还在跑就报告日志末尾，下一轮再取。

### 第 4 步 取回

用 `--get` 取回本机 `04_results/demo/data_remote/`：

```
cp_basis_meta.json  export_check.json  export_check.log
weights/full_B_official_best_model.pt
weights/full_B_field_best_model.pt
weights/full_A_best_model.pt
```

三个权重加起来大约三 MB，逐个取。

### 第 5 步 本机核对

把 `09_tmp/check_web_assets.py` 再跑一次，这次第 2 条与第 6 条也应该过。另外加两条：

1. `export_check.json` 里 `b_cp` 与 `b_field` 的 `maxAbsDiff` 都小于 1e-4，
   `allPassed` 是 true。把两个最大绝对差打印出来。
2. 本机用 onnxruntime 拿 `cp_basis_meta.json` 的 `smean/sstd/gmean/gstd` 构造
   一条 1 乘 133 的输入，跑 `b_cp.onnx`，把输出 8 个数打印出来，
   再乘 `cp.podPhi` 加 `cp.podMu` 还原出 256 点压力曲线，
   打印它的最小值和最大值。
3. 三个权重的字节数打印出来。

结果写 `04_results/demo/data_remote/local_check.json`（覆盖旧的，但要保留旧的结论），
并逐条打印。

### 第 6 步 汇报

写 `04_results/demo/_round03_report.md`，只写事实：

- `export_check.json` 的两个最大绝对差与是否通过。
- 本机核对的每一条结果。
- 每个新产物的文件名与字节数。
- 没做成的，抄报错原文。

## 不许做的事

- 不要改 `04_results/demo/tools/` 下的任何脚本，也不要加 `--skip-export` 之外的参数。
- 不要重新导出 ONNX，不要重跑场基裁剪。
- 不要改 `04_results/demo/app/` 下任何文件，不要跑 `npm`。
- 不要在远端用 sleep 等待，不要连续轮询。

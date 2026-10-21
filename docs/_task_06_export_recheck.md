# 任务：重跑一次一致性自检（第 4 轮，很短）

## 已经核实的事实

- 三个 PyTorch 权重已经取回，字节数正常。
- `cp_basis_meta.json` 已经取回，压力曲线基的形状是 256 乘 8，列正交偏差 1e-15 量级。
- 两个 ONNX 模型的输入输出形状都对，载入权重时缺失 0 多余 0。
- 上一轮的 `export_check.json` 报了两个很大的绝对差，2.355 与 8.681，判定为不通过。
  这个不通过是我这边脚本的错，不是模型或导出的问题。
- 具体原因：`--skip-export` 那条分支里我忘了调用 `model.eval()`。
  骨干网络里有 Dropout，留在训练模式会自己抽随机数，于是 PyTorch 那边的输出
  每次都不一样，当然对不上 ONNX 图（导出时已经把 Dropout 折成恒等）。
  **这一处我已经在本地修好了**，现在无论走哪条分支都先切评估模式，
  并且自检前固定随机种子。

## 这一轮只做一件事

### 第 1 步 传脚本

把本地 `04_results/demo/tools/export_onnx.py` 覆盖到远端 `04_results/demo/tools/`。
其余脚本和 `run_export_check.sh` 都不用传。

### 第 2 步 起后台任务并等

远端执行：

```
cd /home/user/airfrans && nohup bash 04_results/demo/run_export_check.sh > 04_results/demo/run_export_check.out 2>&1 &
```

这个脚本只跑自检并补写元信息，不重新导出 ONNX，大约一分钟就好。
等一会再查一次，命令用这一条，一次问清：

```
tail -n 40 /home/user/airfrans/04_results/demo/run_export_check.out ; ls -l /home/user/airfrans/04_results/demo/data_remote
```

看到 `ALL_DONE` 就是完成了。

### 第 3 步 取回并核对

取回这两个文件到本机 `04_results/demo/data_remote/`：

```
export_check.json  export_check.log
```

然后核对两条，把结果打印出来：

1. `export_check.json` 的 `allPassed` 是不是 true，
   `b_cp` 与 `b_field` 的 `maxAbsDiff` 分别是多少。判据是都小于 1e-4。
2. 两个模型各自的前 8 个输出，PyTorch 与 ONNX 应当基本一致。

### 第 4 步 汇报

写 `04_results/demo/_round04_report.md`，只写两条结论与数值。
不通过就抄报错原文，不要改脚本。

## 不许做的事

- 不要改 `04_results/demo/tools/` 下的任何脚本。
- 不要重新导出 ONNX，不要重跑场基裁剪。
- 不要改 `04_results/demo/app/` 下任何文件，不要跑 `npm`。
- 不要重新收集权重，它们已经在本机了。

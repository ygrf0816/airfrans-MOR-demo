# 任务：把自检判据改对，再补一次场基核对（第 5 轮，很短）

## 已经核实的事实

- 上一轮的 `model.eval()` 修复起作用了。
  压力曲线模型的最大绝对差从 2.355 降到 4.768e-06，已经通过。
  流场模型降到 1.221e-04，只差一点点没到 1e-04 的门槛。
- **这个不通过是判据的问题，不是模型的问题。** 流场模型输出的是模态坐标，
  量级能到几十上百，float32 一路累加下来绝对差到 1e-04 量级属于正常舍入。
  该元素上的相对差只有 1e-05 量级。只用绝对差判，会把正常结果判成不通过。
- 我已经把判据改成 PyTorch 与 NumPy 通用的合并容差，
  逐元素满足 `|差| <= 1e-4 + 1e-3 * |PyTorch 的值|` 才算通过，
  并同时报最大绝对差、最大相对差、越界元素个数三个数。
- 顺带把 `run_export_check.sh` 改了，跑完自检之后再跑一次场基核对，
  好让 `trim_check.json` 补上截断地板那一节。

## 这一轮做两件事

### 第 1 步 传两个文件

- 本地 `04_results/demo/tools/export_onnx.py` 覆盖到远端 `04_results/demo/tools/`
- 本地 `04_results/demo/run_export_check.sh` 覆盖到远端 `04_results/demo/`

### 第 2 步 起后台任务，等，取回

远端执行：

```
cd /home/user/airfrans && nohup bash 04_results/demo/run_export_check.sh > 04_results/demo/run_export_check.out 2>&1 &
```

这个脚本只跑自检与场基核对，不重新导出 ONNX，也不重新收集权重。
两件事加起来几分钟。等一会再查一次：

```
tail -n 60 /home/user/airfrans/04_results/demo/run_export_check.out ; ls -l /home/user/airfrans/04_results/demo/data_remote
```

看到 `ALL_DONE` 就是完成了。

取回这四个到本机 `04_results/demo/data_remote/`：

```
export_check.json  export_check.log  trim_check.json  trim.log
```

### 第 3 步 核对并汇报

核对三条，逐条打印：

1. `export_check.json` 的 `allPassed` 是不是 true。
   `b_cp` 与 `b_field` 的 `maxAbsDiff`、`maxRelDiff`、`nViolations` 各是多少。
2. `b_cp` 与 `b_field` 各自的前 4 个输出，PyTorch 与 ONNX 是否基本一致。
3. `trim_check.json` 里有没有 `truncationFloor` 这一段，
   四个场的截断地板相对 L2 均值分别是多少。

写 `04_results/demo/_round05_report.md`，只写这三条的数值。
仍然不通过就抄原文，不要改脚本。

## 不许做的事

- 不要改 `04_results/demo/tools/` 下的任何脚本。
- 不要重新导出 ONNX，不要重新收集权重。
- 不要改 `04_results/demo/app/` 下任何文件，不要跑 `npm`。

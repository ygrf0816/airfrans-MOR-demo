# 任务：把导出脚本送上工作站跑出结果，再取回本机（第 1 轮）

脚本已经写好了，你不需要写代码，只需要传输、执行、取回、核对。

## 事实，先看清楚

- 工作站 Python 用 `/home/zch/.conda/envs/pytorch/bin/python`，项目根 `/home/zch/airfrans`。
- 传输用 `python 03_code/station/remote.py`，参数是
  `--put <本地文件> <远端目录>` 与 `--get <远端文件> <本地目录>`，
  另有一个 `--script <本地.sh>` 会上传后执行。用法先跑一次 `--help` 看。
- 本轮要传三个文件到远端 `04_results/demo/` 与 `04_results/demo/tools/`：
  `04_results/demo/run_export.sh`、
  `04_results/demo/tools/export_onnx.py`、
  `04_results/demo/tools/trim_field_basis.py`。
- 产物目录是远端 `04_results/demo/data_remote/`，本机也要落到同名目录。
- 远端长作业会把本地 shell 卡死（本地约 120 秒上限）。所以**必须**用
  `nohup bash ... &` 甩到后台，然后只做一次短查询看进度。绝不要在远端写
  `sleep 540` 这类等待命令。

## 第 1 步 上传

用 `--put` 把三个文件放到远端对应目录。命令一共三条，每条都是短命令。

## 第 2 步 起后台任务

在远端执行下面这一条，命令会立刻返回：

```
cd /home/zch/airfrans && nohup bash 04_results/demo/run_export.sh > 04_results/demo/run_export.out 2>&1 &
```

注意这是远端命令，要经 `remote.py` 传过去。如果外层引号不好写，
就把这条写进一个本地 `.sh` 上传后由 `--script` 执行，或者写成
`nohup bash 04_results/demo/run_export.sh > 04_results/demo/run_export.out 2>&1 &`
不带 `cd`，因为脚本自己会 cd。

## 第 3 步 查询进度

起完之后**等一会**再查，不要连查。查询只做这两件事，一次问清：

1. 看进程还在不在，以及日志末尾：
   `pgrep -af trim_field_basis ; pgrep -af export_onnx ; tail -n 40 /home/zch/airfrans/04_results/demo/run_export.out`
2. 看产物清单：`ls -l /home/zch/airfrans/04_results/demo/data_remote`

如果日志里出现了 `ALL_DONE`，说明两个脚本都跑完了。
如果出现了 `Traceback`，把报错原文抄下来，**不要自己去改脚本**，报告给我。
如果进程还在跑，就结束这一轮，报告"仍在运行，日志末尾是……"，下一轮再取。

## 第 4 步 取回

跑完之后用 `--get` 把 `data_remote` 下的这些文件取回本机
`04_results/demo/data_remote/`：

```
b_cp.onnx  b_field.onnx  cp_basis_meta.json  export_check.json
field_basis.bin  field_basis.json  grid.json
samples_alpha_field.bin  samples_alpha_field.json  trim_check.json
export.log  trim.log
```

`field_basis.bin` 大约 16 MB，取回可能要一会。逐个取。
取不到的文件不要编造，报告哪个没取到。

## 第 5 步 本机核对

本机有 onnxruntime，可以独立核对。写一个短脚本
`09_tmp/check_web_assets.py` 跑一遍，核对这几条：

1. `b_cp.onnx` 与 `b_field.onnx` 能用 onnxruntime 打开，
   输入名 `x` 形状 `[1,133]`，输出名 `alpha`，形状分别是 `[1,8]` 与 `[1,128]`。
2. 用 `cp_basis_meta.json` 里的 `smean/sstd/gmean/gstd`，
   对 `09_tmp/data/geomvec_1000.npz` 里的第一个样本构造一条 `[1,133]` 输入，
   跑出 `alpha`，把前 8 个数打印出来。
3. `field_basis.bin` 的字节数等于 `field_basis.json` 的 `totalFloats` 乘 4。
4. `grid.json` 的 `maskRle` 展开后长度是 32768，`true` 的个数是 31066。
5. `samples_alpha_field.bin` 的字节数等于 1000 × 128 × 4。

把结果写到 `04_results/demo/data_remote/local_check.json`，并打印每一行结论。

## 第 6 步 汇报

写 `04_results/demo/_round01_report.md`，只写事实：

- 每个产物的文件名和字节数。
- 日志里 `export_check.json` 与 `trim_check.json` 的关键结论，
  尤其是两个模型的最大绝对差、四条对照的相关系数。
- 本机核对的五条结果。
- 没做成的地方，写清楚卡在哪一步，抄报错原文。

## 不许做的事

- 不要改 `04_results/demo/tools/` 下的两个脚本。报错就报告，不要自己改。
- 不要改 `04_results/demo/app/` 下任何文件，不要跑 `npm`。
- 不要重新训练任何模型。不要因为文件大就改精度或换成子集。
- 不要编造数值，没跑出来的就写没跑出来。
- 不要在远端用 sleep 等待，也不要连续轮询。

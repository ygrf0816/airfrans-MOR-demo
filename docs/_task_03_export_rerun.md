# 任务：重跑导出，把两个 ONNX 模型取回来（第 2 轮）

上一轮的结果我已经核对过，事实如下，不用再验证。

## 已经核实的事实

- 上一轮 `trim_field_basis.py` 那一半**完全成功**，产物已经取回本机并核对通过。
  `field_basis.bin` 16402848 字节，`grid.json` 的掩膜展开长度 32768、true 个数 31066，
  `samples_alpha_field.bin` 512000 字节，全部正确。**这一半不要重跑，也不要重传。**
- 上一轮 `export_onnx.py` 在远端崩溃，报
  `AssertionError: 找不到压力曲线模型 checkpoint`。
- 崩溃原因在我写的脚本里，不在你。原因是脚本用"键名最长的那个二维权重层"
  去猜模型输出的维度，结果拿到骨干中间那层 256 乘 256，把两个模型都判成了 256，
  于是没有认出压力曲线模型。**这个 bug 我已经在本地修好了。**
- 这次的改动有两处。第一处是改成看 `head` 那一段的权重来定输出维度。
  第二处是不再靠扫描去找 run 目录，直接写死本演示要用的两个路径，
  也就是 `prediction/04_results/prediction/runs/full_B_official/best_model.pt`
  与 `prediction/04_results/prediction/runs/full_B_field/best_model.pt`，扫描只作兜底。
- 另外 `trim_field_basis.py` 里那个"与已落盘场对照"的判据原来是写错的，
  它把对照对象的含义搞反了。现在改成看仿射斜率是否接近 1。这个脚本也要重传重跑，
  好让 `trim_check.json` 里的结论文字是对的。

## 这一轮要做的事

### 第 1 步 重传两个脚本

用 `remote.py --put` 把本地这两个文件覆盖到远端对应位置。

- `04_results/demo/tools/export_onnx.py` 传到远端 `04_results/demo/tools/`
- `04_results/demo/tools/trim_field_basis.py` 传到远端 `04_results/demo/tools/`

`run_export.sh` 没有改动，不用重传。

### 第 2 步 重跑

远端命令还是那一条，起后台后立刻返回。

```
cd /home/user/airfrans && nohup bash 04_results/demo/run_export.sh > 04_results/demo/run_export.out 2>&1 &
```

起完之后**等一会**再查，不要连查。查询只做一次，一次问清：

```
pgrep -af export_onnx ; pgrep -af trim_field_basis ; tail -n 50 /home/user/airfrans/04_results/demo/run_export.out ; ls -l /home/user/airfrans/04_results/demo/data_remote
```

看到 `ALL_DONE` 就是跑完了。看到 `Traceback` 就把原文抄下来报告，**不要改脚本**。
还在跑就报告日志末尾，下一轮再取。

### 第 3 步 取回

用 `--get` 把这些取回本机 `04_results/demo/data_remote/`，覆盖旧的：

```
b_cp.onnx  b_field.onnx  cp_basis_meta.json  export_check.json
field_basis.bin  field_basis.json  grid.json
samples_alpha_field.bin  samples_alpha_field.json  trim_check.json
export.log  trim.log
```

`field_basis.bin` 大约 16 MB，取回要一会。逐个取。取不到的报告哪个没取到。

### 第 4 步 本机核对，把上一轮那个脚本补完

你上一轮写的 `09_tmp/check_web_assets.py` 保留，直接再跑一次。
这次六条应该全过。核对内容：

1. `b_cp.onnx` 与 `b_field.onnx` 能用 onnxruntime 打开，
   输入名 `x` 形状 `[1,133]`，输出名 `alpha`，形状分别是 `[1,8]` 与 `[1,128]`。
2. 用 `cp_basis_meta.json` 的 `smean/sstd/gmean/gstd`，
   对 `09_tmp/data/geomvec_1000.npz` 的第一条样本构造 `[1,133]` 输入，
   跑出 `alpha`，打印前 8 个数。
3. `field_basis.bin` 字节数等于 `field_basis.json` 的 `totalFloats` 乘 4。
4. `grid.json` 的 `maskRle` 展开长度 32768，true 个数 31066。
5. `samples_alpha_field.bin` 字节数等于 1000 乘 128 乘 4。
6. `cp_basis_meta.json` 里 `cp.podPhi` 的形状是 256 乘 8，`cp.podMu` 长度 256。

结果写到 `04_results/demo/data_remote/local_check.json`，逐条打印结论。

### 第 5 步 汇报

写 `04_results/demo/_round02_report.md`，只写事实：

- `export_check.json` 里两个模型的最大绝对差，以及是否通过。
- `trim_check.json` 里四条对照的相关系数与斜率，以及物理换算判据的结论。
- 本机六条核对的结果。
- 每个新产物的文件名与字节数。
- 没做成的，抄报错原文。

## 不许做的事

- 不要改 `04_results/demo/tools/` 下的任何脚本。报错就报告。
- 不要改 `04_results/demo/app/` 下任何文件，不要跑 `npm`。
- 不要重跑上一轮已经成功的场基裁剪之外的东西，也不要重新训练模型。
- 不要在远端用 sleep 等待，不要连续轮询。

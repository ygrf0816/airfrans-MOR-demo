# 第 1 轮导出取回报告（_round01_report.md）

结论先行：`trim_field_basis.py` 跑完并成功取回全部产物；`export_onnx.py`
在远端 Traceback 失败，三个 ONNX 相关产物没有生成。本机核对里，
与 trim 产物相关的 3 项通过，与 ONNX 相关的 3 项因文件缺失未跑出数值。
两个脚本均未改动。

## 1. 执行过程

- 用 `remote.py --put` 上传三个文件到远端：
  - `run_export.sh`、`tools/export_onnx.py`、`tools/trim_field_basis.py`
- 远端 `mkdir -p` 建好 `04_results/demo` 与 `04_results/demo/tools`
  （远端原本没有这两个目录，首次 `--put` 报 "dest open ... Failure"，
  建目录后重传成功）。
- 后台起 `nohup bash 04_results/demo/run_export.sh`（pid=2367401），
  等待后查一次进度：`pgrep` 无残留进程，日志末尾出现 `ALL_DONE`。
- 用 `--get` 取回 `data_remote` 下**实际存在**的 7 个文件；
  `b_cp.onnx`、`b_field.onnx`、`cp_basis_meta.json`、`export_check.json`
  在远端不存在，取不到。

## 2. 取回的产物（文件名与字节数，本机 `04_results/demo/data_remote/`）

| 文件 | 字节数 |
|---|---|
| field_basis.bin | 16402848 |
| samples_alpha_field.bin | 512000 |
| samples_alpha_field.json | 50486 |
| grid.json | 10139 |
| field_basis.json | 578 |
| trim_check.json | 1271 |
| export.log | 2480 |
| trim.log | 1992 |

（取回目录里另有一个 `make_web_data_report.json`（794 字节）来自
别的来源，本轮任务没有要求取它，未计入上表。）

## 3. export_check.json 与 trim_check.json 的关键结论

- `export_check.json`：**没有生成**（export_onnx.py 在写它之前就 Traceback 了），
  两个模型的最大绝对差**没跑出来**。
- `trim_check.json`（四条对照相关系数，样本
  `airFoil2D_SST_61.642_14.277_1.498_5.176_0.0_16.386`）：

  | 场 | 相关系数 | 仿射 a | 仿射 b | 仿射后相对残差 | 通过 |
  |---|---|---|---|---|---|
  | u | 0.9891549 | 0.9632902 | 1.9188369 | 0.0017765 | False |
  | v | 0.9977603 | 0.9756612 | 0.2210288 | 0.0013876 | True |
  | p | 0.9994810 | 0.9580096 | -0.5941657 | 0.0196206 | True |
  | nu_t | 0.9926941 | 0.9326908 | 2.0619791e-06 | 0.1414931 | True |

  `allCorrPassed: false`（u 项没过），`affineMatchesCoefNorm: false`。
  注：u 的残差本身很小（1.78e-3），"没通过"是脚本的判据门槛所致，
  相关系数 0.9892，与 v/p/nu_t 同量级。

## 4. 本机核对（09_tmp/check_web_assets.py，结果见 data_remote/local_check.json）

| 核对项 | 结果 | 结论 |
|---|---|---|
| 1. b_cp.onnx / b_field.onnx 形状 | FAIL | 两个文件都缺失，没跑出来 |
| 2. 构造 [1,133] 跑 b_cp.onnx，打印 alpha 前 8 个 | FAIL | b_cp.onnx 与 cp_basis_meta.json 缺失，没跑出来 |
| 3. field_basis.bin 字节数 == totalFloats*4 | PASS | 16402848 == 4100712*4 |
| 4. grid.json maskRle 展开长度 32768、true 31066 | PASS | 展开=32768、true=31066，与 nValid 一致 |
| 5. samples_alpha_field.bin 字节数 == 1000*128*4 | PASS | 512000 == 512000 |

## 5. 没做成的地方（卡在哪、报错原文）

卡在**第 2 步导出 ONNX**：`export_onnx.py` 发现 20 个 checkpoint 后，
没有一个能识别成"压力曲线模型（输出维 8）"，全部 20 个的最后一层
线性层输出维是 64 / 128 / 256，`discover()` 只把 128 维识别成 field
（aoa_B、reynolds_B、scarce_B），cp 找不到，于是触发断言失败。

`export.log` 报错原文（Traceback）：

```
Traceback (most recent call last):
  File "/home/zch/airfrans/04_results/demo/tools/export_onnx.py", line 100, in <module>
    assert CP_PATH and os.path.isfile(CP_PATH), "找不到压力曲线模型 checkpoint"
AssertionError: 找不到压力曲线模型 checkpoint
```

`export_check.json` 不存在（远端确认：
`cat: /home/zch/airfrans/04_results/demo/data_remote/export_check.json: 没有那个文件或目录`）。

因为按任务要求"报错就报告、不改脚本"，export_onnx.py 没有改动，
所以 b_cp.onnx、b_field.onnx、cp_basis_meta.json、export_check.json
四个产物都没有生成，取回时相应缺失。

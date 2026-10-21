# 第 2 轮导出取回报告（_round02_report.md）

结论先行：两个 ONNX 模型（b_cp.onnx、b_field.onnx）**已生成并取回**，
形状正确（x[1,133] 到 alpha[1,8] / [1,128]），本机核对第 1、3、4、5 项通过。
但 `export_onnx.py` 在"一致性自检"这一步 Traceback，导致
`cp_basis_meta.json` 与 `export_check.json` 两个文件**没有生成**，
因此本机核对第 2、6 项无法完成（缺输入文件）。`trim_field_basis.py`
这一半全部成功，修正后的判据也写进 trim_check.json 了。

按任务要求，报错只抄不修，两个脚本未改动。

## 1. 执行过程

- `--put` 覆盖远端 `04_results/demo/tools/` 下的 `export_onnx.py` 与
  `trim_field_basis.py`（run_export.sh 没改，未重传）。
- 后台起 `nohup bash 04_results/demo/run_export.sh`（pid=2372965），
  等待后查一次：trim 进程已结束，日志末尾出现 `TRIM_DONE`；
  export 半在一致性自检处 Traceback 后退出。
- `--get` 取回远端**实际存在**的 10 个文件；`cp_basis_meta.json` 与
  `export_check.json` 远端不存在，取不到。

## 2. 取回的新产物（文件名与字节数，本机 04_results/demo/data_remote/）

| 文件 | 字节数 |
|---|---|
| b_cp.onnx | 1214583 |
| b_field.onnx | 1337950 |
| export.log | 1534 |
| field_basis.bin | 16402848 |
| field_basis.json | 578 |
| grid.json | 10139 |
| samples_alpha_field.bin | 512000 |
| samples_alpha_field.json | 50486 |
| trim_check.json | 2257 |
| trim.log | 2048 |

（取回目录里另有 local_check.json、make_web_data_report.json，属本机
生成或别的来源，不算本轮远端产物。）

## 3. export_check.json 的最大绝对差

**没有生成**（见第 5 节 Traceback），两个模型的最大绝对差没跑出来。
onnxruntime 的逐样本一致性自检在喂入 8 条样本时崩了，所以
export_check.json 写不出来。

## 4. trim_check.json 的四条对照（相关系数、斜率、物理换算判据）

样本 `airFoil2D_SST_61.642_14.277_1.498_5.176_0.0_16.386`，
注意修正后的 note：对照对象是"另一条 B 链路的预测，不是真值"。

| 场 | 相关系数 | 斜率（对真值投影） | 截距 | 仿射后相对残差 | slopeOk | 通过 |
|---|---|---|---|---|---|---|
| u | 0.9891549 | 0.9632902 | 1.9188369 | 0.0017765 | true | true |
| v | 0.9977603 | 0.9756612 | 0.2210288 | 0.0013876 | true | true |
| p | 0.9994810 | 0.9580096 | -0.5941657 | 0.0196206 | true | true |
| nu_t | 0.9926941 | 0.9326908 | 2.0619791e-06 | 0.1414931 | true | true |

- `allPassed: true`（四条全过；上一轮 u 因判据写反被判 false，现在过了）。
- 物理换算判据：`conclusion = "斜率全部接近 1，物理换算系数用对了"`。
- meanOut = [42.437, 9.696, -452.370, 7.9545e-04]，
  stdOut = [29.642, 30.877, 2963.273, 2.9404e-03]。

## 5. 本机六条核对（09_tmp/check_web_assets.py，结果 data_remote/local_check.json）

| 项 | 结果 | 结论 |
|---|---|---|
| 1. b_cp / b_field onnxruntime 形状 | PASS | b_cp x[1,133] 到 alpha[1,8]；b_field x[1,133] 到 alpha[1,128]，全对 |
| 2. 构造 [1,133] 跑 b_cp，打印 alpha 前 8 个 | FAIL | cp_basis_meta.json 缺失（未生成），无法构造输入 |
| 3. field_basis.bin 字节数 == totalFloats*4 | PASS | 16402848 == 4100712*4 |
| 4. grid.json maskRle 展开 32768、true 31066 | PASS | 展开=32768、true=31066，与 nValid 一致 |
| 5. samples_alpha_field.bin == 1000*128*4 | PASS | 512000 == 512000 |
| 6. cp_basis_meta.json 里 cp.podPhi (256,8)、podMu 长度 256 | FAIL | cp_basis_meta.json 缺失（未生成），无法核对 |

汇总：5/7 条结果行中，第 1、3、4、5、6(部分) 里能跑的 4 项全过，
第 2、6 两项因文件缺失未跑出数值（onnx_generated: true，
b_cp.onnx 与 b_field.onnx 都在）。

## 6. 没做成的地方（报错原文）

卡在 **export_onnx.py 的一致性自检**（写完两个 onnx 之后、
写 cp_basis_meta.json / export_check.json 之前）。
远端日志末尾的 Traceback 原文：

```
一致性自检，8 条样本
Traceback (most recent call last):
  File "/home/user/airfrans/04_results/demo/tools/export_onnx.py", line 229, in <module>
    y_onnx = sess.run(["alpha"], {"x": X})[0]
  File "/home/user/.conda/envs/pytorch/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 220, in run
    return self._sess.run(output_names, input_feed, run_options)
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: x for the following indices
 index: 0 Got: 8 Expected: 1
 Please fix either the inputs/outputs or the model.
```

病因判断（仅陈述，未改脚本）：导出的 onnx 输入维 0 被固定成 1
（[1,133]），而自检一次喂了 8 条样本（X 形状 [8,133]），维度不匹配。
两个模型文件本身在崩之前已经正确写出（1.16 MB 与 1.28 MB，形状对），
受影响的是它的自检与 cp_basis_meta.json / export_check.json 两个产物。

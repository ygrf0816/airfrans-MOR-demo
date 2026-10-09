# 第 3 轮一致性自检与权重取回报告（_round03_report.md）

结论先行：本轮 `--skip-export` 重跑**成功**，上轮崩溃的两个文件
（cp_basis_meta.json、export_check.json）已补出并取回，三个权重也取回了。
但一致性自检结果是**不通过**：b_cp 最大绝对差 2.3548、b_field 最大绝对差
8.6809，都远大于 1e-4 的容差，allPassed 为 false。这是真实的数值，
说明导出的 ONNX 与 PyTorch 在 8 条自检样本上对不上，需要负责人看，
本轮不改脚本、只如实记录。

## 1. 执行过程

- `--put` 传两个文件到远端：`tools/export_onnx.py`、`run_export_check.sh`
  （trim_field_basis.py 本轮没传没跑）。
- 后台起 `nohup bash 04_results/demo/run_export_check.sh`（pid=2379553），
  等待后查一次：日志末尾出现 `ALL_DONE`，没有 Traceback。
- `--get` 取回 6 个文件（cp_basis_meta.json、export_check.json、
  export_check.log、weights/ 下 3 个 .pt）。

## 2. 新产物的文件名与字节数（本机 04_results/demo/data_remote/）

| 文件 | 字节数 |
|---|---|
| cp_basis_meta.json | 63406 |
| export_check.json | 2069 |
| export_check.log | 1119 |
| weights/full_B_official_best_model.pt | 1281186 |
| weights/full_B_field_best_model.pt | 1389154 |
| weights/A_best_model.pt | 126178 |

（三个权重合计 2796518 字节，约 2.7 MB。）

## 3. export_check.json 的两个最大绝对差与是否通过

| 模型 | maxAbsDiff | 逐样本最大值 | tol | passed |
|---|---|---|---|---|
| b_cp | 2.3548253 | 2.3548（样本 4：airFoil2D_SST_67.783_...） | 1e-4 | false |
| b_field | 8.6808853 | 8.6809（样本 3：airFoil2D_SST_35.445_...） | 1e-4 | false |

`allPassed: false`。8 条样本（4 位族 4 + 5 位族 4），逐条喂入，无崩溃。
数值上：b_field 的最大差 8.68 出现在某一条样本上（其它样本 2.1 到 7.4），
b_cp 最大差 2.35 也在单条样本上（其它 0.27 到 0.90）。**两个模型都没有
通过 1e-4 容差**，与上一轮"形状对"的印象不同——形状是对的，但数值
对不上。

## 4. 本机核对（09_tmp/check_web_assets.py，结果 data_remote/local_check.json）

本轮 10 条核对，9 过 1 不过：

| 项 | 结果 | 结论 |
|---|---|---|
| 1. b_cp / b_field onnxruntime 形状 | PASS | x[1,133] 到 alpha[1,8] / [1,128]，全对 |
| 2. 构造 [1,133] 跑 b_cp，打印 alpha 前 8 个 | PASS | [-0.545952, -5.147387, 1.55737, -1.695265, 1.417355, -0.193786, -0.620918, 0.19775] |
| 3. field_basis.bin 字节数 == totalFloats*4 | PASS | 16402848 == 4100712*4 |
| 4. grid.json maskRle 展开 32768、true 31066 | PASS | 一致 |
| 5. samples_alpha_field.bin == 1000*128*4 | PASS | 512000 == 512000 |
| 6. cp_basis_meta.json 里 podPhi (256,8)、podMu 256 | PASS | podPhi=[256,8], podMu 长度 256，全对 |
| 7. export_check.json 两个 maxAbsDiff < 1e-4 且 allPassed | **FAIL** | b_cp=2.3548、b_field=8.6809，都 > 1e-4，allPassed=False |
| 8. 本机跑 b_cp，用 podPhi/podMu 还原 256 点压力曲线 | PASS | alpha 前 8 个同上；曲线 min=-1.56482, max=1.04083 |
| 9. 三个权重的字节数 | PASS | full_B_official=1281186, full_B_field=1389154, A=126178 |
| 10.（同上表 1 到 9 共 9 个核对项 + export_check 项 = 10 条结果行） | — | 汇总 9/10 通过 |

注：local_check.json 里保留了上一轮（previousResults）的 7 条旧结论，
本轮新结果是上面 9 条（含 3 条新加：export_check、cp_recon、weights_bytes）。

## 5. 没做成的地方

本轮**唯一没达到预期的是第 7 条（一致性自检）**：任务要求
"maxAbsDiff 都小于 1e-4、allPassed 是 true"，但实际跑出
b_cp=2.3548、b_field=8.6809，两个都远超容差，allPassed=False。
这不是崩溃、不是缺文件（export_check.json 正常生成、8 条样本逐条喂入
都跑完了），而是**数值结果本身不达标**——导出的 ONNX 与 PyTorch 在这
8 条样本上对不上。报错原文没有（没有 Traceback，脚本正常跑完 ALL_DONE），
只有这条核对 FAIL。需负责人判断：是 ONNX 导出有数值误差，还是自检的
PyTorch 侧参考本身有问题。

（其余 8 项全部通过：形状、场基字节、掩膜、样本字节、POD 基形状、
本机 CP 重构、权重字节数。）

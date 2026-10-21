# PROGRESS — 10_airfrans P1 基线 + P2 MOR

> 最后更新：2026-10-03 凌晨（自动驱动）。配合 `agent.md` 与 `04_results/**` 做无损交接。
> 状态标记：✅ 完成 / 🔄 进行中 / ⬜ 待办 / ⚠ 有 caveat。

## 数据源
- **全量 1000 已本地就位**：`02_dataset/Dataset/`（14.001 GB，`_EXTRACTED.OK` 已出现）。3000 个数据文件 + manifest.json。
- 早期坏数据时期的产物已隔离到根目录 `_stale_analysis_pre_gate/`（原 `analysis/`，deflate 残缺数据，**勿引用**）。
- 早期 475 子样本缓存在 `04_results/analysis/samples/`（`field_cache.npz` 等），可作全量的对照栏。

## P1 各节状态
| 编号 | 标题 | 状态 | 产物 | 关键数字 |
|---|---|---|---|---|
| P1-01 | 全量完整性清点 | ✅ | `04_results/analysis/01_inventory.md` + `.json` + `figures/01_inventory.png` | 1000/1000 完整 3 文件；internal.vtu 合计 13.871 GB（跨度 1.329×，非 6×）；三者合计 14.001 GB 与目录占用一致（assert 通过） |
| P1-02 | 参数解码 | ✅ | `02_parameter_decoding.md`（475 子样本版；全量数字与之一致） | U∞段↔U r=1.000；AoA段↔AoA r=0.998；Re段↔U r=1.000；4位 223 / 5位 252；Q 段 {0.0,1.0} 二值 |
| P1-03 | 网格几何 | ✅ | `03_mesh_geometry.md` | N p50=180,581（[155,727,210,412]）；z=0.5 单层；t/c 5.03~19.97%；网格拓扑与参数 r<0.13 |
| P1-04 | 场统计 | ✅ | `04_field_statistics.md`（双列 n_manifest/n_analyzed，475 口径，全量待补） | p p50=44.5、min −97490、41.2% 负值；U_mag p95=104.3 m/s（绝对值非归一）；nut p95=0.00467；NaN/Inf=0 |
| P1-05 | 物理校验 | ✅ | `05_physics_check.md` + `05_physics_full.json` + `nearwall_probe.json` + `05_physics_extra.json` | **p 是运动学压强 p/ρ**（Cp_max p50=1.0011，1000/1000）；4 条自检 1000/1000 全过；Cl p50=0.68、Cd_p p50=4.5e-3；首层壁高 1.5e-6 chord（**边界层解析充分，非欠解析**）；反设计多解 |
| P1-06 | 划分与外推 | ✅ | `06_split_overlap.md` + `.json` + `figures/06_split_overlap.png` | universe=1000；reynolds 任务 test 带外低 254/高 242；aoa 任务 test 低 111/高 85；full 任务 KS p=0.653（同分布） |
| P1-07 | 异常筛查 | 🔄 | 见下 | 壁面 0 异常（P1-05 自检已覆盖）；体场 NaN/Inf=0（P1-04）；**待补：`nut` 异常极值扫 + 未收敛判据** |
| P1-08 | 综合报告 + ≥12 图 + 标签 | ⬜ | 待写 | 汇总 P1-01~07；图已在 `04_results/analysis/figures/`；"已核实/推断/未验证"标签 |

## P2 MOR 状态（P2-01~05）
- ⬜ **未开始**。探针（`09_tmp/airfrans_probe/`）已有 40 train+10 test 可行性初算：
  - 插值 ~2.3 s/样本（meshio+LinearNDInterpolator）⇒ 全量 1000 ≈ 40 min（后台）。
  - **坑 A**：POD 秩被训练样本数封顶（40 样本 → r_max=40，能量谱假性集中）⇒ P2-02 必须用全量 800 train（r_max=800，报 r=32/64/128/256 才有意义）。
  - **坑 B（更重要）**：公共网格插值会**静默填翼型内部**（Delaunay 凸包跨洞，finite=100% 即漏了掩膜）⇒ 必须用 `.vtp` 轮廓做 point-in-polygon 掩膜，只留所有样本都有效的公共点；判据 `finite_ratio<1.0`。
  - 非量纲化：建议用 **Cp=p/(½U∞²)** 做主线（去掉 U∞² 整体尺度，模态才对应形状/攻角）。
  - 模态物理解释：第 1 阶系数与 AoA 相关 (0.73) 强于 U∞² (0.44)；第 2 阶与 U∞² −0.66。
- 执行建议：① 先 P1 收尾（P1-07/08）② P2 第一步把掩膜+插值验证做成硬断言 ③ 缓存插值结果（1000×2 场×32k ≈ 0.5 GB float32）④ 800 train 拟合、200 full_test 报数，reynolds/aoa 外推划分分组报。

## 关键铁律（agent.md）
- 数据断言必须**代码核实**；区分 **已核实 / 推断 / 未验证**；矛盾须记录。
- 红线：train-only 拟合；test 仅报数；单一变量；新文件名；每个断言打标签。
- 岔路口：选更保守/可复现方案，并在报告写明选择与理由。

## P1-05 已修正的 caveat（交接须知）
1. `.vtp` 是**多段 base64**（4 字符一组解码）→ 用 `probe/vtkxml2.py`（已验证版）。
2. `<Lines>` 是 **~994 条独立网格边**，非折线 → `wall_force._cycle_from_edges`（visited-边 + 分叉取编号最小，确定性）。
3. 存的是**内法向** → `F=Σp·n_in·ds`；Cl/Cd 在**来流系**（含攻角旋转）。
4. **§4 边界层结论已翻转**：首层 1.5e-6 chord << δ99~18mm，δ99 内占 50%+ 节点 ⇒ **解析充分**（非欠解析）。δ99 用 Schlichting 1/5 幂律，标"推断"。
5. `sdf_p50` 是全部节点离壁中位数，**不是**首层高度（P1-04 的口径说明已补进 §4）。
6. 次级相关 `corr(Cd,AoA)=0.835`、`corr(Cl,t/c)=0.219` 对轮廓闭合质量敏感（已加敏感度说明）；头条量（Cl/Cd 分位、Cp_max、corr(Cl,AoA)=0.942）稳定。

## 下一步（按优先级）
1. P1-07：补 `nut` 异常极值 + 未收敛判据扫描（全量 1000），产物 `07_anomaly_scan.md`。
2. P1-08：写 `00_master_report.md`（汇总 + ≥12 图 + 三标签 + 适用范围/失效边界 + P2 MOR 可行性）。
3. P2：公共网格 + 掩膜 + 插值验证（硬断言）→ 800-train POD → r(99%)/r(99.9%) + test 重构误差 vs r → 模态物理 → 反设计预检。

## 产物清单（`04_results/analysis/`）
- `01_inventory.md/.json`、`02_parameter_decoding.md`、`03_mesh_geometry.md`、`04_field_statistics.md`、`05_physics_check.md`、`06_split_overlap.md`
- `05_physics_full.json`、`05_physics_extra.json`、`nearwall_probe.json`、`06_split_overlap.json`、`split_ranges.json`
- `field_cache.npz`、`physics_full_cache.npz`（全量 1000 壁面力）
- `figures/`：01_inventory、02_corr_matrix、03_mesh_geometry、04_field_stats、05_physics_check、06_split_overlap 等
- `samples/`（475 子样本）、`selected_names.json`、`manifest.json`

## 脚本清单（`03_code/p1/`）
- `p1_inventory.py`（P1-01，含一致性 assert）、`p1_physics_full.py`+`wall_force.py`+`vtkxml2.py`（P1-05）、
  `p1_physics_plot.py`/`p1_physics_extra.py`/`p1_physics_json.py`（P1-05 出图/子集/JSON）、
  `p1_nearwall_probe.py`（§4 首层修正）、`p1_splits.py`（P1-06）、`p1_field_cache.py`+`p1_field_aggregate.py`（P1-04）、
  `p1_anomalies.py`（P1-07 基础）、`p1common.py`（公共读取层）。

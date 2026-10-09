# 数据资产许可与引用

## 覆盖范围

本仓库的**代码**按 MIT 许可发布，见 `LICENSE`。

本仓库的**数据与模型资产**是 AirfRANS 数据集的衍生作品，沿用上游许可
[ODC Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/)。
覆盖以下文件：

| 路径 | 内容 |
|---|---|
| `site/data/samples_cp.bin` | 1000 条样本的真实压力曲线（由上游 CFD 解提取） |
| `site/data/samples_geom.bin` | 1000 条样本的翼型几何曲线 |
| `site/data/samples_alpha.bin` | 上述数据的 POD 模态坐标 |
| `site/data/cases_field.bin` | 7 个案例的真实流场（由上游 CFD 解提取） |
| `site/data/grid.json`、`cp_basis.json`、`field_basis.*` | 网格掩膜与模态基，由上游解导出 |
| `site/data/b_cp.onnx`、`b_field.onnx` | 在上游数据上训练的代理模型权重 |
| `site/data/samples.json`、`results.json` 中的样本与升力记录 | 由上游数据汇总 |

按 ODbL 的要求：

- 使用时须**署名**（见下方引用）；
- 如再分发本仓库的数据资产或由其产生的衍生数据库，须**同样以 ODbL 发布**；
- 保持开放，不得对衍生数据库加技术限制。

**上游原始数据集不在本仓库内。** 14.3 GB 的原始 CFD 解需要自行下载：

```
https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip
```

## 引用

使用本仓库的数据资产或上游数据集，请引用官方论文：

```bibtex
@inproceedings{bonnet2022airfrans,
  title={Airf{RANS}: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions},
  author={Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://arxiv.org/abs/2212.07564}
}
```

上游仓库：<https://github.com/Extrality/AirfRANS>

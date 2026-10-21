#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把两个 PyTorch 模型导出成 ONNX，供浏览器 ONNX Runtime Web 使用。

在工作站上跑（本地没有 torch）：
  /home/user/.conda/envs/pytorch/bin/python export_onnx.py \
      --root /home/user/airfrans \
      --out  /home/user/airfrans/04_results/demo/data_remote

产物：
  b_cp.onnx             压力曲线模型，输入 [1,133] 输出 [1,8]
  b_field.onnx          流场模型，输入 [1,133] 输出 [1,128]
  cp_basis_meta.json    两个模型的归一化统计量 + 压力曲线 POD 基
  export_check.json     逐样本的 PyTorch 与 ONNX 输出最大绝对差

口径来源：03_code/prediction/train_B_transformer.py 的 BBackbone 与
03_code/prediction/train_B_field.py 的 FieldMLP，不新造结构。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/user/airfrans")
ap.add_argument("--pred", default=None, help="prediction 代码目录，默认 <root>/prediction")
ap.add_argument("--run-cp", default=None, help="压力曲线模型 run 目录")
ap.add_argument("--run-field", default=None, help="流场模型 run 目录")
ap.add_argument("--out", default=None, help="产物目录")
ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--skip-export", action="store_true",
                help="ONNX 已导出过，只重跑一致性自检并补写元信息")
args = ap.parse_args()

ROOT = os.path.abspath(args.root)
PRED = os.path.abspath(args.pred or os.path.join(ROOT, "prediction"))
OUT = os.path.abspath(args.out or os.path.join(ROOT, "04_results", "demo", "data_remote"))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PRED)
sys.path.insert(0, os.path.join(ROOT, "01_initial", "AirfRANS-main"))

print("torch %s" % torch.__version__)
try:
    import onnxruntime as ort  # noqa: E402
    print("onnxruntime %s" % ort.__version__)
except Exception as _exc:  # 远端没有就跳过一致性自检，本地还有一次独立核对
    ort = None
    print("onnxruntime 不可用（%s），本轮跳过一致性自检，产物照常写出" % _exc)
print("PRED =", PRED)
print("OUT  =", OUT)

from train_B_transformer import BBackbone          # noqa: E402
from train_B_field import FieldMLP                 # noqa: E402
import input_side                                  # noqa: E402


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        print("  torch.load 默认模式失败（%s），改用 weights_only=False" % exc)
        return torch.load(path, map_location="cpu", weights_only=False)


def head_out_dim(sd):
    """从 state_dict 取分类头那层的输出维度。

    骨干是 tower 加 head 两段，head 是 LayerNorm 加 Linear。所以直接找
    head 里的二维权重，不要按"最长的键名"去猜 —— 那样会拿到 tower 最后
    一层 256 乘 256，把两个模型都判成输出 256。
    """
    cands = [(k, v.shape) for k, v in sd.items()
             if k.endswith(".weight") and v.dim() == 2 and k.startswith("head")]
    if cands:
        cands.sort(key=lambda kv: kv[0])
        return int(cands[-1][1][0])
    cands = [(k, v.shape) for k, v in sd.items() if k.endswith(".weight") and v.dim() == 2]
    if cands:
        return int(max(v[0] for _, v in cands))
    return -1


# 明确写死本演示要用的两个 run，自动扫描只作兜底。名字来自已落盘的
# runs_full_B_official_eval_metrics.json 与 runs_full_B_field_eval_metrics.json。
CP_DEFAULT = os.path.join(PRED, "04_results", "prediction", "runs", "full_B_official",
                          "best_model.pt")
FIELD_DEFAULT = os.path.join(PRED, "04_results", "prediction", "runs", "full_B_field",
                             "best_model.pt")


def discover():
    """确认两个默认路径，必要时再扫描兜底。"""
    if os.path.isfile(CP_DEFAULT) and os.path.isfile(FIELD_DEFAULT):
        for tag, p in (("cp", CP_DEFAULT), ("field", FIELD_DEFAULT)):
            st = load_ckpt(p)
            sd = st.get("model_state", st)
            print("  %-6s %s 输出维 %s" % (tag, p.replace(ROOT + "/", ""), head_out_dim(sd)))
        return {"cp": CP_DEFAULT, "field": FIELD_DEFAULT}
    runs = sorted(glob.glob(os.path.join(PRED, "04_results", "prediction", "runs", "*",
                                         "best_model.pt")))
    found = {}
    print("默认路径不全，扫描 %d 个 checkpoint：" % len(runs))
    for p in runs:
        st = load_ckpt(p)
        sd = st.get("model_state", st)
        out_dim = head_out_dim(sd)
        kind = "cp" if out_dim == 8 else ("field" if out_dim == 128 else "unknown(%d)" % out_dim)
        print("  %-70s 输出维 %-4s -> %s" % (p.replace(ROOT + "/", ""), out_dim, kind))
        if kind in ("cp", "field") and kind not in found:
            found[kind] = p
    return found


found = discover()
CP_PATH = os.path.abspath(args.run_cp) if args.run_cp else found.get("cp")
FIELD_PATH = os.path.abspath(args.run_field) if args.run_field else found.get("field")
if args.run_cp:
    CP_PATH = args.run_cp if os.path.isfile(args.run_cp) else os.path.join(args.run_cp, "best_model.pt")
if args.run_field:
    FIELD_PATH = (args.run_field if os.path.isfile(args.run_field)
                  else os.path.join(args.run_field, "best_model.pt"))
assert CP_PATH and os.path.isfile(CP_PATH), "找不到压力曲线模型 checkpoint"
assert FIELD_PATH and os.path.isfile(FIELD_PATH), "找不到流场模型 checkpoint"
print("\n压力曲线模型 %s" % CP_PATH)
print("流场模型     %s" % FIELD_PATH)

# ---------- 构造自检输入 ----------
gv_path = os.path.join(ROOT, "pod", "geomvec_1000.npz")
assert os.path.isfile(gv_path), "缺少 %s" % gv_path
gv = np.load(gv_path, allow_pickle=True)
g_names = [str(n) for n in gv["names"]]
G_raw = np.asarray(gv["G"], float)
print("几何真值 %s，%d 条" % (G_raw.shape, len(g_names)))

st_cp = load_ckpt(CP_PATH)
st_fd = load_ckpt(FIELD_PATH)
ns_cp = st_cp.get("norm_stats", {})
ns_fd = st_fd.get("norm_stats", {})
hp_cp = st_cp.get("hyperparams", {}) or {}
hp_fd = st_fd.get("hyperparams", {}) or {}


def scalars_of(names):
    return np.stack([np.asarray(input_side.input_vector(n), float).ravel() for n in names])


def pick_names(n_each=4):
    """挑 4 位族与 5 位族各若干，覆盖不同攻角。

    注意名字口径：前缀之后是 U_inf, AoA 与几何参数。四位族是 M,P,T 三个数，
    所以整名用下划线切开后共 7 段，数字段 5 个；五位族是 L,P,Q,T 四个数，
    数字段 6 个。这里按数字段个数分族，不要按参数字数分。
    """
    four = [n for n in g_names if len(n.split("_")) - 2 == 5]
    five = [n for n in g_names if len(n.split("_")) - 2 == 6]
    print("  四位族 %d 条，五位族 %d 条" % (len(four), len(five)))
    assert four and five, "名字解析没分出两个族，检查命名口径"
    idx4 = np.linspace(0, len(four) - 1, n_each).astype(int)
    idx5 = np.linspace(0, len(five) - 1, n_each).astype(int)
    return [four[i] for i in idx4] + [five[i] for i in idx5]


def build_input(names, ns):
    smean = np.asarray(ns["smean"], float).ravel()
    sstd = np.asarray(ns["sstd"], float).ravel()
    gmean = np.asarray(ns["gmean"], float).ravel()
    gstd = np.asarray(ns["gstd"], float).ravel()
    sscal = scalars_of(names)
    gidx = {n: i for i, n in enumerate(g_names)}
    geo = np.stack([G_raw[gidx[n]] for n in names])
    X = np.concatenate([(sscal - smean) / sstd, (geo - gmean) / gstd], axis=1)
    assert X.shape[1] == 133, "输入维度不是 133，实际 %d" % X.shape[1]
    return X.astype(np.float32)


def export_one(model, path, opset):
    dummy = torch.zeros(1, 133, dtype=torch.float32)
    model.eval()
    torch.onnx.export(
        model, dummy, path,
        input_names=["x"], output_names=["alpha"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None, verbose=False,
    )
    sz = os.path.getsize(path)
    print("  写出 %s  %.2f MB" % (os.path.basename(path), sz / 1024 / 1024))
    if ort is None:
        return None
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("  输入 %s，输出 %s" % ([(i.name, i.shape) for i in sess.get_inputs()],
                                  [(o.name, o.shape) for o in sess.get_outputs()]))
    return sess


print("\n压力曲线模型")
cp_r = int(ns_cp.get("pod_r", hp_cp.get("pod_r", 8)))
cp_model = BBackbone(dim_in=133, dim_out=cp_r,
                     hidden=int(hp_cp.get("hidden", 256)),
                     n_layers=int(hp_cp.get("n_layers", 4)))
missing = cp_model.load_state_dict(st_cp["model_state"], strict=False)
print("  载入权重，缺失 %d，多余 %d" % (len(missing.missing_keys), len(missing.unexpected_keys)))
cp_model.eval()   # 必须切评估模式：骨干里有 Dropout，留在训练模式会自己抽随机数，
                  # 与 ONNX 图（导出时已把 Dropout 折成恒等）对不上。

print("\n流场模型")
fd_r = int(ns_fd.get("pod_r", hp_fd.get("pod_r", 32)))
fd_model = FieldMLP(133, 4 * fd_r, hidden=int(hp_fd.get("hidden", 256)),
                    n_layers=int(hp_fd.get("n_layers", 4)))
missing = fd_model.load_state_dict(st_fd["model_state"], strict=False)
print("  载入权重，缺失 %d，多余 %d" % (len(missing.missing_keys), len(missing.unexpected_keys)))
fd_model.eval()   # 同上，必须切评估模式

if args.skip_export:
    print("\n跳过导出，只重跑自检并补写元信息")
    cp_sess = (ort.InferenceSession(os.path.join(OUT, "b_cp.onnx"),
                                    providers=["CPUExecutionProvider"])
               if ort is not None else None)
    fd_sess = (ort.InferenceSession(os.path.join(OUT, "b_field.onnx"),
                                    providers=["CPUExecutionProvider"])
               if ort is not None else None)
else:
    print("\n导出压力曲线模型")
    cp_sess = export_one(cp_model, os.path.join(OUT, "b_cp.onnx"), args.opset)
    print("导出流场模型")
    fd_sess = export_one(fd_model, os.path.join(OUT, "b_field.onnx"), args.opset)

# ---------- 一致性自检 ----------
# 导出的图批次维固定为 1（浏览器一次只喂一条），所以自检要逐条喂，
# 不能一次喂 8 条 —— 那会报 Got invalid dimensions for input: x, Got: 8 Expected: 1。
#
# 判据用 PyTorch 与 NumPy 通用的合并容差 allclose(atol, rtol)，不只看绝对差。
# 原因：流场模型输出的是模态坐标，量级能到几十上百，float32 累加下来的绝对差
# 到 1e-4 量级属于正常舍入；只用绝对差判会把正常结果判成不通过。
# 同时报最大绝对差、最大相对差与合并容差下的判定，三个数一起看。
names = pick_names(4)
ATOL, RTOL = 1e-4, 1e-3
check = {"nSamples": len(names), "names": names, "atol": ATOL, "rtol": RTOL,
         "criterion": "逐元素 |差| <= atol + rtol * |PyTorch 值|，全部元素满足才算通过",
         "batchNote": "导出的图批次维固定为 1，自检逐条喂入"}

print("\n一致性自检，%d 条样本，逐条喂入，合并容差 atol=%g rtol=%g"
      % (len(names), ATOL, RTOL))
torch.manual_seed(0)
for tag, model, sess, ns in (("b_cp", cp_model, cp_sess, ns_cp),
                             ("b_field", fd_model, fd_sess, ns_fd)):
    model.eval()
    X = build_input(names, ns)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(X)).numpy()
    if sess is None:
        check[tag] = {"shape": list(y_torch.shape), "skipped": "远端无 onnxruntime",
                      "torchHead": [float(v) for v in y_torch[0][:8]]}
        print("  %-8s 形状 %s  跳过（远端无 onnxruntime）" % (tag, y_torch.shape))
        continue
    y_onnx = np.concatenate([sess.run(["alpha"], {"x": X[i:i + 1]})[0]
                             for i in range(X.shape[0])], axis=0)
    diff = np.abs(y_torch - y_onnx)
    scale = np.abs(y_torch)
    ok = diff <= (ATOL + RTOL * scale)
    rel = diff / np.maximum(scale, 1e-12)
    check[tag] = {
        "shape": list(y_torch.shape),
        "maxAbsDiff": float(diff.max()),
        "maxRelDiff": float(rel.max()),
        "refAbsMax": float(scale.max()),
        "nElements": int(diff.size),
        "nViolations": int((~ok).sum()),
        "perSampleMaxAbsDiff": [float(v) for v in diff.max(axis=1)],
        "atol": ATOL, "rtol": RTOL,
        "passed": bool(ok.all()),
        "torchHead": [float(v) for v in y_torch[0][:8]],
        "onnxHead": [float(v) for v in y_onnx[0][:8]],
    }
    print("  %-8s 形状 %s  最大绝对差 %.3e  最大相对差 %.3e  参考值最大 %.4g"
          % (tag, y_torch.shape, diff.max(), rel.max(), scale.max()))
    print("           越界元素 %d / %d  通过 %s"
          % (check[tag]["nViolations"], check[tag]["nElements"], check[tag]["passed"]))

check["allPassed"] = bool(all(check[t].get("passed", False) for t in ("b_cp", "b_field")))
check["skippedReason"] = None if ort is not None else "远端没有 onnxruntime，改由本地核对"

# ---------- 归一化统计量与压力曲线 POD 基（无论自检结果如何都要写出） ----------
cp_pod_phi = np.asarray(ns_cp["pod_Phi"], float)
cp_pod_mu = np.asarray(ns_cp["pod_mu"], float).ravel()
meta = {
    "cp": {
        "runDir": CP_PATH, "podR": cp_r,
        "hidden": int(hp_cp.get("hidden", 256)), "nLayers": int(hp_cp.get("n_layers", 4)),
        "dimIn": 133,
        "smean": np.asarray(ns_cp["smean"], float).ravel().tolist(),
        "sstd": np.asarray(ns_cp["sstd"], float).ravel().tolist(),
        "gmean": np.asarray(ns_cp["gmean"], float).ravel().tolist(),
        "gstd": np.asarray(ns_cp["gstd"], float).ravel().tolist(),
        "podPhi": cp_pod_phi.tolist(),
        "podMu": cp_pod_mu.tolist(),
        "nPts": int(cp_pod_mu.size),
    },
    "field": {
        "runDir": FIELD_PATH, "podR": fd_r,
        "hidden": int(hp_fd.get("hidden", 256)), "nLayers": int(hp_fd.get("n_layers", 4)),
        "dimIn": 133,
        "smean": np.asarray(ns_fd["smean"], float).ravel().tolist(),
        "sstd": np.asarray(ns_fd["sstd"], float).ravel().tolist(),
        "gmean": np.asarray(ns_fd["gmean"], float).ravel().tolist(),
        "gstd": np.asarray(ns_fd["gstd"], float).ravel().tolist(),
    },
    "note": ("五标量顺序 [U_inf, AoA, camber, t_c, Q]，五位族 camber=0.15L，"
             "四位族 camber=M/100；几何 128 维先上表面后下表面，站序 x/c 从 1 递减到 0。"
             "压力曲线 256 点是前缘锚定弧长序，s=linspace(0,1,256)。"),
}
print("\n压力曲线 POD 基 %s，均值长度 %d" % (cp_pod_phi.shape, cp_pod_mu.size))
assert cp_pod_phi.shape == (256, cp_r), "POD 基形状 %s 不符" % (cp_pod_phi.shape,)

json.dump(meta, open(os.path.join(OUT, "cp_basis_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
json.dump(check, open(os.path.join(OUT, "export_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n-> %s" % OUT)
print("EXPORT_DONE ort=%s allPassed=%s" % (ort is not None, check["allPassed"]))
if ort is not None and not check["allPassed"]:
    sys.exit(3)

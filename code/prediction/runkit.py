"""训练脚手架公共件：4 件套落盘 + nohup 日志 + watchdog + 冒烟闸门。

4 件套（每个训练任务必落盘到 runs/<task>_<链路>/）：
  1. hyperparams.json   超参 + 族别开关 + POD 基来源 + 任务划分名
  2. loss_history.csv   每 epoch train/val loss + 时间戳
  3. best_model.pt      val loss 最低时存（带 epoch 元信息）
  4. eval_metrics.json  指标 + 族别分层 + B 链路两段误差拆解 + 总时长

无人监管设计：
  - 长任务用 nohup 起，log 必落盘
  - watchdog：loss_history.csv 超过 STALL_SEC 没更新 -> 写 watchdog_alert.txt 并记
  - 冒烟闸门：100 样本 x 3 epoch，NaN/OOM/报错 -> 非零退出，不放行整批
"""
import os, json, time, math, csv, traceback
import numpy as np


class RunDir:
    """一个训练任务的落盘目录，封装 4 件套写入。"""

    def __init__(self, task, chain, root="04_results/prediction/runs"):
        self.dir = os.path.join(root, f"{task}_{chain}")
        os.makedirs(self.dir, exist_ok=True)
        self._t0 = time.time()

    def save_hyperparams(self, hp: dict):
        hp["_task"] = self._task_name()
        hp["_started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(os.path.join(self.dir, "hyperparams.json"), "w", encoding="utf-8") as f:
            json.dump(hp, f, ensure_ascii=False, indent=1, default=float)

    def _task_name(self):
        return os.path.basename(self.dir)

    def append_loss(self, epoch, train_loss, val_loss, extra=None):
        p = os.path.join(self.dir, "loss_history.csv")
        new = not os.path.exists(p)
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                cols = ["epoch", "train_loss", "val_loss", "ts"]
                cols += list(extra.keys()) if extra else []
                w.writerow(cols)
            row = [epoch, train_loss, val_loss, time.strftime("%H:%M:%S")]
            row += [extra.get(k, "") for k in (extra or {})]
            w.writerow(row)

    def last_loss_update(self):
        """watchdog 用：loss_history.csv 最后修改时间（秒）。"""
        p = os.path.join(self.dir, "loss_history.csv")
        return os.path.getmtime(p) if os.path.exists(p) else 0

    def save_best(self, state: dict, val_loss, epoch, norm_stats=None):
        """存最优权重。

        norm_stats：可选的归一化统计量（dict）。必须一并存入，否则权重文件
        不完整 —— 加载后无法对新样本做与训练一致的归一化，等于不可用。
        含 A 链路的 coef_norm（mean_in/std_in/mean_out/std_out）或
        B 链路的 smean/sstd/gmean/gstd + POD 基路径/超参。
        """
        p = os.path.join(self.dir, "best_model.pt")
        state["_val_loss"] = float(val_loss)
        state["_epoch"] = int(epoch)
        if norm_stats is not None:
            state["_norm_stats"] = norm_stats
        import torch
        torch.save(state, p)

    def save_eval(self, metrics: dict):
        metrics["_total_train_sec"] = round(time.time() - self._t0, 1)
        metrics["_finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(os.path.join(self.dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=1, default=float)


def is_nan(x):
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def smoke_gate(train_fn, n_smoke=100, n_epoch=3, label="smoke"):
    """冒烟闸门：train_fn(subset_idx) 跑 n_epoch；NaN/OOM/报错 -> 返回 False。"""
    rd = RunDir(label, "smoke")
    rd.save_hyperparams({"smoke": True, "n_smoke": n_smoke, "n_epoch": n_epoch})
    try:
        idx = list(range(n_smoke))
        for ep in range(n_epoch):
            tl, vl = train_fn(idx, ep, on_loss=lambda e, t, v: rd.append_loss(e, t, v))
            if is_nan(tl) or is_nan(vl):
                print(f"[smoke] NaN at epoch {ep} (train={tl} val={vl}) -> GATE FAIL")
                with open(os.path.join(rd.dir, "smoke_result.txt"), "w") as f:
                    f.write("FAIL: NaN\n")
                return False
        print(f"[smoke] {label} passed {n_epoch} epochs, no NaN/OOM -> GATE PASS")
        with open(os.path.join(rd.dir, "smoke_result.txt"), "w") as f:
            f.write("PASS\n")
        return True
    except MemoryError:
        print(f"[smoke] OOM -> GATE FAIL")
        with open(os.path.join(rd.dir, "smoke_result.txt"), "w") as f:
            f.write("FAIL: OOM\n")
        return False
    except Exception:
        print(f"[smoke] {label} raised -> GATE FAIL:\n{traceback.format_exc()}")
        with open(os.path.join(rd.dir, "smoke_result.txt"), "w") as f:
            f.write("FAIL: exception\n" + traceback.format_exc())
        return False


def watchdog_loop(rd: RunDir, stall_sec=1800, sleep=30, label=""):
    """轮询 loss_history.csv，超 stall_sec 没更新就告警。供 nohup 后台任务守护。"""
    while True:
        last = rd.last_loss_update()
        if last and (time.time() - last) > stall_sec:
            with open(os.path.join(rd.dir, "watchdog_alert.txt"), "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} loss stall > {stall_sec}s\n")
        time.sleep(sleep)

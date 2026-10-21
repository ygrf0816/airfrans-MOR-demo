"""P1 过夜驱动器（长时间轮询）：
  1) 等待 02_dataset\_EXTRACTED.OK 出现（外部下载/解压进程负责）
  2) 检查样本是否已就位，没有则从本地 zip 抽取
  3) 依次跑 p1 各分析脚本，每步打印进度
  4) 把日志写 09_tmp/overdrive.log
每 120s 打印一次心跳（下载百分比）。可反复调用，幂等。
"""
import os, sys, time, json, subprocess, glob

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MARKER = os.path.join(ROOT, '02_dataset', '_EXTRACTED.OK')
ZIPLOCAL = os.path.join(ROOT, '02_dataset', 'Dataset.zip')
DATASET = os.path.join(ROOT, '02_dataset', 'Dataset')
SAMPLES = os.path.join(ROOT, '04_results', 'analysis', 'samples')
SEL = os.path.join(ROOT, '04_results', 'analysis', 'selected_names.json')
LOG = os.path.join(ROOT, '09_tmp', 'overdrive.log')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def log(msg):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def samples_ready():
    return glob.glob(os.path.join(SAMPLES, '*', '*_internal.vtu'))


def wait_extracted(timeout_s=6 * 3600):
    """轮询 _EXTRACTED.OK；同时显示 F 盘下载进度。"""
    part = r'F:\temp download path for tandem\未确认 157643.crdownload'
    t0 = time.time()
    last_beat = 0
    while not os.path.exists(MARKER):
        if time.time() - t0 > timeout_s:
            log(f'[!] 等待 _EXTRACTED.OK 超时 {timeout_s}s，放弃全量部分，仅用样本')
            return False
        pct = '?'
        try:
            sz = os.path.getsize(part)
            pct = f'{sz/1048576:.0f} MB / 9534 MB ({100*sz/9534/1024/1024:.1f}%)'
        except Exception:
            pass
        zl = os.path.exists(ZIPLOCAL) and os.path.getsize(ZIPLOCAL) if os.path.exists(ZIPLOCAL) else 0
        now = time.time()
        if now - last_beat > 120:
            log(f'heartbeat: 下载 {pct} | Dataset.zip {zl/1048576:.0f} MB | marker={"有" if os.path.exists(MARKER) else "无"}')
            last_beat = now
        time.sleep(30)
    log(f'_EXTRACTED.OK 出现！等待 15s 让文件稳定…')
    time.sleep(15)
    return True


def ensure_samples():
    if len(samples_ready()) >= 30:
        log(f'样本已就位 {len(samples_ready())} 个 internal.vtu')
        return
    sel = json.load(open(SEL))['selected']
    log(f'抽样本 {len(sel)} 个（优先本地 zip，否则远端 Range）…')
    script = os.path.join(ROOT, '03_code', 'p1', 'extract_samples.py')
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    log(r.stdout[-2000:] if r.stdout else '(no out)')
    if r.returncode != 0:
        log(f'[!] 抽取脚本 rc={r.returncode}: {r.stderr[-500:]}')


def run_stage(name, script, logname):
    sp = os.path.join(ROOT, '03_code', 'p1', script)
    lp = os.path.join(ROOT, '09_tmp', logname)
    log(f'=== 开始 {name}: {script} ===')
    t0 = time.time()
    r = subprocess.run([sys.executable, sp], capture_output=True, text=True)
    with open(lp, 'w', encoding='utf-8') as fh:
        fh.write(f'=== {name} @ {time.strftime("%Y-%m-%d %H:%M:%S")}\nrc={r.returncode}\n\n--- stdout ---\n{r.stdout}\n\n--- stderr ---\n{r.stderr}\n')
    log(f'=== {name} rc={r.returncode} {time.time()-t0:.0f}s ===')
    tail = (r.stdout or '').strip().splitlines()
    for l in tail[-8:]:
        log('   ' + l)
    return r.returncode == 0


def main():
    log('===== 过夜驱动器启动 =====')
    ok = wait_extracted(timeout_s=8 * 3600)
    ensure_samples()
    # 顺序：先 3/4（样本即可），再 1/5/6/7（全量），最后 8 汇总
    if ok:
        run_stage('P1-03 网格几何(全量)', 'p1_mesh_full.py', 'p1_mesh_full.log')
        run_stage('P1-04 场统计(全量)', 'p1_field_full.py', 'p1_field_full.log')
        run_stage('P1-05 物理校验', 'p1_physics.py', 'p1_physics.log')
        run_stage('P1-06 划分重叠', 'p1_splits.py', 'p1_splits.log')
        run_stage('P1-01 清点(全量)', 'p1_inventory.py', 'p1_inventory.log')
        run_stage('P1-07 异常清单', 'p1_anomalies.py', 'p1_anomalies.log')
    else:
        log('[!] 全量数据未就绪，只跑样本级 03/04/05')
        run_stage('P1-03 网格几何(样本)', 'p1_mesh_sample.py', 'p1_mesh_sample.log')
        run_stage('P1-04 场统计(样本)', 'p1_field_sample.py', 'p1_field_sample.log')
        run_stage('P1-05 物理校验(样本)', 'p1_physics_sample.py', 'p1_physics_sample.log')
    run_stage('P1-08 汇总', 'p1_report.py', 'p1_report.log')
    log('===== 过夜驱动器结束 =====')


if __name__ == '__main__':
    main()

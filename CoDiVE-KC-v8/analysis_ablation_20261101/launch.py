#!/usr/bin/env python3
"""预约 #454 内的 8 个推理分析任务：每卡 2 个顺序执行；窗口结束前 3 分钟停止；状态同步 FocusForge。"""
import datetime as dt
import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = '/data/dyl/CoDiVE-KC-v7/.venv/bin/python'
V7, V8 = Path('/data/dyl/CoDiVE-KC-v7'), Path('/data/dyl/CoDiVE-KC-v8')
TZ = dt.timezone(dt.timedelta(hours=8))
QUEUE_ID = 'codive_v7_v8_evidence_ablation_20261101'
RES_ID = 454


def job(version, ds, seed):
    if version == 'v8':
        root, cfg = V8, V8 / f'config/{ds.lower()}_concept_evidence_seed{seed}.json'
    else:
        root = V7 if seed == 200408 else V7 / 'seed200409'
        cfg = V7 / ('config/' + f'{ds.lower()}_pair_evidence' + ('' if seed == 200408 else '_seed200409') + '.json')
    name = f'{version}_{ds}_seed{seed}'
    return dict(id=f'ablation_{name}', title=f'{version} {ds} seed{seed} 冻结模型证据开关消融+易混对导出（仅推理）',
                dataset=ds, seed=seed, status='WAITING_FOR_RESERVATION', gpus=[], reason='等待预约 #454',
                cmd=[PY, str(HERE / 'dump_components.py'), '--root', str(root), '--config', str(cfg),
                     '--out', str(HERE / 'dumps' / f'{name}.pt')])


PLAN = {0: [job('v8', 'Math', 200408), job('v7', 'Math', 200408)],
        1: [job('v8', 'Math', 200409), job('v7', 'Math', 200409)],
        2: [job('v8', 'Biology', 200408), job('v7', 'Biology', 200408)],
        3: [job('v8', 'Biology', 200409), job('v7', 'Biology', 200409)]}
JOBS = [j for js in PLAN.values() for j in js]


def now():
    return dt.datetime.now(TZ).replace(microsecond=0)


def window():
    con = sqlite3.connect('file:/data/gpu_booking/reservations.sqlite?mode=ro', uri=True)
    row = con.execute("select start_ts,end_ts,status,owner from reservations where id=?", (RES_ID,)).fetchone()
    con.close()
    if not row or row[2] != 'active' or row[3] != '邓郁霖':
        return None
    s, e = (dt.datetime.fromisoformat(x).replace(tzinfo=TZ) for x in row[:2])
    return (s, e) if s <= now() < e else None


def publish():
    sts = {j['status'] for j in JOBS}
    snap = dict(schema='focusforge.queue.v1', queue_id=QUEUE_ID, server='jnu', owner='邓郁霖',
                updated_at=now().isoformat(), status='COMPLETED' if sts <= {'COMPLETED', 'FAILED', 'CANCELLED'} else
                ('RUNNING' if 'RUNNING' in sts else 'WAITING_FOR_RESERVATION'),
                jobs=[{k: j[k] for k in ('id', 'title', 'status', 'dataset', 'seed', 'gpus', 'reason')} |
                      dict(family='CoDiVE-KC evidence ablation', resumable=False) for j in JOBS])
    f = HERE / 'focusforge_queue.json'
    f.write_text(json.dumps(snap, ensure_ascii=False, indent=2))
    subprocess.run(['python3', '/data/dyl/focusforge/focusforge_task_feed.py', 'publish', '--server', 'jnu',
                    '--file', str(f)], capture_output=True)
    time.sleep(1.1)       # 保证下次 updated_at 严格递增


def main():
    (HERE / 'dumps').mkdir(exist_ok=True)
    (HERE / 'logs').mkdir(exist_ok=True)
    publish()
    while not window():
        time.sleep(10)
    running = {}
    queues = {g: list(js) for g, js in PLAN.items()}
    while queues or running:
        w = window()
        for g in list(queues):
            if g in running or not queues[g]:
                queues.pop(g, None) if g not in running and not queues.get(g) else None
                continue
            if not w or (w[1] - now()).total_seconds() < 600:
                for j in queues.pop(g):
                    j.update(status='WAITING_FOR_RESERVATION', reason='#454 剩余时间不足，未启动')
                publish()
                continue
            j = queues[g].pop(0)
            log = open(HERE / 'logs' / f"{j['id']}.log", 'w')
            p = subprocess.Popen(j['cmd'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE='1',
                                 TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false'),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(HERE))
            running[g] = (j, p)
            j.update(status='RUNNING', gpus=[g], reason='')
            publish()
        for g, (j, p) in list(running.items()):
            rc = p.poll()
            if rc is None:
                if not w or (w[1] - now()).total_seconds() < 180:
                    os.killpg(p.pid, signal.SIGTERM)
                    time.sleep(20)
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGKILL)
                    j.update(status='PAUSED', gpus=[], reason='预约窗口将结束，已停止并释放 GPU')
                    running.pop(g)
                    publish()
                continue
            j.update(status='COMPLETED' if rc == 0 else 'FAILED', gpus=[],
                     reason='' if rc == 0 else f'退出码 {rc}，见 logs/{j["id"]}.log')
            running.pop(g)
            publish()
            if not queues.get(g):
                queues.pop(g, None)
        time.sleep(5)
    publish()
    print('ALL DONE', now().isoformat(), flush=True)


if __name__ == '__main__':
    main()

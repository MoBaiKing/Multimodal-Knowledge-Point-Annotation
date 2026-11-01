#!/usr/bin/env python3
"""CoDiVE-KC v8 JNU train and final GPU test queue: live bookings, independent GPU slots and GPU final tests.

No GPU work is performed by --check or --publish-only. A live runner publishes
the full stable-ID FocusForge queue before launching any worker. The pipeline
owns run/status.json and checkpoint/test verification; this controller owns
runs/queue/state.json and the public queue snapshot. SIGTERM asks the pipeline
to checkpoint, and inherited file locks survive an unexpected controller exit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Shanghai")
OWNER = "邓郁霖"
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
PIPELINE_STATUSES = {"RUNNING", "VALIDATING", "TRAINED", "TESTED", "FAILED", "PAUSED"}
REFERENCE_STATUSES = {"RUNNING", "PAUSED", "FAILED", "COMPLETED"}
TERMINAL = {"TESTED", "COMPLETED", "CANCELLED"}


def now() -> datetime:
    return datetime.now(TZ)


def stamp() -> str:
    return now().isoformat(timespec="microseconds")


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=TZ) if parsed.tzinfo is None else parsed.astimezone(TZ)


def read_json(path: Path, missing: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return missing


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict:
    cfg = read_json(path)
    if not isinstance(cfg, dict):
        raise ValueError("queue config must be a JSON object")
    if cfg.get("owner", OWNER) != OWNER or cfg.get("server", "jnu") != "jnu":
        raise ValueError("This queue only schedules 邓郁霖's JNU reservations")
    if not ID_PATTERN.fullmatch(cfg.get("queue_id", "")):
        raise ValueError("queue_id must be a stable FocusForge identifier")
    for key, default in (("poll_seconds", 5), ("save_margin_seconds", 300),
                         ("release_margin_seconds", 60), ("max_stop_seconds", 60)):
        cfg[key] = float(cfg.get(key, default))
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 1 <= cfg["poll_seconds"] <= 10:
        raise ValueError("poll_seconds must be between 1 and 10 for booking changes")
    if cfg["save_margin_seconds"] <= cfg["release_margin_seconds"] + cfg["poll_seconds"]:
        raise ValueError("save margin must leave time before the hard release margin")
    cfg.setdefault("candidate_gpus", [0, 1, 2, 3])
    # 2026-10-31 用户要求拉满利用率：允许本队列自己的任务同卡并行（只与本队列任务合卡，不与他人任务合卡）
    cfg["max_jobs_per_gpu"] = int(cfg.get("max_jobs_per_gpu", 1))
    if not 1 <= cfg["max_jobs_per_gpu"] <= 3:
        raise ValueError("max_jobs_per_gpu must be 1-3")
    if (not cfg["candidate_gpus"] or len(set(cfg["candidate_gpus"])) != len(cfg["candidate_gpus"])
            or any(type(g) is not int or g not in range(4) for g in cfg["candidate_gpus"])):
        raise ValueError("candidate_gpus must be a unique nonempty subset of GPU 0–3")
    cfg["queue_dir"] = str(resolve(cfg.get("queue_dir", "runs/queue")))
    jobs = cfg.get("jobs", [])
    ids, directories = set(), set()
    if not jobs:
        raise ValueError("queue must include at least one job")
    for job in jobs:
        if not ID_PATTERN.fullmatch(job.get("id", "")) or job["id"] in ids:
            raise ValueError("jobs need unique stable IDs")
        ids.add(job["id"])
        if job.get("dataset") not in {"Math", "Biology"} or type(job.get("seed")) is not int:
            raise ValueError("job dataset must be Math/Biology and seed must be an integer")
        if job["seed"] < 0:
            raise ValueError("seed must be nonnegative")
        for key in ("config", "run_dir"):
            job[key] = str(resolve(job[key]))
        if job["run_dir"] in directories:
            raise ValueError("each job must have a distinct run_dir")
        directories.add(job["run_dir"])
        job.setdefault("enabled", True)
        if type(job["enabled"]) is not bool:
            raise ValueError("enabled must be a boolean")
        job.setdefault("priority", 100)
        job.setdefault("kind", "main")
        if job["kind"] not in {"main", "reference"}:
            raise ValueError("job kind must be main or reference")
        job.setdefault("hold_reason", "等待首轮 H1/H2 审查后启用")
    return cfg


def reservations(cfg: dict) -> list[dict]:
    """The host-local database is JNU's authority; never write bookings."""
    path = Path(cfg.get("reservation_db", "/data/gpu_booking/reservations.sqlite"))
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3) as db:
        rows = db.execute("SELECT id,owner,gpus,start_ts,end_ts FROM reservations WHERE status='active'").fetchall()
    result = []
    for identity, owner, gpu_text, start, end in rows:
        gpus = [int(g.strip()) for g in gpu_text.split(",") if g.strip()]
        start, end = parse_time(start), parse_time(end)
        if end <= start:
            raise ValueError(f"Invalid reservation interval: {identity}")
        result.append(dict(id=identity, owner=owner, gpus=gpus, start=start, end=end))
    return result


def window_for(rows: list[dict], gpu: int, point: datetime) -> dict | None:
    """Merge adjacent own bookings, clipping at any conflicting foreign booking."""
    own = sorted((r for r in rows if r["owner"] == OWNER and gpu in r["gpus"]),
                 key=lambda row: row["start"])
    current = [r for r in own if r["start"] <= point < r["end"]]
    foreign = [r for r in rows if r["owner"] != OWNER and gpu in r["gpus"]]
    if not current or any(r["start"] <= point < r["end"] for r in foreign):
        return None
    start, end = min(r["start"] for r in current), max(r["end"] for r in current)
    ids = {r["id"] for r in current}
    for row in own:
        if row["start"] <= end and row["end"] > end:
            end = row["end"]
            ids.add(row["id"])
    for row in foreign:
        if point < row["start"] < end:
            end = row["start"]
    return dict(start=start, end=end, reservation_ids=sorted(ids))


def next_window(rows: list[dict], candidates: list[int], point: datetime) -> dict | None:
    future = sorted({r["start"] for r in rows if r["owner"] == OWNER and r["start"] > point
                     and set(r["gpus"]) & set(candidates)})
    for start in future:
        for gpu in candidates:
            window = window_for(rows, gpu, start)
            if window:
                return dict(gpu=gpu, **window)
    return None


def gpu_inventory() -> dict[int, dict]:
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    busy = {line.split(",", 1)[0].strip() for line in processes.stdout.splitlines() if "," in line}
    result = {}
    for line in inventory.stdout.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
        result[int(index)] = dict(uuid=uuid, memory_mib=int(memory), utilization=int(utilization),
                                  idle=uuid not in busy and int(memory) < 512 and int(utilization) < 5)
    return result


def lock_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def claim_gpu(cfg: dict, gpu: int, uuid: str) -> list:
    # Respect both the current LEDGER dispatch lock and older reproduction locks.
    paths = [Path(directory) / f"gpu{gpu}.lock" for directory in cfg.get("gpu_lock_dirs", [
        "/data/dyl/LEDGER-FND/workspaces/scheduled/.gpu_dispatch_locks",
        "/data/dyl/reproduce2/gpu_locks"])]
    paths += [Path(directory) / f"{uuid}.lock" for directory in cfg.get("gpu_uuid_lock_dirs", [
        "/data/dyl/LEDGER-FND/ablation/RQ2/queues/locks"])]
    handles = []
    for path in sorted(set(paths)):
        handle = lock_file(path)
        if handle is None:
            for previous in handles:
                previous.close()
            return []
        handles.append(handle)
    return handles


def process_start(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def pipeline_status(job: dict) -> dict:
    status = read_json(Path(job["run_dir"]) / "status.json", {})
    if not isinstance(status, dict):
        raise ValueError("pipeline status must be an object")
    allowed = REFERENCE_STATUSES if job.get("kind", "main") == "reference" else PIPELINE_STATUSES
    if status and status.get("status") not in allowed:
        raise ValueError(f"Invalid {job.get('kind', 'main')} status {status.get('status')!r}")
    return status


def command(cfg: dict, job: dict) -> list[str]:
    # 参考教师任务可指定自己的解释器与入口（seed200409 MacBERT 参考沿用 v5-RL 的 dyl_reproduce 环境）
    return [job.get("python", cfg["python"]), job.get("script", "scripts/worker.py"), "--config", job["config"],
            "--device", "cuda:0"]


def completed(job: dict, row: dict) -> bool:
    """A completed auxiliary OOF worker cannot complete a main experiment."""
    expected = "COMPLETED" if job.get("kind", "main") == "reference" else "TESTED"
    return row["status"] in {expected, "CANCELLED"}


def selection_provenance(job: dict) -> None:
    """Link the actual OOF selection threshold without mutating frozen weights.

    The checkpoint's legacy calibration field describes the initial text
    temperature artifact. This sidecar identifies visual selection and final
    deployment calibration separately.
    """
    directory = Path(job['run_dir'])
    complete = directory/'visual/complete.json'
    training = directory/'visual/training_calibration.json'
    checkpoint = directory/'visual/best.pt'
    if not all(path.is_file() for path in (complete, training, checkpoint)):
        return
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    ledger = read_json(complete)
    final = directory/'frozen_test_calibration.json'
    record = dict(signature=ledger['signature'], checkpoint_sha256=ledger['checkpoint_sha256'],
                  visual_training_calibration_sha256=digest(training),
                  final_test_calibration_sha256=digest(final) if final.is_file() else None,
                  selection_calibration_path=str(training), final_calibration_path=str(final),
                  note='Visual checkpoint and fusion policy selected on val_select with its provisional q; final deployment uses frozen val_calibration q. Frozen v4 reference OOF is used only for training candidates and hardness weighting.')
    target = directory/'visual/selection_provenance.json'
    previous = read_json(target, {})
    if all(previous.get(key) == value for key, value in record.items()):
        return
    if digest(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Visual checkpoint differs from completion record')
    write_json(target, dict(updated_at=stamp(), **record))


def priority(row: dict, job: dict) -> tuple:
    # GPU final tests and interrupted work always precede a fresh student run.
    rank = 0 if row["status"] == "TRAINED" else 1 if row.get("resumable") else 2
    role = 1 if job.get("kind", "main") == "reference" else 0
    return rank, role, job["priority"], job["seed"], job["dataset"]


@dataclass
class Worker:
    job_id: str
    gpu: int
    pid: int
    start: str
    process: Any = None
    handles: list = field(default_factory=list)
    stream: Any = None
    stopping_reason: str = ""
    kill_at: datetime | None = None

    def poll(self):
        if self.process is not None:
            return self.process.poll()
        return None if process_start(self.pid) == self.start else -255


class Runner:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.cfg = load_config(cfg_path)
        self.directory = Path(self.cfg["queue_dir"])
        self.state_path = self.directory / "state.json"
        self.public_path = self.directory / "focusforge.json"
        self.state = read_json(self.state_path, {"jobs": {}, "created_at": stamp()})
        self.workers: dict[str, Worker] = {}
        self.stop = False
        self.last_public = ""
        self.sync_config(self.cfg)

    def event(self, kind: str, **details) -> None:
        row = dict(time=stamp(), event=kind, **details)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def sync_config(self, cfg: dict) -> None:
        if cfg["queue_id"] != self.cfg["queue_id"] or cfg["queue_dir"] != self.cfg["queue_dir"]:
            raise ValueError("queue_id and queue_dir cannot change while a controller runs")
        previous_ids = set(self.state["jobs"])
        current_ids = {j["id"] for j in cfg["jobs"]}
        if previous_ids - current_ids:
            raise ValueError("Keep historical jobs in config; set enabled=false instead of removing IDs")
        for job in cfg["jobs"]:
            row = self.state["jobs"].setdefault(job["id"], dict(status="PENDING", resumable=False,
                        attempts=0, reason="", updated_at=stamp()))
            identity = {key: job[key] for key in ("dataset", "seed", "config", "run_dir")}
            if row.get("identity", identity) != identity:
                raise ValueError(f"Stable job identity changed: {job['id']}")
            row["identity"] = identity
        self.cfg = cfg

    def refresh_status(self, job: dict) -> None:
        row = self.state["jobs"][job["id"]]
        try:
            status = pipeline_status(job)
        except (OSError, ValueError) as exc:
            row.update(status="BLOCKED", reason=f"Cannot read pipeline state: {exc}", updated_at=stamp())
            return
        if status:
            # Never replace a controller failure with a stale prelaunch state.
            launched_at = row.get("launched_at")
            fresh = not launched_at or parse_time(status.get("updated_at", "1970-01-01")) >= parse_time(launched_at)
            if fresh or row["status"] in {"PENDING", "HELD", "WAITING_FOR_RESERVATION", "QUEUED"}:
                row.update({key: status[key] for key in ("status", "stage", "reason", "resumable", "updated_at") if key in status})
                row['reason'] = status.get('reason', '')
            if job.get('kind', 'main') == 'main' and status.get('status') in {'TRAINED', 'TESTED'}:
                selection_provenance(job)
        if row["status"] == "COMPLETED" and job.get("kind", "main") == "main":
            row.update(status="BLOCKED", reason="Main experiment requires frozen final TESTED status; COMPLETED is reserved for reference workers", updated_at=stamp())
        if job["id"] not in self.workers and row["status"] in {"RUNNING", "VALIDATING"}:
            row.update(status="PAUSED", reason="Controller recovered; no live worker remains",
                       resumable=bool(status.get("resumable")), updated_at=stamp())

    def adopt(self) -> None:
        for job in self.cfg["jobs"]:
            row = self.state["jobs"][job["id"]]
            if row.get("pid") and process_start(row["pid"]) == row.get("process_start"):
                self.workers[job["id"]] = Worker(job["id"], row["gpu"], row["pid"], row["process_start"])
                self.event("worker_adopted", job=job["id"], pid=row["pid"], gpu=row["gpu"])

    def request_stop(self, worker: Worker, reason: str, deadline: datetime) -> None:
        if not worker.stopping_reason:
            worker.stopping_reason = reason
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.event("checkpoint_requested", job=worker.job_id, gpu=worker.gpu, reason=reason)
        worker.kill_at = min(worker.kill_at, deadline) if worker.kill_at else deadline

    def reconcile(self, rows: list[dict] | None) -> None:
        point = now()
        for job in self.cfg["jobs"]:
            identity = job["id"]
            worker = self.workers.get(identity)
            row = self.state["jobs"][identity]
            if worker:
                code = worker.poll()
                if code is not None:
                    # Release locks as soon as GPU work exits, before bookkeeping.
                    for handle in worker.handles:
                        handle.close()
                    if worker.stream:
                        worker.stream.close()
                    del self.workers[identity]
                    self.refresh_status(job)
                    if not completed(job, row):
                        if worker.stopping_reason:
                            row.update(status="TRAINED" if row["status"] == "TRAINED" else "PAUSED",
                                       reason=worker.stopping_reason, updated_at=stamp())
                        elif code == 75 and row["status"] == "PAUSED" and row.get("resumable"):
                            # Cooperative OOF workers yield their GPU while another
                            # worker owns the missing folds; this is a normal retry.
                            pass
                        elif code != 0 or (row["status"] != "TRAINED" and not (
                                row["status"] == "PAUSED" and row.get("resumable"))):
                            missing = "OOF reference completion missing" if job.get("kind") == "reference" else "final GPU test incomplete"
                            row.update(status="FAILED", reason=f"Worker exited {code}; {missing}",
                                       updated_at=stamp())
                            row["failures"] = row.get("failures", 0) + 1
                        # A finished student waiting for its final test is immediately ready.
                        row["retry_after"] = (point + timedelta(seconds=0 if row["status"] == "TRAINED"
                            else self.cfg.get("retry_seconds", 60))).isoformat()
                    row.update(exit_code=code, finished_at=stamp(), last_gpu=worker.gpu)
                    for key in ("pid", "process_start", "gpu"):
                        row.pop(key, None)
                    self.event("worker_exited", job=identity, exit_code=code, status=row["status"])
                    continue
                self.refresh_status(job)
                window = window_for(rows, worker.gpu, point) if rows is not None else None
                if window:
                    row.update(reservation_ids=window["reservation_ids"],
                               planned_start=window["start"].isoformat(), planned_end=window["end"].isoformat())
                    save_at = window["end"] - timedelta(seconds=self.cfg["save_margin_seconds"])
                    release_at = window["end"] - timedelta(seconds=self.cfg["release_margin_seconds"])
                    if point >= save_at:
                        self.request_stop(worker, "预约保存时间已到，保存断点并等待后续有效预约", release_at)
                else:
                    # Cancellation/revocation has no authorized grace period.
                    self.request_stop(worker, "预约已取消、范围变化或预约库不可读，立即释放 GPU", point)
                if not job["enabled"]:
                    self.request_stop(worker, job["hold_reason"], point + timedelta(seconds=self.cfg["max_stop_seconds"]))
                if self.stop:
                    self.request_stop(worker, "调度器收到停止信号，保存断点", point + timedelta(seconds=self.cfg["max_stop_seconds"]))
                if worker.kill_at and point >= worker.kill_at:
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.event("hard_release", job=identity, gpu=worker.gpu, reason=worker.stopping_reason)
            else:
                self.refresh_status(job)
                if completed(job, row):
                    continue
                if (row["status"] == "BLOCKED" and row.get("reason", "").startswith("缺少实验配置:")
                        and Path(job["config"]).is_file()):
                    row.update(status="QUEUED", reason="实验配置已就绪，等待任一授权 GPU", updated_at=stamp())
                if not job["enabled"]:
                    row.update(status="HELD", reason=job["hold_reason"], updated_at=stamp())
                elif row["status"] not in {"TRAINED", "PAUSED", "FAILED", "BLOCKED"}:
                    active = rows is not None and any(window_for(rows, g, point) for g in self.cfg["candidate_gpus"])
                    row.update(status="QUEUED" if active else "WAITING_FOR_RESERVATION",
                               reason="等待任一授权 GPU 空闲" if active else "等待 gpubook 本人最新有效预约", updated_at=stamp())
                elif row["status"] == "HELD":
                    row.update(status="QUEUED", reason="已启用，等待任一授权 GPU", updated_at=stamp())

    def publish(self, rows: list[dict] | None, force: bool = False) -> None:
        point = now()
        active = bool(self.workers)
        all_complete = all(completed(j, self.state["jobs"][j["id"]]) for j in self.cfg["jobs"])
        queue_status = "RUNNING" if active else "COMPLETED" if all_complete else "PAUSED" if self.stop else "QUEUED"
        self.state.update(updated_at=stamp(), scheduler_pid=os.getpid(), status=queue_status,
                          booking_error=self.state.get("booking_error", ""))
        write_json(self.state_path, self.state)
        future = next_window(rows, self.cfg["candidate_gpus"], point) if rows is not None else None
        jobs = []
        for job in self.cfg["jobs"]:
            row = self.state["jobs"][job["id"]]
            worker = self.workers.get(job["id"])
            window = window_for(rows, worker.gpu, point) if rows is not None and worker else future
            reference = job.get("kind", "main") == "reference"
            title = (f"CoDiVE-KC v8 {job['dataset']} seed{job['seed']} OOF 5-fold 辅助（同一种子）" if reference
                     else f"CoDiVE-KC v8 {job['dataset']} seed{job['seed']}")
            title = job.get('title', title)
            jobs.append(dict(id=job["id"], title=title,
                             family=job.get('family', "DIVE-KC-v4-reference" if reference else "DIVE-KC-v4"), dataset=job["dataset"], seed=job["seed"], status=row["status"],
                             gpus=[worker.gpu] if worker else [], reason=row.get("reason") or (
                                 f"阶段: {row['stage']}" if row.get("stage") else ""),
                             resumable=bool(row.get("resumable")), updated_at=row.get("updated_at", stamp()),
                             planned_start=window["start"].isoformat() if window else None,
                             planned_end=window["end"].isoformat() if window else None))
        public = dict(schema="focusforge.queue.v1", queue_id=self.cfg["queue_id"], server="jnu", owner=OWNER,
                      updated_at=stamp(), status=queue_status, jobs=jobs)
        # Ignore heartbeat timestamps when deciding whether a fresh publication is needed.
        signature = json.dumps({"status": queue_status, "jobs": [
            {k: v for k, v in j.items() if k != "updated_at"} for j in jobs]}, sort_keys=True, ensure_ascii=False)
        if not force and signature == self.last_public:
            return
        write_json(self.public_path, public)
        result = subprocess.run([self.cfg.get("feed_python", "python3"),
            self.cfg.get("focusforge_feed", "/data/dyl/focusforge/focusforge_task_feed.py"),
            "publish", "--server", "jnu", "--file", str(self.public_path)],
            capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError("FocusForge publish failed: " + result.stderr.strip() + result.stdout.strip())
        self.last_public = signature

    def eligible(self, job: dict) -> bool:
        row = self.state["jobs"][job["id"]]
        if not job["enabled"] or job["id"] in self.workers or completed(job, row) or row["status"] in {"BLOCKED", "HELD"}:
            return False
        if row["status"] == "FAILED" and (not row.get("resumable") or row.get("failures", 0) >= self.cfg.get("max_attempts", 3)):
            return False
        return not row.get("retry_after") or now() >= parse_time(row["retry_after"])

    def launch_available(self) -> None:
        candidates = sorted((j for j in self.cfg["jobs"] if self.eligible(j)),
                            key=lambda j: priority(self.state["jobs"][j["id"]], j))
        if not candidates:
            return
        inventory = gpu_inventory()
        next_row = self.state["jobs"][candidates[0]["id"]]
        preferred = next_row.get("last_gpu") if next_row["status"] == "TRAINED" else None
        gpu_order = sorted(self.cfg["candidate_gpus"], key=lambda gpu: gpu != preferred)
        own = {}
        for w in self.workers.values():
            own[w.gpu] = own.get(w.gpu, 0) + 1
        # 先填空卡，再在本队列已占的卡上合卡（每卡最多 max_jobs_per_gpu 个），每轮最多新合一张卡
        gpu_order = sorted(gpu_order, key=lambda gpu: own.get(gpu, 0))
        packed_this_round = False
        for gpu in gpu_order:
            count = own.get(gpu, 0)
            if not candidates or count >= self.cfg["max_jobs_per_gpu"]:
                continue
            packing = count > 0
            if packing:
                if packed_this_round:
                    continue
                handles = []          # 共享 GPU 锁已由本队列在该卡上的任务持有
            else:
                if gpu not in inventory or not inventory[gpu]["idle"]:
                    continue
                handles = claim_gpu(self.cfg, gpu, inventory[gpu]["uuid"])
                if not handles:
                    continue
            stream = None
            try:
                # Recheck bookings AND actual occupancy after obtaining shared locks.
                rows = reservations(self.cfg)
                window = window_for(rows, gpu, now())
                if (not window or (window["end"] - now()).total_seconds() <= self.cfg["save_margin_seconds"]
                        or (not packing and not gpu_inventory().get(gpu, {}).get("idle"))):
                    continue
                packed_this_round = packed_this_round or packing
                job = candidates.pop(0)
                if not Path(job["config"]).is_file():
                    self.state["jobs"][job["id"]].update(status="BLOCKED", reason="缺少实验配置: " + job["config"], updated_at=stamp())
                    continue
                job_lock = lock_file(Path(job["run_dir"]) / ".pipeline.lock")
                if job_lock is None:
                    continue
                handles.append(job_lock)
                self.publish(rows)  # Registration must succeed before each new/resumed process.
                env = os.environ.copy()
                for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                    env.pop(key, None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
                self.directory.mkdir(parents=True, exist_ok=True)
                stream = (self.directory / f"{job['id']}.log").open("a")
                row = self.state["jobs"][job["id"]]
                launched_at = stamp()
                process = subprocess.Popen(command(self.cfg, job), cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(h.fileno() for h in handles))
                start = process_start(process.pid)
                worker = Worker(job["id"], gpu, process.pid, start, process, handles, stream)
                self.workers[job["id"]] = worker
                row.update(status="RUNNING", reason="", updated_at=launched_at, launched_at=launched_at,
                           attempts=row["attempts"] + 1, pid=process.pid, process_start=start, gpu=gpu,
                           reservation_ids=window["reservation_ids"], planned_start=window["start"].isoformat(),
                           planned_end=window["end"].isoformat())
                handles, stream = [], None
                own[gpu] = own.get(gpu, 0) + 1
                self.event("worker_launched", job=job["id"], gpu=gpu, pid=process.pid, packed=packing,
                           reservation_ids=window["reservation_ids"], command=command(self.cfg, job))
                self.publish(rows, force=True)
            finally:
                for handle in handles:
                    handle.close()
                if stream:
                    stream.close()

    def run(self, publish_only: bool = False) -> int:
        guard = lock_file(self.directory / "queue.lock")
        if guard is None:
            raise RuntimeError("A DIVE-KC controller already owns this queue")
        try:
            self.adopt()
            if publish_only and self.workers:
                raise RuntimeError("--publish-only found a live orphan; use normal run to supervise it")
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, lambda *_: setattr(self, "stop", True))
            while True:
                rows = None
                configuration_ok = True
                try:
                    self.sync_config(load_config(self.cfg_path))
                    rows = reservations(self.cfg)
                    self.state["booking_error"] = ""
                except Exception as exc:
                    configuration_ok = False
                    self.state["booking_error"] = str(exc)
                    self.event("authority_or_configuration_error", error=str(exc))
                self.reconcile(rows)
                published = False
                try:
                    self.publish(rows)
                    published = True
                except Exception as exc:
                    self.event("focusforge_error", error=str(exc))
                if publish_only:
                    return 0 if published else 1
                if self.stop and not self.workers:
                    return 0
                if configuration_ok and published and not self.stop:
                    try:
                        self.launch_available()
                    except Exception as exc:
                        self.event("dispatch_error", error=str(exc))
                if not self.workers and all(completed(j, self.state["jobs"][j["id"]]) for j in self.cfg["jobs"]):
                    return 0
                # One-second checks respect hard release even when poll_seconds is 5.
                wake = time.monotonic() + (1 if any(w.stopping_reason for w in self.workers.values()) else self.cfg["poll_seconds"])
                while not self.stop and time.monotonic() < wake:
                    time.sleep(min(1, max(0, wake - time.monotonic())))
                if self.stop:
                    time.sleep(0.1)
        finally:
            guard.close()


def check(cfg: dict) -> dict:
    rows = reservations(cfg)
    point = now()
    windows = {}
    for gpu in cfg["candidate_gpus"]:
        window = window_for(rows, gpu, point)
        windows[str(gpu)] = {k: v.isoformat() if isinstance(v, datetime) else v
                             for k, v in window.items()} if window else None
    try:
        inventory = gpu_inventory()
        gpu_error = ""
    except Exception as exc:
        inventory, gpu_error = {}, str(exc)
    return dict(checked_at=point.isoformat(), queue_id=cfg["queue_id"], windows=windows,
                gpu_inventory=inventory, gpu_error=gpu_error,
                enabled_count=sum(j["enabled"] for j in cfg["jobs"]),
                enabled_roles={kind: sum(j["enabled"] and j.get("kind", "main") == kind for j in cfg["jobs"])
                               for kind in ("main", "reference")},
                jobs=[dict(id=j["id"], kind=j.get("kind", "main"), enabled=j["enabled"], config_exists=Path(j["config"]).is_file(),
                           run_dir=j["run_dir"], command=command(cfg, j)) for j in cfg["jobs"]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/queue.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only booking/GPU/command inspection")
    mode.add_argument("--publish-only", action="store_true", help="Publish stable jobs to FocusForge, without launching GPU work")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(load_config(args.config)), ensure_ascii=False, indent=2))
        return 0
    return Runner(args.config).run(publish_only=args.publish_only)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        print(f"queue_runner: {error}", file=sys.stderr)
        raise SystemExit(1)

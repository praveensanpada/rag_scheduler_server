import os
import re
import time
import signal
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import yaml
from dotenv import load_dotenv

import logging
from logging.handlers import TimedRotatingFileHandler

# =========================
# Env & Config
# =========================
load_dotenv()

CONFIG_FILE = os.getenv("CONFIG_FILE", "./config.yaml")
RUN_ONCE = os.getenv("RUN_ONCE", "0") == "1"

GLOBAL_TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "1200"))   # 20 min default
GLOBAL_RETRY_MAX = int(os.getenv("RETRY_MAX", "3"))
GLOBAL_RETRY_BASE_MS = int(os.getenv("RETRY_BASE_MS", "1200"))

LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DEFAULT_ITEMS_KEYS = ["data", "result", "rows", "items", "users"]  # just in case some jobs return arrays

_stop_flag = False
def _handle_stop(signum, frame):
    global _stop_flag
    _stop_flag = True
signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

# =========================
# Logging Setup
# =========================
def ensure_logs_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception as e:
        print(f"[FATAL] Cannot create log dir {LOG_DIR}: {e}")

def _fmt():
    return logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")

def build_root_logger() -> logging.Logger:
    ensure_logs_dir()
    logger = logging.getLogger("poller")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    ch.setFormatter(_fmt())
    logger.addHandler(ch)

    fh = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "app.log"),
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fh.setFormatter(_fmt())
    logger.addHandler(fh)

    return logger

ROOT_LOGGER = build_root_logger()

def get_job_logger(job_name: str) -> logging.Logger:
    logger = logging.getLogger(f"poller.{job_name}")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    ch.setFormatter(_fmt())
    logger.addHandler(ch)

    fh = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, f"{job_name}.log"),
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fh.setFormatter(_fmt())
    logger.addHandler(fh)

    return logger

# =========================
# Utilities
# =========================
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def sleep_ms(ms: int):
    if ms <= 0:
        return
    end_at = time.time() + (ms / 1000.0)
    while not _stop_flag and time.time() < end_at:
        time.sleep(0.2)

def backoff_ms(attempt: int, base_ms: int, cap_ms: int = 60_000) -> int:
    # exponential backoff with light jitter
    jitter = int(250 * (os.urandom(1)[0] / 255.0))
    return min(cap_ms, base_ms * (2 ** max(0, attempt - 1))) + jitter

def _resolve_dot_path(obj: Any, path: str) -> Optional[Any]:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

# =========================
# Payload extraction
# =========================
def _extract_items(payload: Any, items_key: Union[str, List[str], None]) -> List[Any]:
    # Top-level array
    if isinstance(payload, list):
        return payload

    # Specific key(s)
    if isinstance(items_key, str) and items_key.strip():
        val = _resolve_dot_path(payload, items_key)
        if isinstance(val, list):
            return val
    elif isinstance(items_key, list) and items_key:
        for k in items_key:
            val = _resolve_dot_path(payload, str(k))
            if isinstance(val, list):
                return val

    # Defaults
    if isinstance(payload, dict):
        for key in DEFAULT_ITEMS_KEYS:
            val = payload.get(key)
            if isinstance(val, list):
                return val

    return []

def _extract_count_from_key(payload: Any, count_key: Optional[str]) -> Optional[int]:
    if not count_key:
        return None
    val = _resolve_dot_path(payload, count_key)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return None

def _extract_count_from_message(payload: Any, regex: Optional[str]) -> Optional[int]:
    if not regex or not isinstance(payload, dict):
        return None
    msg = payload.get("message")
    if not isinstance(msg, str):
        return None
    m = re.search(regex, msg)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (IndexError, ValueError):
        return None

def get_count_and_items(payload: Any, job: Dict[str, Any]) -> Tuple[int, List[Any]]:
    """
    Precedence:
      1) items (array) → count = len(items)
      2) count_key (dot path; e.g., "data.total_records")
      3) message_count_regex (e.g., r"is\\s+(\\d+)$")
      4) fallback: 0
    """
    items_key = job.get("items_key")
    items = _extract_items(payload, items_key)
    if items:
        return len(items), items

    count_key = job.get("count_key")
    c1 = _extract_count_from_key(payload, count_key)
    if c1 is not None:
        return c1, []

    msg_regex = job.get("message_count_regex")
    c2 = _extract_count_from_message(payload, msg_regex)
    if c2 is not None:
        return c2, []

    return 0, []

# =========================
# HTTP (logs URL & status)
# =========================
def fetch_with_retry(joblog: logging.Logger, job_name: str, url: str,
                     timeout_seconds: int, retry_max: int, retry_base_ms: int) -> Optional[Any]:
    attempt = 0
    last_exc = None
    # keep connect timeout small, read timeout large (so very slow responses can finish)
    timeout_tuple = (10, timeout_seconds)  # (connect, read)

    while attempt <= retry_max and not _stop_flag:
        try:
            joblog.info(f"[REQUEST] {job_name} GET {url}")
            r = requests.get(url, timeout=timeout_tuple)
            status = r.status_code
            joblog.info(f"[RESPONSE] {job_name} {status} {r.reason} url={url}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt > retry_max:
                break
            status = getattr(e, "response", None).status_code if hasattr(e, "response") and e.response else None
            retryable = (status is None) or (status in (408, 425, 429, 500, 502, 503, 504))
            if not retryable:
                joblog.error(f"[ERROR-NONRETRY] {job_name} url={url} err={e}")
                break
            wait = backoff_ms(attempt, retry_base_ms)
            joblog.warning(f"[RETRY] {job_name} attempt {attempt}/{retry_max} wait={wait}ms url={url} status={status or 'NO_RESPONSE'} err={e}")
            sleep_ms(wait)

    joblog.error(f"[GIVEUP] {job_name} url={url} err={last_exc}")
    return None

# =========================
# One full paginated run
# =========================
def run_once_job(job: Dict[str, Any]) -> Dict[str, float]:
    """
    Run a single job once: paginate until last page.
    Returns summary: {"pages": X, "records": Y, "runtime_s": float, "final_status": "..."}
    """
    name = job.get("name", "job")
    joblog = get_job_logger(name)

    base_url = job["base_url"]
    limit = int(job.get("limit", 10000))
    start_page = int(job.get("start_page", 1))
    max_pages = int(job.get("max_pages", 0))  # 0 = no cap

    timeout_seconds = int(job.get("timeout_seconds", GLOBAL_TIMEOUT_SECONDS))
    retry_max = int(job.get("retry_max", GLOBAL_RETRY_MAX))
    retry_base_ms = int(job.get("retry_base_ms", GLOBAL_RETRY_BASE_MS))

    started = time.time()
    page = start_page
    total_records = 0
    total_pages = 0
    had_error = False

    joblog.info(f"▶ START {name} page={page} limit={limit} max_pages={max_pages or '∞'} (timeout={timeout_seconds}s, retries={retry_max})")

    while not _stop_flag:
        if max_pages and total_pages >= max_pages:
            joblog.warning(f"⛔ {name} reached max_pages={max_pages}")
            break

        url = f"{base_url}?page={page}&limit={limit}"
        payload = fetch_with_retry(joblog, name, url, timeout_seconds, retry_max, retry_base_ms)
        if payload is None:
            joblog.warning(f"[STOP] {name} stopping run due to error at page={page}")
            had_error = True
            break

        count, items = get_count_and_items(payload, job)
        total_records += int(count)
        total_pages += 1
        joblog.info(f"[PAGE] {name} page={page} count={count} total_records={total_records}")

        # If array results are ever returned, process 'items' here.

        if count < limit:
            joblog.info(f"🏁 {name} last page likely reached (count {count} < limit {limit})")
            break

        page += 1

    runtime_s = round(time.time() - started, 2)
    final_status = "success"
    if total_pages == 0 or total_records == 0:
        final_status = "empty" if not had_error else "partial"
    elif had_error:
        final_status = "partial"

    joblog.info(f"✔ DONE {name} pages={total_pages} records={total_records} runtime={runtime_s}s final_status={final_status}")
    ROOT_LOGGER.info(f"[SUMMARY] {name} pages={total_pages} records={total_records} runtime={runtime_s}s final_status={final_status}")

    return {"pages": float(total_pages), "records": float(total_records), "runtime_s": float(runtime_s)}

# =========================
# Scheduling (non-overlap by default)
# =========================
def _loop_no_overlap(job: Dict[str, Any]):
    name = job.get("name", "job")
    joblog = get_job_logger(name)

    interval_minutes = int(job.get("interval_minutes", 10))
    interval_seconds = max(1, interval_minutes * 60)
    schedule_mode = job.get("schedule_mode", "fixed_delay")

    joblog.info(f"⏱  LOOP start {name} every {interval_minutes} min ({schedule_mode}, overlap=false)")

    if RUN_ONCE:
        run_once_job(job)
        joblog.info(f"⏹  LOOP stop {name} (RUN_ONCE)")
        return

    while not _stop_flag:
        start_t = time.time()
        summary = run_once_job(job)
        if _stop_flag:
            break

        runtime = summary.get("runtime_s", time.time() - start_t)
        sleep_sec = interval_seconds if schedule_mode != "fixed_rate" else max(0, interval_seconds - runtime)
        if sleep_sec > 0:
            joblog.info(f"⏳ {name} sleeping {int(sleep_sec)}s")
        end_at = time.time() + sleep_sec
        while not _stop_flag and time.time() < end_at:
            time.sleep(0.2)

    joblog.info(f"⏹  LOOP stop {name}")

def _loop_allow_overlap(job: Dict[str, Any]):
    name = job.get("name", "job")
    joblog = get_job_logger(name)

    interval_minutes = int(job.get("interval_minutes", 10))
    interval_seconds = max(1, interval_minutes * 60)

    joblog.info(f"⏱  LOOP start {name} every {interval_minutes} min (overlap=true)")

    def _runner():
        try:
            run_once_job(job)
        except Exception as e:
            joblog.error(f"[THREAD-ERROR] {name} run thread failed: {e}")

    if RUN_ONCE:
        _runner()
        joblog.info(f"⏹  LOOP stop {name} (RUN_ONCE)")
        return

    while not _stop_flag:
        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        end_at = time.time() + interval_seconds
        while not _stop_flag and time.time() < end_at:
            time.sleep(0.2)

    joblog.info(f"⏹  LOOP stop {name}")

# =========================
# Main
# =========================
def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def job_loop(job: Dict[str, Any]):
    allow_overlap = bool(job.get("allow_overlap", False))
    if allow_overlap:
        _loop_allow_overlap(job)
    else:
        _loop_no_overlap(job)

def main():
    cfg = load_config(CONFIG_FILE)
    jobs = cfg.get("jobs", [])
    jobs = [j for j in jobs if j.get("enabled", True)]

    if not jobs:
        ROOT_LOGGER.error(f"No enabled jobs in {CONFIG_FILE}")
        return

    ROOT_LOGGER.info(f"Loaded {len(jobs)} job(s). RUN_ONCE={RUN_ONCE}")

    threads: List[threading.Thread] = []
    for job in jobs:
        t = threading.Thread(target=job_loop, args=(job,), daemon=True)
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    for t in threads:
        t.join(timeout=5)

    ROOT_LOGGER.info("Exiting.")

if __name__ == "__main__":
    main()

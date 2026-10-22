"""Write and read the per-experiment metrics files that eval/report.py collects."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

import config


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def summarize(values):
    """mean, sd (sample, ddof=1 when n > 1), n for a list of per-seed values."""
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "sd": sd, "n": int(len(a)), "values": [float(v) for v in a]}


def write_metrics(name, metrics, cfg=None):
    """Save metrics to reports/metrics/<name>.json. `metrics` is any JSON-able dict."""
    config.METRICS.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "metrics": metrics,
        "config": cfg or {},
        "config_hash": config_hash(cfg or {}),
        "git_sha": _git_sha(),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = config.METRICS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def read_metrics(name):
    path = config.METRICS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None

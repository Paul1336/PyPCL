#!/usr/bin/env python3
"""
CLLab experiment logger — two-phase tracker for Cowork sync.

Usage
-----
Before a run starts:
    python scripts/log_experiment.py --mode start --run_id <id> [--desc "what this tests"]

After a run finishes (or stops):
    python scripts/log_experiment.py --mode finish --run_id <id> [--status resumable]

Let the hook auto-detect the latest completed run:
    python scripts/log_experiment.py --mode auto

Reads from:
    results/<run_id>/run_config.json   — hyperparams
    results/<run_id>/results.csv       — per-cell accuracy
    results/<run_id>/*_progress.json  — grid completion counts

Writes to:
    logs/pending.json  — accumulated records; paste to Cowork or let it auto-sync
"""

import argparse, csv, glob, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path


PENDING  = Path("logs/pending.json")
RUNS_DIR = Path("results")


# ─── git ──────────────────────────────────────────────────────────────────────

def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# ─── readers ──────────────────────────────────────────────────────────────────

def read_config(run_dir: Path) -> dict:
    p = run_dir / "run_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def read_progress(run_dir: Path) -> dict:
    """
    Count completed cells from *_progress.json files.
    Handles two common shapes:
      - {"completed": N, "total": M}            (dict with counts)
      - [{"status": "done", ...}, ...]           (list of cell records)
    Returns {"grid_done": N, "grid_total": M}.
    """
    files = list(run_dir.glob("*_progress.json"))
    if not files:
        return {}

    total_done = 0
    total_all  = 0

    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue

        if isinstance(data, dict):
            done  = data.get("completed", data.get("done", data.get("finished", 0)))
            total = data.get("total",     data.get("all",  0))
            if isinstance(done,  (int, float)): total_done += int(done)
            if isinstance(total, (int, float)): total_all  += int(total)

        elif isinstance(data, list):
            total_all += len(data)
            for item in data:
                if isinstance(item, dict):
                    s = item.get("status", "")
                    if s in ("done", "completed", "finished") or item.get("done"):
                        total_done += 1

    return {"grid_done": total_done, "grid_total": total_all}


def read_results(run_dir: Path) -> dict:
    """
    Parse results.csv → best / mean / worst accuracy + winning configs.
    Auto-detects the accuracy column and a config identifier column.
    Assumes accuracy is in [0,1] range if max value ≤ 1, else percentage.
    """
    p = run_dir / "results.csv"
    if not p.exists():
        return {}

    rows = []
    try:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    if not rows:
        return {}

    headers = list(rows[0].keys())

    # Pick accuracy column — priority list covers common naming conventions
    _ACC_EXACT = ("final_accuracy", "test_acc", "test_accuracy", "accuracy", "acc",
                  "final_acc", "val_acc", "val_accuracy", "top1", "top1_acc")
    acc_col = next(
        (c for c in headers if c.lower() in _ACC_EXACT),
        next((c for c in headers if "acc" in c.lower()), None),
    )
    if acc_col is None:
        return {}

    # Pick config identifier column — priority list covers common naming conventions
    _ID_EXACT = ("algorithm", "config", "method", "model", "name", "run",
                 "experiment", "setting", "condition")
    id_col = next(
        (c for c in headers if c.lower() in _ID_EXACT),
        None,
    )

    entries = []
    for row in rows:
        try:
            v = float(row[acc_col])
            label = row[id_col].strip() if id_col and row.get(id_col) else ""
            entries.append((v, label))
        except (ValueError, KeyError, TypeError):
            pass

    if not entries:
        return {}

    # Normalise to percentage
    vals = [v for v, _ in entries]
    if max(vals) <= 1.0:
        entries = [(v * 100, lbl) for v, lbl in entries]

    best_v,  best_cfg  = max(entries, key=lambda x: x[0])
    worst_v, worst_cfg = min(entries, key=lambda x: x[0])
    mean_v = sum(v for v, _ in entries) / len(entries)

    return {
        "best_acc":    round(best_v,  2),
        "mean_acc":    round(mean_v,  2),
        "worst_acc":   round(worst_v, 2),
        "best_config":  best_cfg,
        "worst_config": worst_cfg,
        "cells_in_csv": len(entries),
    }


# ─── config field extraction ───────────────────────────────────────────────────

_FIELD_KEYS = {
    "epochs":      ["epochs", "num_epochs", "max_epochs"],
    "batch_size":  ["batch_size", "bs", "batch"],
    "lr":          ["lr", "learning_rate", "base_lr"],
    "wd":          ["wd", "weight_decay"],
    "num_classes": ["num_classes", "n_classes", "C", "classes", "K"],
    "seeds":       ["seeds", "seed"],
    "dataset":     ["dataset", "data", "data_name", "dataset_name"],
    "methods":     ["method", "methods", "model", "models", "algorithm"],
    "description": ["description", "desc", "note", "experiment_name"],
}

def extract_hparams(cfg: dict) -> dict:
    out = {}
    for field, keys in _FIELD_KEYS.items():
        for k in keys:
            if k in cfg:
                out[field] = cfg[k]
                break
    # normalise to list
    if "methods" in out and isinstance(out["methods"], str):
        out["methods"] = [out["methods"]]
    if "seeds"   in out and isinstance(out["seeds"],   int):
        out["seeds"]   = [out["seeds"]]
    return out


# ─── record builders ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def build_start_record(run_id: str, runs_dir: Path, extra_desc: str | None = None) -> dict:
    run_dir = runs_dir / run_id
    cfg     = read_config(run_dir)
    hparams = extract_hparams(cfg)

    record = {
        "run_id":     run_id,
        "status":     "running",
        "date_start": _today(),
        "git_commit": git_commit(),
        "grid_done":  0,
        "grid_total": 0,
        "logged_at":  _now_iso(),
        **hparams,
    }
    if extra_desc:
        record["description"] = extra_desc
    return record


def build_finish_record(run_id: str, runs_dir: Path, force_status: str | None = None) -> dict:
    run_dir  = runs_dir / run_id
    cfg      = read_config(run_dir)
    hparams  = extract_hparams(cfg)
    metrics  = read_results(run_dir)
    progress = read_progress(run_dir)

    grid_done  = progress.get("grid_done",  metrics.get("cells_in_csv", 0))
    grid_total = progress.get("grid_total", 0)

    if force_status:
        status = force_status
    elif grid_total > 0 and grid_done >= grid_total:
        status = "completed"
    elif grid_done > 0:
        status = "stopped"
    else:
        status = "stopped"

    record = {
        "run_id":       run_id,
        "status":       status,
        "date_start":   _today(),
        "git_commit":   git_commit(),
        "grid_done":    grid_done,
        "grid_total":   grid_total,
        "best_acc":     metrics.get("best_acc"),
        "mean_acc":     metrics.get("mean_acc"),
        "worst_acc":    metrics.get("worst_acc"),
        "best_config":  metrics.get("best_config", ""),
        "worst_config": metrics.get("worst_config", ""),
        "logged_at":    _now_iso(),
        **hparams,
    }
    return record


# ─── pending.json ─────────────────────────────────────────────────────────────

def load_pending() -> list:
    if PENDING.exists():
        try:
            return json.loads(PENDING.read_text())
        except Exception:
            pass
    return []


def save_pending(entries: list):
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(json.dumps(entries, indent=2))


def upsert_pending(action: str, run_id: str, data: dict):
    entries = load_pending()
    for i, e in enumerate(entries):
        if e.get("data", {}).get("run_id") == run_id and e.get("action") == action:
            entries[i] = {"action": action, "data": data}
            save_pending(entries)
            return
    entries.append({"action": action, "data": data})
    save_pending(entries)


def logged_run_ids() -> set:
    return {e.get("data", {}).get("run_id") for e in load_pending()}


# ─── auto-detect ──────────────────────────────────────────────────────────────

def detect_latest_run(runs_dir: Path) -> str | None:
    """Most recently modified run directory that has results.csv but isn't logged."""
    already = logged_run_ids()
    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name in already:
            continue
        csv_p = d / "results.csv"
        if csv_p.exists():
            candidates.append((csv_p.stat().st_mtime, d.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ─── output ───────────────────────────────────────────────────────────────────

def print_log_block(action: str, data: dict):
    payload = json.dumps({"action": action, "data": data}, indent=2)
    print()
    print("╔═══ EXPERIMENT LOG ════════════════════════════════════════╗")
    print(payload)
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print("→ Paste the block above into your Cowork tracker session.")
    print("  (Or it's already saved to logs/pending.json for auto-sync.)")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="CLLab experiment logger")
    ap.add_argument("--mode",     required=True, choices=["start", "finish", "auto"])
    ap.add_argument("--run_id",   default=None,  help="Run ID (folder name under results/)")
    ap.add_argument("--runs_dir", default="results")
    ap.add_argument("--status",   default=None,
                    choices=["running", "completed", "stopped", "resumable"],
                    help="Override inferred status (finish mode)")
    ap.add_argument("--desc",     default=None,  help="Short description of what the run tests")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)

    # ── start ──
    if args.mode == "start":
        if not args.run_id:
            ap.error("--run_id is required for --mode start")
        record = build_start_record(args.run_id, runs_dir, extra_desc=args.desc)
        upsert_pending("set", args.run_id, record)
        print(f"[start] Logged '{args.run_id}' as running.")
        print_log_block("set", record)

    # ── finish ──
    elif args.mode == "finish":
        run_id = args.run_id
        if not run_id:
            run_id = detect_latest_run(runs_dir)
            if not run_id:
                print("No new completed runs detected.")
                sys.exit(0)
            print(f"[finish] Auto-detected run: {run_id}")
        record = build_finish_record(run_id, runs_dir, force_status=args.status)
        upsert_pending("update", run_id, record)
        print(f"[finish] Logged '{run_id}' as {record['status']}.")
        print_log_block("update", record)

    # ── auto ──
    elif args.mode == "auto":
        run_id = detect_latest_run(runs_dir)
        if not run_id:
            # Silent exit — hook fires on every Stop, most of the time there's nothing new
            sys.exit(0)
        record = build_finish_record(run_id, runs_dir, force_status=args.status)
        upsert_pending("update", run_id, record)
        print(f"[auto] Logged '{run_id}' as {record['status']} → logs/pending.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse.io_utils import save_json  # noqa: E402
from continual_learning.common import PHASES, PROBES, domain_of, load_m3cot_rows, rows_to_tasks, save_tasks  # noqa: E402


def take_domain(rows, domain: str, n: int, *, exclude_ids: set[str]) -> list:
    out = []
    for row in rows:
        rid = str(row.get("id"))
        if rid in exclude_ids:
            continue
        if str(row.get("domain")).strip().lower() != domain:
            continue
        out.append(row)
        exclude_ids.add(rid)
        if len(out) >= n:
            break
    if len(out) < n:
        raise SystemExit(f"Not enough rows for domain={domain}: need {n}, got {len(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed M3CoT continual-learning smoke splits.")
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--probe-count", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split", default="validation")
    parser.add_argument("--probe-split", default="test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows_all = load_m3cot_rows(args.train_split)
    probe_rows_all = load_m3cot_rows(args.probe_split)

    used_train: set[str] = set()
    used_probe: set[str] = set()
    manifest = {
        "train_split": args.train_split,
        "probe_split": args.probe_split,
        "train_count_per_domain": args.train_count,
        "probe_count_per_domain": args.probe_count,
        "phases": {},
        "probes": {},
        "available_train_domains": dict(Counter(str(row.get("domain")).lower() for row in train_rows_all)),
        "available_probe_domains": dict(Counter(str(row.get("domain")).lower() for row in probe_rows_all)),
    }

    for phase_key, domain, name in PHASES:
        rows = take_domain(train_rows_all, domain, args.train_count, exclude_ids=used_train)
        tasks = rows_to_tasks(rows)
        save_tasks(out_dir / f"{name}.jsonl", tasks)
        manifest["phases"][name] = {
            "phase_key": phase_key,
            "domain": domain,
            "count": len(tasks),
            "sample_ids": [task.sample_id for task in tasks],
        }

    for probe_key, domain, name in PROBES:
        rows = take_domain(probe_rows_all, domain, args.probe_count, exclude_ids=used_probe)
        tasks = rows_to_tasks(rows)
        save_tasks(out_dir / f"{name}.jsonl", tasks)
        manifest["probes"][name] = {
            "probe_key": probe_key,
            "domain": domain,
            "count": len(tasks),
            "sample_ids": [task.sample_id for task in tasks],
        }

    save_json(out_dir / "manifest.json", manifest)
    print(f"Saved continual-learning splits to: {out_dir}")


if __name__ == "__main__":
    main()

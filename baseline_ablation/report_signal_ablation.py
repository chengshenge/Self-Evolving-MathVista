#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_ablation.common import read_jsonl, slice_name, summarize_rows, write_csv  # noqa: E402


def load_eval_dir(path: Path):
    if (path / "eval_with_library.jsonl").exists():
        rows = read_jsonl(path / "eval_with_library.jsonl")
        tasks = read_first_existing(path, ["eval_tasks.jsonl", "eval_tasks_original.jsonl", "eval_tasks_used.jsonl"])
        summary_path = path / "summary.json"
    else:
        rows = read_jsonl(path / "base_evolution.jsonl")
        tasks = read_first_existing(path, ["eval_tasks.jsonl", "eval_tasks_original.jsonl", "eval_tasks_used.jsonl"])
        summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else summarize_rows(rows)
    return rows, tasks, summary


def read_first_existing(root: Path, names):
    for name in names:
        path = root / name
        if path.exists():
            return read_jsonl(path)
    return []


def library_size(eval_dir: Path, summary: dict) -> int:
    configured = (summary.get("config") or {}).get("library_dir")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend([eval_dir / "library", eval_dir / "generated_skills", eval_dir / "generated_after_seed"])
    for root in candidates:
        if root.exists():
            return sum(1 for child in root.iterdir() if child.is_dir() and not child.name.startswith("_"))
    return 0


def summarize_group(label: str, eval_dir: Path):
    rows, tasks, summary = load_eval_dir(eval_dir)
    task_by_pid = {str(t.get("sample_id") or t.get("pid")): t for t in tasks}
    out = {
        "group": label,
        "overall_accuracy": summary.get("accuracy") if "accuracy" in summary else (summary.get("base_evolution") or {}).get("accuracy"),
        "num_samples": summary.get("num_samples") if "num_samples" in summary else (summary.get("base_evolution") or {}).get("num_samples"),
        "num_saved_skills": library_size(eval_dir, summary),
        "num_reused_generated_skills": summary.get("num_reused_generated_skills") if "num_reused_generated_skills" in summary else (summary.get("base_evolution") or {}).get("num_reused_generated_skills"),
        "reuse_subset_accuracy": None,
    }
    reused = [r for r in rows if r.get("used_generated_skill")]
    if reused:
        out["reuse_subset_accuracy"] = summarize_rows(reused)["accuracy"]
    for sl in ["identity_age_gap", "geometry", "bar_chart", "synthetic_counting"]:
        bucket = [r for r in rows if slice_name(task_by_pid.get(str(r.get("pid")), {})) == sl]
        out[f"{sl}_accuracy"] = summarize_rows(bucket)["accuracy"] if bucket else None
        out[f"{sl}_n"] = len(bucket)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare G0/G1/G2/G3 downstream eval outputs.")
    parser.add_argument("--g0-dir", required=True, help="Existing G0 result directory; read-only reference.")
    parser.add_argument("--g1-dir", required=True)
    parser.add_argument("--g2-dir", required=True)
    parser.add_argument("--g3-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    groups = [
        ("G0_existing_reference", Path(args.g0_dir)),
        ("G1_oracle_outcome_verifier", Path(args.g1_dir)),
        ("G2_reward_only", Path(args.g2_dir)),
        ("G3_noisy_reward", Path(args.g3_dir)),
    ]
    rows = [summarize_group(label, path) for label, path in groups]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "overall_accuracy", "num_samples", "num_saved_skills", "num_reused_generated_skills",
        "reuse_subset_accuracy", "identity_age_gap_accuracy", "identity_age_gap_n", "geometry_accuracy",
        "geometry_n", "bar_chart_accuracy", "bar_chart_n", "synthetic_counting_accuracy", "synthetic_counting_n",
    ]
    write_csv(out_dir / "signal_ablation.csv", rows, fields)
    lines = ["| group | overall | saved | reused | reuse acc | identity | geometry | bar | synthetic |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['group']} | {fmt(row['overall_accuracy'])} | {row['num_saved_skills']} | {row['num_reused_generated_skills']} | "
            f"{fmt(row['reuse_subset_accuracy'])} | {fmt(row['identity_age_gap_accuracy'])} | {fmt(row['geometry_accuracy'])} | "
            f"{fmt(row['bar_chart_accuracy'])} | {fmt(row['synthetic_counting_accuracy'])} |"
        )
    (out_dir / "signal_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved ablation report to: {out_dir}")


def fmt(value):
    return "" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()

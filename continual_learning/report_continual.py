#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_learning.common import PROBES, write_csv  # noqa: E402

CHECKPOINTS = ["ckpt0", "ckpt1", "ckpt2", "ckpt3"]
SYSTEMS = ["S0", "S1", "S2"]


def load_summary(stream_dir: Path, ckpt: str) -> dict:
    path = stream_dir / "eval" / ckpt / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def acc(summary: dict, probe_name: str):
    return (summary.get("probes") or {}).get(probe_name, {}).get("accuracy")


def diff(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def fmt(value):
    return "" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, success_rows, fwt_rows, forgetting_rows):
    lines = ["# Continual Learning Report", "", "## Success Curve", ""]
    lines.append("| system | checkpoint | Probe_CS | Probe_SCI | Probe_MATH |")
    lines.append("|---|---|---:|---:|---:|")
    for row in success_rows:
        lines.append(f"| {row['system']} | {row['checkpoint']} | {fmt(row['Probe_CS'])} | {fmt(row['Probe_SCI'])} | {fmt(row['Probe_MATH'])} |")
    lines.extend(["", "## Forward Transfer", "", "| system | metric | value |", "|---|---|---:|"])
    for row in fwt_rows:
        lines.append(f"| {row['system']} | {row['metric']} | {fmt(row['value'])} |")
    lines.extend(["", "## Forgetting", "", "| system | metric | value |", "|---|---|---:|"])
    for row in forgetting_rows:
        lines.append(f"| {row['system']} | {row['metric']} | {fmt(row['value'])} |")
    lines.extend(["", "## Skill Pollution", "", "See `skill_pollution.csv` for checkpoint/probe-level reuse stats."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report continual-learning success/FWT/forgetting/pollution metrics.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.output_dir) if args.output_dir else run_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    success_rows = []
    pollution_rows = []
    stream_dirs = {system: run_root / system for system in SYSTEMS if (run_root / system).exists()}
    for system, stream_dir in stream_dirs.items():
        for ckpt in CHECKPOINTS:
            summary = load_summary(stream_dir, ckpt)
            row = {"system": system, "checkpoint": ckpt}
            for _probe_key, _domain, probe_name in PROBES:
                row[probe_name] = acc(summary, probe_name)
                probe_summary = (summary.get("probes") or {}).get(probe_name, {})
                pollution_rows.append({
                    "system": system,
                    "checkpoint": ckpt,
                    "probe": probe_name,
                    "domain": probe_summary.get("domain"),
                    "accuracy": probe_summary.get("accuracy"),
                    "num_saved_skills": summary.get("num_saved_skills"),
                    "num_reused_generated_skills": probe_summary.get("num_reused_generated_skills"),
                    "num_unique_reused_skills": probe_summary.get("num_unique_reused_skills"),
                    "num_reuse_candidates_total": probe_summary.get("num_reuse_candidates_total"),
                    "num_reuse_attempts_total": probe_summary.get("num_reuse_attempts_total"),
                    "num_reuse_accepts_total": probe_summary.get("num_reuse_accepts_total"),
                    "num_used_generated_skills_total": probe_summary.get("num_used_generated_skills_total"),
                    "reuse_accuracy": probe_summary.get("reuse_accuracy"),
                    "old_domain_reuse_precision": probe_summary.get("old_domain_reuse_precision"),
                    "reuse_entropy": probe_summary.get("reuse_entropy"),
                    "top1_skill_share": probe_summary.get("top1_skill_share"),
                    "top3_skill_share": probe_summary.get("top3_skill_share"),
                    "cross_phase_reuse_count": probe_summary.get("cross_phase_reuse_count"),
                    "cross_phase_toxic_reuse_count": probe_summary.get("cross_phase_toxic_reuse_count"),
                    "cross_phase_toxic_reuse_ratio": probe_summary.get("cross_phase_toxic_reuse_ratio"),
                    "cross_phase_toxic_reuse_accuracy": probe_summary.get("cross_phase_toxic_reuse_accuracy"),
                })
            success_rows.append(row)

    fwt_rows = []
    forgetting_rows = []
    by_system_ckpt = {(row["system"], row["checkpoint"]): row for row in success_rows}
    for system in sorted(stream_dirs):
        c0 = by_system_ckpt[(system, "ckpt0")]
        c1 = by_system_ckpt[(system, "ckpt1")]
        c2 = by_system_ckpt[(system, "ckpt2")]
        c3 = by_system_ckpt[(system, "ckpt3")]
        fwt_rows.extend([
            {"system": system, "metric": "FWT_CS_to_SCI", "value": diff(c1["Probe_SCI"], c0["Probe_SCI"])},
            {"system": system, "metric": "FWT_CS_to_MATH", "value": diff(c1["Probe_MATH"], c0["Probe_MATH"])},
            {"system": system, "metric": "FWT_CS_SCI_to_MATH", "value": diff(c2["Probe_MATH"], c0["Probe_MATH"])},
        ])
        forgetting_rows.extend([
            {"system": system, "metric": "Forget_CS_after_SCI", "value": diff(c1["Probe_CS"], c2["Probe_CS"])},
            {"system": system, "metric": "Forget_CS_after_MATH", "value": diff(c1["Probe_CS"], c3["Probe_CS"])},
            {"system": system, "metric": "Forget_SCI_after_MATH", "value": diff(c2["Probe_SCI"], c3["Probe_SCI"])},
        ])

    write_csv(out_dir / "success_curve.csv", success_rows, ["system", "checkpoint", "Probe_CS", "Probe_SCI", "Probe_MATH"])
    write_csv(out_dir / "forward_transfer.csv", fwt_rows, ["system", "metric", "value"])
    write_csv(out_dir / "forgetting.csv", forgetting_rows, ["system", "metric", "value"])
    write_csv(
        out_dir / "skill_pollution.csv",
        pollution_rows,
        [
            "system", "checkpoint", "probe", "domain", "accuracy", "num_saved_skills",
            "num_reused_generated_skills", "num_unique_reused_skills", "num_reuse_candidates_total",
            "num_reuse_attempts_total", "num_reuse_accepts_total", "num_used_generated_skills_total",
            "reuse_accuracy", "old_domain_reuse_precision", "reuse_entropy", "top1_skill_share",
            "top3_skill_share", "cross_phase_reuse_count", "cross_phase_toxic_reuse_count",
            "cross_phase_toxic_reuse_ratio", "cross_phase_toxic_reuse_accuracy",
        ],
    )
    write_markdown(out_dir / "summary.md", success_rows, fwt_rows, forgetting_rows)
    print(f"Saved continual report to: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_ablation.common import read_jsonl, spearman, write_csv  # noqa: E402


def metrics(rows):
    total_pos = sum(1 for r in rows if r["oracle_correct"])
    total_neg = sum(1 for r in rows if not r["oracle_correct"])
    kept = [r for r in rows if r["keep"]]
    tp = sum(1 for r in kept if r["oracle_correct"])
    fp = sum(1 for r in kept if not r["oracle_correct"])
    fn = sum(1 for r in rows if (not r["keep"]) and r["oracle_correct"])
    tn = sum(1 for r in rows if (not r["keep"]) and (not r["oracle_correct"]))
    corr = spearman([float(r.get("score", 0.0) or 0.0) for r in rows], [1.0 if r["oracle_correct"] else 0.0 for r in rows])
    return {
        "signal_mode": rows[0]["signal_mode"] if rows else "unknown",
        "num_candidates": len(rows),
        "num_kept": len(kept),
        "precision_at_keep": tp / (tp + fp) if (tp + fp) else None,
        "recall_at_keep": tp / total_pos if total_pos else None,
        "false_positive_rate": fp / total_neg if total_neg else None,
        "false_negative_rate": fn / total_pos if total_pos else None,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "spearman_with_oracle": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report candidate-level signal reliability against oracle correctness.")
    parser.add_argument("candidate_pool")
    parser.add_argument("signal_scores", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in args.signal_scores:
        rows = read_jsonl(path)
        summaries.append(metrics(rows))

    fields = [
        "signal_mode", "num_candidates", "num_kept", "precision_at_keep", "recall_at_keep",
        "false_positive_rate", "false_negative_rate", "true_positive", "false_positive",
        "true_negative", "false_negative", "spearman_with_oracle",
    ]
    write_csv(out_dir / "signal_reliability.csv", summaries, fields)
    lines = ["| signal | kept | precision | recall | FPR | FNR | spearman |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(
            f"| {row['signal_mode']} | {row['num_kept']} | {fmt(row['precision_at_keep'])} | {fmt(row['recall_at_keep'])} | "
            f"{fmt(row['false_positive_rate'])} | {fmt(row['false_negative_rate'])} | {fmt(row['spearman_with_oracle'])} |"
        )
    (out_dir / "signal_reliability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved reliability report to: {out_dir}")


def fmt(value):
    return "" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()

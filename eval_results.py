from __future__ import annotations

import argparse
from pathlib import Path

from muse.io_utils import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved prediction rows.")
    parser.add_argument("--predictions", required=True, help="Path to JSONL produced by run_mathvista.py")
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    total = sum(1 for row in rows if row.get("correct") is not None)
    correct = sum(1 for row in rows if row.get("correct") is True)
    print(f"Rows: {len(rows)}")
    if total:
        print(f"Accuracy: {correct}/{total} = {correct / total:.3f}")
    else:
        print("No gold answers found in predictions.")


if __name__ == "__main__":
    main()

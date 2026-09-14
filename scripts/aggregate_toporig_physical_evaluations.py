#!/usr/bin/env python3
"""Combine completed TopoRig physical-evaluation summaries into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluation import atomic_write_csv, atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    rows: list[dict[str, object]] = []
    checkpoints: dict[str, str] = {}
    for run_name in args.runs:
        path = result_root / run_name / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete evaluation report: {path}")
        checkpoints[run_name] = str(payload["checkpoint"])
        model_rows = payload.get("models")
        if not isinstance(model_rows, list) or len(model_rows) != 3:
            raise ValueError(f"Expected three source summaries in {path}.")
        for row in model_rows:
            rows.append({"evaluation": run_name, **row})

    fields = (
        "evaluation",
        "model",
        "source",
        "head_count",
        "action_unit_count",
        "sample_count",
        "mae_mm",
        "mae_q95_mm",
        "element_weighted_mae_mm",
        "moving_vertex_count",
        "total_vertex_count",
        "moving_vertex_ratio",
        "standard_head_height_mm",
        "standard_head_mae_mm",
        "standard_head_mae_q95_mm",
    )
    atomic_write_csv(result_root / "all_checkpoint_summary.csv", fields, rows)
    atomic_write_json(
        result_root / "all_checkpoint_summary.json",
        {
            "status": "complete",
            "evaluations": list(args.runs),
            "checkpoints": checkpoints,
            "rows": rows,
        },
    )
    print(
        "[RESULT] evaluation source moving_MAE_mm moving_MAE_Q95_mm "
        "standard_head_MAE_mm standard_head_MAE_Q95_mm",
        flush=True,
    )
    for row in rows:
        print(
            f"[RESULT] {row['evaluation']:24s} {row['source']:14s} "
            f"{float(row['mae_mm']):.6f} {float(row['mae_q95_mm']):.6f} "
            f"{float(row['standard_head_mae_mm']):.6f} "
            f"{float(row['standard_head_mae_q95_mm']):.6f}",
            flush=True,
        )
    print(f"[DONE] Combined reports: {result_root}", flush=True)


if __name__ == "__main__":
    main()

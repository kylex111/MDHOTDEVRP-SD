"""Summarize Pareto objectives from run.json files saved by the common entry point."""

from __future__ import annotations
import json
from pathlib import Path


def summarize(results):
    path = Path(results)
    paths = [path] if path.is_file() else sorted(path.rglob("run.json"))
    rows = []
    for source in paths:
        record = json.loads(source.read_text(encoding="utf-8-sig"))
        if record.get("status") != "completed":
            continue
        points = record.get("pareto_points", [])
        rows.append(
            {
                "path": str(source.resolve()),
                "algorithm": record.get("algorithm"),
                "instance": record.get("instance_path"),
                "run_id": record.get("run_id"),
                "archive_size": len(record.get("archive", [])),
                "minimum_cost": min((p[0] for p in points), default=None),
                "minimum_ra": min((p[1] for p in points), default=None),
                "runtime_sec": record.get("runtime_sec"),
            }
        )
    return {"runs": rows, "count": len(rows)}

"""JSON serialization, content hashes, and environment provenance."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [json_value(v) for v in sorted(value)]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported artifact type: {type(value).__name__}")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(
        json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_record(project):
    project = Path(project).resolve()

    def git(*args):
        try:
            return subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={project.as_posix()}",
                    "-C",
                    str(project),
                    "--no-optional-locks",
                    *args,
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "pymoo", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_hashes = {
        str(p.relative_to(project)): sha256_file(p)
        for p in sorted(project.rglob("*.py"))
        if not set(p.parts) & {".git", "venv", ".venv", "__pycache__"}
    }
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": versions,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain"),
        "source_sha256": source_hashes,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }

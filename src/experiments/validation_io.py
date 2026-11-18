"""Atomic writes, source fingerprints, and read-only record checks for scenario validation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

from .artifacts import json_value, sha256_file


def read_json(path):
    """On Windows, allow another process to atomically replace the file while reading."""
    path = Path(path).resolve()
    if os.name != "nt":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    import ctypes
    from ctypes import wintypes
    import msvcrt
    create = ctypes.windll.kernel32.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError()
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, "r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.writing")
    temporary.write_text(value, encoding="utf-8")
    for attempt in range(31):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 30:
                raise
            time.sleep(min(0.01 * (attempt + 1), 0.1))


def write_json(path, value):
    write_text(path, json.dumps(json_value(value), ensure_ascii=False,
                               allow_nan=False, indent=2))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        write_text(path, "\ufeff")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(json_value(v), ensure_ascii=False)
                         if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()})
    write_text(path, "\ufeff" + stream.getvalue())


def install_recording_adapter():
    """Retry the same save payload only, never search or evaluation. Install before importing run."""
    from . import artifacts
    artifacts.save_json = write_json


def content_hash(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_inventory(project):
    project = Path(project)
    return {p.relative_to(project).as_posix(): sha256_file(p)
            for p in [project / "run.py", *sorted((project / "src").rglob("*.py"))]}


def git_state(root):
    root = Path(root)
    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
            "-C", str(root), "--no-optional-locks", *args], text=True,
            encoding="utf-8", stderr=subprocess.DEVNULL).strip()
    return {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain"), "index": git("ls-files", "--stage")}


def preserved_project_state(root):
    root = Path(root)
    inventory = {}
    excluded = {".git", "results", "beta_saa_results", "__pycache__", ".venv", "venv"}
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = [name for name in subdirs if name not in excluded]
        for name in names:
            path = Path(directory) / name
            if path.suffix.lower() in {".py", ".json", ".md", ".toml", ".ini", ".cfg", ".txt"}:
                inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    return {**git_state(root), "files": inventory}


def task_directory(batch, task, stage):
    return Path(batch) / stage / f"S{task['scenario_count']:03d}" / task["directory"]


def verify_hashes(files):
    for name, expected in files.items():
        if sha256_file(name) != expected:
            raise ValueError(f"Frozen file changed: {name}")

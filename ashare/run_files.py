"""Publish a completed run directory after all files are written."""

import os
import shutil
import tempfile
from pathlib import Path


def publish_directory(path, writer):
    if path.exists():
        raise ValueError(f"输出目录已存在：{path}。请使用新目录保留历史。")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".run-", dir=path.parent))
    try:
        writer(staging)
        os.rename(staging, path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

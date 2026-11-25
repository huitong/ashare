"""Atomic records shared by the trainer and read-only training dashboard."""

import json
import os
import tempfile
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        staged = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)

# src/pdfscribe/hashutil.py
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Final

_BUF: Final[int] = 1024 * 1024


def sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file.
    It reads the file in chunks to avoid high memory usage.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_BUF)
            if not chunk:
                # EOF
                break
            h.update(chunk)
    return h.hexdigest()

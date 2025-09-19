# src/pdfscribe/cache.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
from .config import DoclingConfig

_MANIFEST = "manifest.json"


@dataclass
class Manifest:
    created_at: float
    input_pdf: str
    input_size: int
    input_mtime: float
    input_sha256: str
    docling_config: Dict[str, Any]
    tool_version: str


def manifest_path(out_dir: Path) -> Path:
    """
    Get the path to the manifest file in the output directory.
    """
    return out_dir / _MANIFEST


def load_manifest(out_dir: Path) -> Dict[str, Any] | None:
    """
    Load the manifest file from the output directory.
    """
    p = manifest_path(out_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_manifest(
    out_dir: Path,
    *,
    input_pdf: Path,
    sha256: str,
    cfg: DoclingConfig,
    tool_version: str,
) -> None:
    """
    Write a manifest file to the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    m = Manifest(
        created_at=time.time(),
        input_pdf=str(input_pdf),
        input_size=input_pdf.stat().st_size,
        input_mtime=input_pdf.stat().st_mtime,
        input_sha256=sha256,
        docling_config=cfg.to_dict(),
        tool_version=tool_version,
    )
    manifest_path(out_dir).write_text(
        json.dumps(asdict(m), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cache_is_valid(
    out_dir: Path,
    *,
    input_pdf: Path,
    sha256: str,
    cfg: DoclingConfig,
    tool_version: str,
) -> bool:
    """
    Check if the cache in the output directory is valid.
    Valid if:
      - SHA256 matches
      - input size + mtime match
      - config matches
      - tool version matches
    """
    m = load_manifest(out_dir)
    if not m:
        return False

    try:
        st = input_pdf.stat()
        return (
            m["input_sha256"] == sha256
            and m["input_size"] == st.st_size
            and abs(m["input_mtime"] - st.st_mtime) < 1.0  # allow +/- 1 second drift
            and m["docling_config"] == cfg.to_dict()
            and m.get("tool_version") == tool_version
        )
    except (KeyError, FileNotFoundError):
        return False

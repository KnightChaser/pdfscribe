# src/pdfscribe/run_index.py
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RunRecord:
    """
    A record of a single run, for indexing and audit.
    """

    input_name: str  # basename e.g. "Elephant.pdf"
    input_path: str  # absolute path
    sha256: str  # full SHA-256
    run_dir_name: str  # sha[:16]
    run_dir_path: str  # absolute path to run dir
    index_md_path: str  # absolute path to index.md (may not exist yet)
    pdf_mtime_iso: str  # ISO8601
    created_at_iso: str  # ISO8601
    tool_version: str


# CSV field order
_CSV_FIELDS = [
    "input_name",
    "input_path",
    "sha256",
    "run_dir_name",
    "run_dir_path",
    "index_md_path",
    "pdf_mtime_iso",
    "created_at_iso",
    "tool_version",
]


def _csv_path(out_root: Path) -> Path:
    """
    Path to 'runs.csv' in the output root directory.
    """
    return out_root / "runs.csv"


def _jsonl_path(out_root: Path) -> Path:
    """
    Path to 'runs.jsonl' in the output root directory.
    """
    return out_root / "runs.jsonl"


def _key(rec: RunRecord) -> Tuple[str, str]:
    """
    Unique by (input_path, sha256)
    """
    return (rec.input_path, rec.sha256)


def _read_csv(path: Path) -> Dict[Tuple[str, str], RunRecord]:
    """
    Read existing CSV, return dict keyed by (input_path, sha256).
    """
    out: Dict[Tuple[str, str], RunRecord] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            try:
                rec = RunRecord(
                    input_name=row["input_name"],
                    input_path=row["input_path"],
                    sha256=row["sha256"],
                    run_dir_name=row["run_dir_name"],
                    run_dir_path=row["run_dir_path"],
                    index_md_path=row["index_md_path"],
                    pdf_mtime_iso=row["pdf_mtime_iso"],
                    created_at_iso=row["created_at_iso"],
                    tool_version=row["tool_version"],
                )
                out[_key(rec)] = rec
            except KeyError:
                # NOTE: ignore malformed rows
                continue
    return out


def _write_csv_atomic(path: Path, records: List[RunRecord]) -> None:
    """
    Write CSV atomically by writing to a temp file and renaming.
    """
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow(asdict(rec))
    # Later, atomic replace
    tmp.replace(path)


def upsert_run_record(out_root: Path, rec: RunRecord) -> None:
    """
    Ensure 'runs.csv' and 'runs.jsonl' reflect this run.
    - CSV: upsert (dedup by input_path+sha256)
    - JSONL: append for audit (no dedup)
    """
    out_root.mkdir(parents=True, exist_ok=True)

    # CSV upsert
    csv_path = _csv_path(out_root)
    existing = _read_csv(csv_path)
    existing[_key(rec)] = rec
    _write_csv_atomic(csv_path, list(existing.values()))

    # JSONL append (audit trail)
    jsonl_path = _jsonl_path(out_root)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False))
        f.write("\n")


def make_record(
    *,
    input_pdf: Path,
    run_dir: Path,
    sha256: str,
    tool_version: str,
) -> RunRecord:
    """
    Create a RunRecord for this run.
    """
    input_pdf = input_pdf.resolve()
    run_dir = run_dir.resolve()
    index_md = run_dir / "index.md"
    mtime = datetime.fromtimestamp(
        input_pdf.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    now = datetime.now(tz=timezone.utc).isoformat()
    return RunRecord(
        input_name=input_pdf.name,
        input_path=str(input_pdf),
        sha256=sha256,
        run_dir_name=sha256[:16],
        run_dir_path=str(run_dir),
        index_md_path=str(index_md),
        pdf_mtime_iso=mtime,
        created_at_iso=now,
        tool_version=tool_version,
    )

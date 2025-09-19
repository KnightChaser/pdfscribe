# src/pdfscribe/pipeline/assets.py
from __future__ import annotations
from pathlib import Path
from typing import List
from PIL import Image
from ..hashutil import sha256_file
from ..models.schema import Element


def find_page_md(run_dir: Path, page: int) -> Path:
    """
    Find the Markdown file for a given page in the run directory.
    """
    p = run_dir / f"page-{page:04d}.md"
    if not p.exists():
        raise FileNotFoundError(f"Markdown for page {page} not found in {run_dir}")
    return p


def read_page_text(run_dir: Path, page: int, max_chars: int = 8000) -> str:
    """
    Read the text content of a page's Markdown file, up to max_chars.
    """
    md = find_page_md(run_dir, page).read_text(encoding="utf-8")
    return md[:max_chars]


def list_page_images(run_dir: Path, page: int) -> List[Element]:
    """
    List image elements for a given page in the run directory.
    """
    img_dir = run_dir / "images" / f"page_{page:04d}"
    if not img_dir.exists():
        return []
    elements: List[Element] = []

    for p in sorted(img_dir.glob("*.png")):
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            # Ignore image read errors
            continue
        elements.append(
            Element(
                id=f"img_p{page:03d}_{p.stem}",
                page=page,
                path=str(p.resolve()),
                sha256=sha256_file(p),
                width=w,
                height=h,
            )
        )

    return elements

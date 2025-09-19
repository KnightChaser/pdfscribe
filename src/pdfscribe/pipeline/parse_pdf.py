# src/pdfscribe/pipeline/parse_pdf.py
from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Iterable

from ..config import DoclingConfig


def _pin_threads() -> None:
    """
    Pin the number of threads used by various libraries to the number of CPU cores.
    """
    cpu_count = os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)

    # NOTE: Silence the pin_memory warning from torch DataLoader (CPU-only runs)
    warnings.filterwarnings(
        "ignore",
        message=".*'pin_memory' argument is set as true.*",
        category=UserWarning,
        module="torch.utils.data.dataloader",
    )


def run_docling(
    src_pdf: Path,
    out_dir: Path,
    cfg: DoclingConfig,
) -> list[Path]:
    """
    Parse PDF with docling and emit per-page Markdown + referenced images.

    Returns a list of generated Markdown file paths.
    """
    _pin_threads()

    # WARNING: Do a lazy import here so environment variables are set first
    # and then libraries are imported with the correct settings.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc.base import ImageRefMode

    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    pipe = PdfPipelineOptions()
    pipe.images_scale = cfg.images_scale
    pipe.generate_picture_images = cfg.generate_picture_images
    pipe.generate_page_images = cfg.generate_page_images

    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)}
    )

    res = conv.convert(src_pdf)

    pages: Iterable[int] = sorted(res.document.pages.keys())
    md_paths: list[Path] = []

    for page in pages:
        img_dir = images_root / f"page_{page:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"page_{page:04d}.md"
        res.document.save_as_markdown(
            md_path,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=img_dir,
            page_no=page,
        )
        md_paths.append(md_path)

    # NOTE: (optional) write a simple index
    index = out_dir / "index.md"
    index.write_text(
        "\n".join(f"- [Page {i}](page_{i:04d}.md)" for i in pages),
        encoding="utf-8",
    )

    return md_paths

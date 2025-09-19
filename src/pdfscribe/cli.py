# src/pdfscribe/cli.py
from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .config import DoclingConfig
from .hashutil import sha256_file
from .cache import cache_is_valid, write_manifest
from .pipeline.parse_pdf import run_docling

app = typer.Typer(add_completion=False, help="pdfscribe: PDF --> Markdown (docling)")

console = Console()


def _abort(msg: str, code: int = 1) -> None:
    """
    Print an error message and exit the program.
    """
    console.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=code)


@app.command("parse")
def parse(
    pdf: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Path to input PDF"
    ),
    outdir: Path = typer.Option(
        Path("output"), "--outdir", "-o", help="Output directory"
    ),
    images_scale: float = typer.Option(
        2.0, help="Image scale factor for extracted pictures"
    ),
    generate_picture_images: bool = typer.Option(
        True, help="Export cropped images (figures, etc.)"
    ),
    generate_page_images: bool = typer.Option(False, help="Export full-page images"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Ignore cache and re-parse"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Less verbose output"),
) -> None:
    """
    Parse PDF using docling and emit per-page Markdown with referenced images.
    Caching is keyed by SHA-256 of the input and the docling config.
    """
    if not pdf.suffix.lower() == ".pdf":
        _abort("Input file must be a PDF")

    cfg = DoclingConfig(
        images_scale=images_scale,
        generate_picture_images=generate_picture_images,
        generate_page_images=generate_page_images,
    )

    # NOTE: Each PDF gets its own subdir named by SHA to isolate runs
    sha = sha256_file(pdf)
    outdir = outdir.resolve()
    run_dir = outdir / sha[:16]  # still unique enough

    if run_dir.exists() and not force:
        if cache_is_valid(
            run_dir, input_pdf=pdf, sha256=sha, cfg=cfg, tool_version=__version__
        ):
            if not quiet:
                console.print(
                    Panel.fit(
                        f"[green]Cache hit[/green]\n"
                        f"Input: [bold]{pdf.name}[/bold]\n"
                        f"SHA: {sha}\n"
                        f"Output: {run_dir}",
                        title="pdfscribe",
                    )
                )
            raise typer.Exit(0)

    # No valid cache --> (re)run the pipeline so far as needed
    if not quiet:
        console.print(
            Panel.fit(
                f"[yellow]Parsing[/yellow]\n"
                f"Input: [bold]{pdf}[/bold]\n"
                f"SHA: {sha}\n"
                f"Output: {run_dir}\n"
                f"Config: scale={cfg.images_scale}, pic_imgs={cfg.generate_picture_images}, page_imgs={cfg.generate_page_images}",
                title=f"pdfscribe v{__version__}",
            )
        )

    md_files = run_docling(pdf, run_dir, cfg)
    write_manifest(
        run_dir, input_pdf=pdf, sha256=sha, cfg=cfg, tool_version=__version__
    )

    if not quiet:
        console.print(
            f"[bold green]Done[/bold green]. {len(md_files)} markdown files emitted."
        )
        console.print(f"Index: [underline]{(run_dir / 'index.md')}[/underline]")


@app.command("version")
def version() -> None:
    """
    Show version information.
    """
    console.print(f"pdfscribe version {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

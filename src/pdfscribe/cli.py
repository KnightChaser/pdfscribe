# src/pdfscribe/cli.py
from __future__ import annotations
from pathlib import Path
import typer
import os
import shutil
from rich.console import Console
from rich.panel import Panel
from openai import OpenAI

from . import __version__
from .config import DoclingConfig
from .hashutil import sha256_file
from .cache import cache_is_valid, write_manifest
from .rate_limit import TokenLimiter
from .pipeline.parse_pdf import run_docling
from .pipeline.describe_page import describe_page
from .pipeline.prefilter import HeuristicConfig

app = typer.Typer(add_completion=False, help="pdfscribe: PDF --> Markdown (docling)")

console = Console()


def _abort(msg: str, code: int = 1) -> None:
    """
    Print an error message and exit the program.
    """
    console.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=code)


def _prepare_render_dir(source: Path, target: Path) -> None:
    """
    Prepare a render directory by copying markdown files and linking images.
    """
    target.mkdir(parents=True, exist_ok=True)

    # Copy the page_*.md markdown files and index.md if present
    for md in source.glob("page_*.md"):
        shutil.copy2(md, target / md.name)
    idx = source / "index.md"
    if idx.exists():
        shutil.copy2(idx, target / "index.md")


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


@app.command("describe")
def describe(
    pdf: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to input PDF (used only to locate cache dir)",
    ),
    outdir: Path = typer.Option(
        Path("output"), "--outdir", "-o", help="Output directory root"
    ),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI vision-capable model"),
    prompt_version: str = typer.Option(
        "v1", help="Prompt/schema version for cache keying"
    ),
    page: int = typer.Option(
        -1, help="Describe only this page (1-based). Use -1 for all pages."
    ),
    max_images_per_page: int = typer.Option(
        6, help="Max candidate images to send to VLM per page"
    ),
    use_cache: bool = typer.Option(True, help="Reuse previous VLM JSON if unchanged"),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Delete VLM cache (run_dir/vlm_cache) before describing",
    ),
    entropy_thresh: float = typer.Option(2.0, help="Heuristic: min entropy to keep"),
    edge_density_thresh: float = typer.Option(
        0.004, help="Heuristic: min edge density to keep"
    ),
    min_w: int = typer.Option(32, help="Heuristic: min width"),
    min_h: int = typer.Option(32, help="Heuristic: min height"),
    max_aspect: float = typer.Option(8.0, help="Heuristic: max aspect ratio"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    write_mode: str = typer.Option(
        "in_place", help="Where to write captions: in_place | copy"
    ),
    render_subdir: str = typer.Option(
        "render", help="Subdirectory name used when write_mode=copy"
    ),
) -> None:
    """
    Classify and describe images per page using a VLM, and inject explanations into Markdown.
    Requires OPENAI_API_KEY.
    """
    # Locate the run_dir from sha of pdf (same convention as parse)
    from .hashutil import sha256_file

    sha = sha256_file(pdf)
    run_dir = (outdir.resolve()) / sha[:16]
    if not run_dir.exists():
        _abort(
            f"Parsed output not found: {run_dir}. Run 'pdfscribe parse {pdf} -o {outdir}' first."
        )

    # Optional: clear VLM cache on demand
    if clear_cache:
        cache_dir = run_dir / "vlm_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            if not quiet:
                console.print(f"[yellow]Cleared cache:[/yellow] {cache_dir}")
        else:
            if not quiet:
                console.print(f"[yellow]No cache to clear:[/yellow] {cache_dir}")

    target_dir = run_dir
    if write_mode == "copy":
        target_dir = run_dir / render_subdir
        _prepare_render_dir(source=run_dir, target=target_dir)

    client = OpenAI()  # needs OPENAI_API_KEY in env
    # Discover page count from existing files
    pages = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("page_*.md"))
    if page != -1:
        if page not in pages:
            _abort(f"Page {page} not found in {run_dir}")
        pages = [page]

    hcfg = HeuristicConfig(
        min_w=min_w,
        min_h=min_h,
        max_aspect=max_aspect,
        entropy_thresh=entropy_thresh,
        edge_density_thresh=edge_density_thresh,
    )

    provider_limit_tpm: int = 200_000
    limiter = TokenLimiter(capacity=int(provider_limit_tpm * 0.8))  # 80% of max
    total = 0

    for pg in pages:
        res = describe_page(
            run_dir=run_dir,
            page=pg,
            model=model,
            prompt_version=prompt_version,
            hcfg=hcfg,
            client=client,
            max_images=max_images_per_page,
            use_cache=use_cache,
            page_md_override=target_dir / f"page_{pg:04d}.md",  # write here
            limiter=limiter,
        )
        total += len(res)
        if not quiet:
            console.print(f"[cyan]Page {pg}[/cyan]: described {len(res)} images.")

    if not quiet:
        console.print(
            f"[bold green]Done[/bold green]. Injected captions/explanations. Total images: {total}"
        )


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

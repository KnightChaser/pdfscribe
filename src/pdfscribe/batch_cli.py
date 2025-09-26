# src/pdfscribe/batch_cli.py
from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

app = typer.Typer(
    add_completion=False, help="pdfscribe-batch: run parse+describe for many PDFs"
)
console = Console()


def _iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    """
    Recursively yield all PDF files in the given directory.
    """
    yield from sorted(pdf_dir.rglob("*.pdf"))


async def _stream_cmd(cmd: List[str], outbuf: Deque[str], outbuf_max: int) -> int:
    """
    Run a subprocess, streaming merged stdout/stderr line-by-lnie into outbuf.
    Returns exit code.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None

    # Read stdout lines as they arrive
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        s = line.decode("utf-8", errors="replace").rstrip("\n")
        outbuf.append(s)
        # keep last N lines
        while len(outbuf) > outbuf_max:
            outbuf.popleft()

    return await proc.wait()


async def _run_job(
    pdf: Path,
    outdir: Path,
    output_lines: Deque[str],
    outbuf_max: int,
) -> tuple[bool, Optional[str]]:
    """
    One job = parse + describe (sequential).
    Streams live output into output_lines.
    """
    # First, convert the given PDF into markdown files with image links.
    parse_cmd = ["pdfscribe", "parse", str(pdf), "-o", str(outdir)]

    # Second, for each image, add a proper annotation using multimodal AI model
    # (currently, OpenAI API-supplied model)
    desc_cmd = [
        "pdfscribe",
        "describe",
        str(pdf),
        "--outdir",
        str(outdir),
    ]

    output_lines.append(f"$ {' '.join(parse_cmd)}")
    rc1 = await _stream_cmd(parse_cmd, output_lines, outbuf_max)
    if rc1 != 0:
        return False, f"parse failed (exit {rc1})"

    output_lines.append(f"$ {' '.join(desc_cmd)}")
    rc2 = await _stream_cmd(desc_cmd, output_lines, outbuf_max)
    if rc2 != 0:
        return False, f"describe failed (exit {rc2})"

    return True, None


def _make_progress(total: int) -> Progress:
    """
    Create a Rich Progress instance with custom columns.
    """
    return Progress(
        TextColumn("[bold cyan]Progress[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
    )


@app.command("run")
def run(
    pdf_dir: Path = typer.Option(
        ...,
        "--pdf-dir",
        "-i",
        exists=True,
        file_okay=False,
        readable=True,
        help="Directory containing PDFs (searched recursively).",
    ),
    outdir: Path = typer.Option(
        Path("output"),
        "--outdir",
        "-o",
        help="Output directory (same semantics as pdfscribe).",
    ),
    tail_lines: int = typer.Option(
        50,
        "--tail-lines",
        help="How many recent output lines to keep visible in the OUTPUT window.",
    ),
    stop_on_error: bool = typer.Option(
        False,
        "--stop-on-error/--keep-going",
        help="Stop the batch at the first failed job (default: keep going).",
    ),
) -> None:
    """
    Sequentially run 'pdfscribe parse' then 'pdfscribe describe' for each PDF in pdf_dir,
    with a live OUTPUT window and a batch progress bar.
    """
    pdfs = list(_iter_pdfs(pdf_dir))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {pdf_dir}[/yellow]")
        raise typer.Exit(code=0)

    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Rolling buffer for the OUTPUT pane
    output_lines: Deque[str] = deque(maxlen=max(10, tail_lines))

    progress = _make_progress(total=len(pdfs))
    task_id = progress.add_task("batch", total=len(pdfs))

    # Group the OUTPUT panel + progress in a single Live layout
    def _render_group() -> Group:
        shown = "\n".join(output_lines) if output_lines else "(waiting for output...)"
        output_panel = Panel(
            shown,
            title="OUTPUT (latest)",
            subtitle="older outputs are omitted",
            border_style="white",
        )
        return Group(output_panel, progress)

    results_ok: int = 0
    results_fail: List[str] = []

    async def _runner() -> None:
        nonlocal results_ok

        for idx, pdf in enumerate(pdfs, start=1):
            output_lines.clear()
            output_lines.append(f"[{idx}/{len(pdfs)}] FILE: {pdf}")

            ok, err = await _run_job(pdf, outdir, output_lines, tail_lines)
            if ok:
                results_ok += 1
                output_lines.append(f"[OK] {pdf.name}")
            else:
                results_fail.append(f"{pdf} :: {err}")
                output_lines.append(f"[FAILED] {pdf.name} :: {err}")
                if stop_on_error:
                    progress.update(task_id, advance=1)
                    break

            progress.update(task_id, advance=1)

    with Live(_render_group(), console=console, refresh_per_second=15, transient=False):
        try:
            asyncio.run(_runner())
        except KeyboardInterrupt:
            output_lines.append("[INTERRUPTED] Ctrl+C")
        finally:
            pass

    # Final summary
    console.rule("[bold]SUMMARY")
    total = len(pdfs)
    console.print(f"[green]Succeeded:[/green] {results_ok}/{total}")
    if results_fail:
        console.print(f"[red]Failed:[/red] {len(results_fail)}")
        for line in results_fail:
            console.print(f"  - {line}")

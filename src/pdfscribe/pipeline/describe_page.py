# src/pdfscribe/pipeline/describe_page.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple
from openai import OpenAI
from ..models.schema import Element, VlmResult
from .assets import read_page_text, list_page_images
from .prefilter import HeuristicConfig, score_image, HeuristicScore


# ---------------------------
# Cache helpers (slim)
# ---------------------------
def _vlm_cache_path(run_dir: Path, image_id: str) -> Path:
    """
    Get the path to the cached VLM result for a given image ID.
    """
    return run_dir / "vlm_cache" / f"{image_id}.json"


def _load_cached(run_dir: Path, image_id: str) -> dict | None:
    """
    Load a cached VLM result for a given image ID, if it exists.
    """
    p = _vlm_cache_path(run_dir, image_id)
    if not p.exists():
        return None

    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cached(run_dir: Path, image_id: str, obj: dict) -> None:
    """
    Save a VLM result to the cache for a given image ID.
    """
    p = _vlm_cache_path(run_dir, image_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------
# Prefiltering
# ---------------------------
def prefilter_elements(
    elems: List[Element], cfg: HeuristicConfig
) -> Tuple[List[Element], List[Tuple[Element, HeuristicScore]]]:
    """
    Pre-filter elements using heuristic scoring.
    Returns a tuple of (kept_elements, skipped_elements_with_scores).
    """
    keep: List[Element] = []
    skipped: List[Tuple[Element, HeuristicScore]] = []
    for e in elems:
        score = score_image(e.path, cfg)
        if score.decision == "KEEP":
            keep.append(e)
        else:
            skipped.append((e, score))
    return keep, skipped


# ---------------------------
# Markdown rendering (idempotent anchors)
# ---------------------------
def _render_block(r: VlmResult) -> str:
    """
    Render a VLM result as a markdown block with an anchored wrapper for idempotency.
    """
    if r.classification == "MEANINGLESS":
        caption = "Decorative/meaningless image. Skipped."
        details = ""
    else:
        caption = r.text.strip()
        details = f"_Provenance_: id={r.image_id}, conf={r.confidence:.2f}, model={r.model}, sha256={r.image_sha256}"

    body = f"""*Caption (VLM):* {caption}

{details}""".strip()

    anchor = f"<!-- pdfscribe:{r.image_id} -->"
    return f"{anchor}\n{body}\n{anchor}"


def inject_explanations(page_md_path: Path, results: List[VlmResult]) -> None:
    """
    Inject VLM explanations into the markdown content of a page.
    """
    md = page_md_path.read_text(encoding="utf-8")
    for r in results:
        block = _render_block(r)
        anchor = f"<!-- pdfscribe:{r.image_id} -->"

        # Replace existing "anchored block" if present
        anchored = re.compile(re.escape(anchor) + r".*?" + re.escape(anchor), re.S)
        if anchored.search(md):
            md = anchored.sub(block, md, count=1)
            continue

        # Otherwise place under the first matching image, fallback append
        png_stem = r.image_id.split("img_p", 1)[-1].split("_", 1)[-1]
        pattern = re.compile(rf"(\!\[.*?\]\([^\)]*{re.escape(png_stem)}\.png\))")
        if pattern.search(md):
            md = pattern.sub(rf"\1\n\n{block}", md, count=1)
        else:
            md += "\n\n" + block

    page_md_path.write_text(md, encoding="utf-8")


def describe_page(
    run_dir: Path,
    page: int,
    *,
    model: str,
    prompt_version: str,
    hcfg: HeuristicConfig,
    client: OpenAI,
    max_images: int = 8,
    use_cache: bool = True,
    page_md_override: Path | None = None,
) -> List[VlmResult]:
    """
    Describe images on a given page using a vision-language model (VLM).
    This function reads the page text and images, pre-filters images using heuristics,
    checks for cached results, calls the VLM for uncached images, and injects the
    results back into the page's markdown file.
    """
    page_text = read_page_text(run_dir, page, max_chars=2000)
    elems = list_page_images(run_dir, page)
    if not elems:
        return []

    # Prefilter and truncate per page
    kept, _ = prefilter_elements(elems, hcfg)
    kept = kept[:max_images]

    # Load from cache when it is possible
    pending: List[Element] = []
    results: List[VlmResult] = []
    for e in kept:
        if use_cache:
            cache = _load_cached(run_dir, e.id)
            if (
                cache
                and cache.get("image_sha256") == e.sha256
                and cache.get("model") == model
                and cache.get("prompt_version") == prompt_version
            ):
                results.append(
                    VlmResult.model_validate(
                        {
                            "page": cache.get("page", page),
                            "image_id": cache["image_id"],
                            "classification": cache["classification"],
                            "text": cache.get("text", ""),
                            "confidence": cache.get(
                                "confidence",
                                0.7
                                if cache.get("classification") == "INFORMATIVE"
                                else 0.6,
                            ),
                            "model": cache["model"],
                            "prompt_version": cache["prompt_version"],
                            "image_sha256": cache["image_sha256"],
                        }
                    )
                )
                continue
        pending.append(e)

    # Call VLM if needed (simple: one image -> one described string)
    if pending:
        from .vlm_simple import describe_images_simple

        # TODO: Add page_text for context later
        pairs = describe_images_simple(
            client=client, model=model, page_text_hint=page_text, elements=pending
        )

        for image_id, text in pairs:
            el = next(e for e in pending if e.id == image_id)
            is_meaningless = (text or "").strip().upper() == "MEANINGLESS"
            rec = {
                "page": page,
                "image_id": image_id,
                "classification": "MEANINGLESS" if is_meaningless else "INFORMATIVE",
                "text": "" if is_meaningless else text.strip(),
                "confidence": 0.6 if is_meaningless else 0.7,
                "model": model,
                "prompt_version": prompt_version,
                "image_sha256": el.sha256,
            }
            _save_cached(run_dir, image_id, rec)
            results.append(VlmResult.model_validate(rec))

    # Inject back into markdown
    page_md = page_md_override or (run_dir / f"page_{page:04d}.md")
    inject_explanations(page_md, results)

    return results

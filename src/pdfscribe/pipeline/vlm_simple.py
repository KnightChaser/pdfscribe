# src/pdfscribe/pipeline/vlm_simple.py
from __future__ import annotations
import base64
import re
from typing import List, Tuple
from openai import OpenAI
from ..models.schema import Element

SYSTEM_PROMPT = (
    "You are a precise, terse image describer.\n"
    "If the image is decorative, a logo, gradient, page background, stock photo, or otherwise not informative for a report, reply EXACTLY: MEANINGLESS\n"
    "Otherwise reply with 1–2 sentences describing the key information in the image (chart trend, main entities, gist). "
    "Do NOT add preambles, labels, JSON, quotes, or extra text."
)


def _to_data_url(path: str) -> str:
    """
    Convert a PNG image file to a data URL for embedding in the prompt.
    """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def describe_images_simple(
    client: OpenAI,
    model: str,
    page_text_hint: str,
    elements: List[Element],
) -> List[Tuple[str, str]]:  # returns list of (image_id, text_or_MEANINGLESS)
    """
    Send ONE image at a time (simplest + most predictable).
    Returns a list of (image_id, raw_text) where raw_text is either 'MEANINGLESS'
    or a 1–2 sentence description.
    """
    out: List[Tuple[str, str]] = []
    for el in elements:
        user_content = [
            {"type": "text", "text": "Page context:\n" + page_text_hint[:2000]},
            {"type": "text", "text": "Analyze this image per the rules."},
            {"type": "image_url", "image_url": {"url": _to_data_url(el.path)}},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,  # determinism
            max_tokens=200,  # enough for 1–2 sentences
        )
        text = (resp.choices[0].message.content or "").strip()
        # heal common nuisances (some models wrap in code fences or quotes)
        if text.startswith("```"):
            text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.S).strip()
        text = text.strip('"').strip()
        out.append((el.id, text))
    return out

# src/pdfscribe/pipeline/vlm_simple.py
from __future__ import annotations
import base64
import re
from typing import List, Tuple
from openai import OpenAI
from ..models.schema import Element
from ..rate_limit import TokenLimiter, image_token_cost, approx_text_tokens

# Prompt: strict two-mode behavior (MEANINGLESS vs detailed text)
SYSTEM_PROMPT = (
    "You are a precise, no-nonsense image describer for technical documents.\n"
    "\n"
    "OUTPUT RULES (very important):\n"
    "1) If the image is decorative or not informative for a report (e.g., logo, divider, gradient, page background, stock photo with no data), reply EXACTLY: MEANINGLESS\n"
    "   - Must be UPPERCASE, no punctuation, no extra text, no quotes.\n"
    "2) Otherwise, reply with FREE TEXT ONLY (no labels, no JSON), describing the visible information as faithfully as possible.\n"
    "   - If the image is a chart/graph/infographic/table snapshot, enumerate ALL clearly visible numeric values verbatim, with units and labels exactly as shown.\n"
    "   - Prefer a concise sentence first, then a compact enumeration (e.g., “2017: 12,000; 2018: 15,993; 2019: 20,019”).\n"
    "   - Do NOT invent, infer, or guess values. If something is unreadable or cut off, omit it (do not estimate).\n"
    "   - Keep it factual and compact. Avoid hedging words like “maybe”, “appears”, “likely”.\n"
    "\n"
    "FORMAT CONSTRAINTS:\n"
    "- Your response must be either exactly MEANINGLESS, or a single paragraph of plain text.\n"
    "- No preambles, no code fences, no markdown headings, no quotes.\n"
)


def _to_data_url(path: str) -> str:
    """
    Convert a PNG image file to a data URL for embedding in the prompt.
    """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# tiny sanitizer to enforce our strict output contract
_CODEFENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.S)
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def _clean(text: str) -> str:
    """
    Clean and normalize model output text.
    Strips code fences, surrounding quotes, collapses whitespace.
    Normalizes "MEANINGLESS" to exact uppercase.
    """
    s = (text or "").strip()
    if not s:
        return "MEANINGLESS"  # fallback

    # Code block
    if s.startswith("```"):
        s = _CODEFENCE_RE.sub("", s).strip()
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        s = s[1:-1].strip()
    s = _WS_RE.sub(" ", s).strip()

    # Final checks (enforce exact MEANINGLESS)
    if not s:
        return "MEANINGLESS"
    if s.upper() == "MEANINGLESS":
        return "MEANINGLESS"
    return s


def describe_images_simple(
    client: OpenAI,
    model: str,
    page_text_hint: str,
    elements: List[Element],
    limiter: TokenLimiter | None = None,
) -> List[Tuple[str, str]]:  # returns list of (image_id, text_or_MEANINGLESS)
    """
    One-image-per-call for predictability.
    Returns either "MEANINGLESS" (verbatim) or a single-paragraph free-text description
    that enumerates numeric info verbatim when present.
    """
    out: List[Tuple[str, str]] = []
    # NOTE: Trim page hint a bit; it's only a hint, not an excuse to hallucinate... :)
    hint = (page_text_hint or "")[:1600]

    # NOTE: Token accounting (safe-side estimates)
    sys_tokens = approx_text_tokens(SYSTEM_PROMPT)
    user_fixed_tokens = approx_text_tokens(
        "Page context (do not guess from it; use only if it clarifies terms):"
    ) + approx_text_tokens(
        "Analyze this image under the OUTPUT RULES and FORMAT CONSTRAINTS."
    )
    hint_tokens = approx_text_tokens(hint)

    for el in elements:
        # NOTE: Estimate token consumption for the current request
        img_tokens = image_token_cost(el.width, el.height, model=model, detail="high")
        est_request_tokens = sys_tokens + user_fixed_tokens + hint_tokens + img_tokens

        # Apply the rate limit (block) if needed
        if limiter is not None:
            limiter.wait_for_budget(est_request_tokens)

        user_content = [
            {
                "type": "text",
                "text": "Page context (do not guess from it; use only if it clarifies terms):\n"
                + hint,
            },
            {
                "type": "text",
                "text": "Analyze this image under the OUTPUT RULES and FORMAT CONSTRAINTS.",
            },
            {"type": "image_url", "image_url": {"url": _to_data_url(el.path)}},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,  # for determinism
            top_p=1,
            max_tokens=1000,  # allow enumerating many labels/values by restricting max token count per image
        )

        raw = resp.choices[0].message.content or ""
        text = _clean(raw)

        # NOTE:Prefer actual usage accounting if the API returns it
        used_input = getattr(resp, "usage", None)
        total_used = 0
        if used_input and hasattr(used_input, "total_tokens"):
            # OpenAI may not always include vision token accounting; be defensive.
            total_used = int(used_input.total_tokens or 0)
        else:
            # Fallback: estimate request + a guess for output tokens
            out_tokens = approx_text_tokens(text if text != "MEANINGLESS" else "")
            total_used = est_request_tokens + out_tokens

        if limiter is not None:
            limiter.spend(total_used)

        if text == "MEANINGLESS":
            out.append((el.id, "MEANINGLESS"))
        else:
            out.append((el.id, text.replace("\n", " ").strip()))

    return out

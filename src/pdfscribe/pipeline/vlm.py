# src/pdfscribe/pipeline/vlm.py (Corrected Version)
from __future__ import annotations
import base64
import json
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from ..models.schema import VlmResult, Element

SYSTEM_PROMPT = (
    "You are a meticulous technical editor. You output ONLY strict JSON matching the schema.\n"
    "Never invent numbers. If labels are unreadable, mark classification 'UNCERTAIN' and confidence <= 0.49.\n"
    "Prefer facts consistent with the supplied page text. If contradiction exists, mention in 'uncertainty_notes'."
)


def _img_to_b64_data_url(path: str) -> str:
    """
    Convert an image file to a base64-encoded data URL for embedding in the prompt.
    """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    # The format for chat completions is slightly different
    return f"data:image/png;base64,{b64}"


def build_user_content(
    page_text: str,
    schema: Dict[str, Any],
    items: List[Tuple[str, str, str]],  # (image_id, abs_path, sha256)
) -> List[Dict[str, Any]]:
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    header = (
        f"PAGE_TEXT:\n<<<\n{page_text}\n>>>\n\n"
        "Analyze N images from this page. For each, return ONE JSON object exactly matching the schema.\n"
        "Classify decorative backgrounds/logos as 'MEANINGLESS'.\n"
        "Return a JSON array containing the results, in the same order as the images below.\n\n"
        f"SCHEMA (verbatim):\n{schema_json}\n\nIMAGES:\n"
    )
    # The user content for chat completions is a list of parts (text and images)
    content: List[Dict[str, Any]] = [{"type": "text", "text": header}]
    for idx, (image_id, path, sha) in enumerate(items, start=1):
        content.append(
            {"type": "text", "text": f"{idx}) image_id={image_id} sha256={sha}"}
        )
        content.append(
            {"type": "image_url", "image_url": {"url": _img_to_b64_data_url(path)}}
        )
    return content


def call_openai_json(
    client: OpenAI,
    model: str,
    page: int,
    page_text: str,
    elements: List[Element],
    prompt_version: str,
) -> List[VlmResult]:
    schema = {
        "page": 0,
        "image_id": "string",
        "classification": "INFORMATIVE | MEANINGLESS | UNCERTAIN",
        "kind": "photo | logo | flowchart | bar_chart | line_chart | pie_chart | table_snapshot | illustration | other",
        "title_guess": "string | null",
        "one_sentence": "string",
        "bullets": ["string"],
        "numbers_present": True,
        "quoted_values": [{"label": "string", "value": "string"}],
        "gfm_table": "string | null",
        "uncertainty_notes": "string | null",
        "confidence": 0.0,
    }

    items = [(e.id, e.path, e.sha256) for e in elements]
    user_content = build_user_content(page_text, schema, items)

    # Use the standard client.chat.completions.create endpoint
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_object"
        },  # This ensures the output is a valid JSON string
        temperature=0.1,  # Lower temperature for more deterministic, factual output
        max_tokens=4096,  # Set a reasonable max token limit
    )

    raw = resp.choices[0].message.content
    if not raw:
        raise ValueError("VLM returned an empty response")

    data = json.loads(raw)

    # Expect either a JSON array or a root object with a key like "results"
    arr = data if isinstance(data, list) else data.get("results")
    if not isinstance(arr, list):
        raise ValueError(
            f"VLM did not return a JSON array in the expected format. Got: {data}"
        )

    if len(arr) != len(elements):
        raise ValueError(
            f"VLM returned {len(arr)} results but was given {len(elements)} images."
        )

    out: List[VlmResult] = []
    for obj, el in zip(arr, elements):
        obj["page"] = page
        obj["image_id"] = el.id
        obj["model"] = model
        obj["prompt_version"] = prompt_version
        obj["image_sha256"] = el.sha256
        # Use a try-except block for more robust parsing
        try:
            out.append(VlmResult.model_validate(obj))
        except Exception as e:
            print(
                f"Warning: Failed to validate VLM output for image {el.id}. Error: {e}. Skipping."
            )
            continue
    return out

# src/pdfscribe/models/schema.py
from __future__ import annotations
from typing import Literal, Optional, Tuple, List, Dict
from pydantic import BaseModel, Field

BBox = Tuple[int, int, int, int]


class VlmResult(BaseModel):
    """
    Result of vision-language model (VLM) analysis on a single image.
    """

    page: int
    image_id: str
    classification: Literal["INFORMATIVE", "MEANINGLESS", "UNCERTAIN"]
    kind: Literal[
        "photo",
        "logo",
        "flowchart",
        "bar_chart",
        "line_chart",
        "pie_chart",
        "table_snapshot",
        "illustration",
        "other",
    ] = "other"
    title_guess: Optional[str] = None
    one_sentence: str
    bullets: List[str] = Field(default_factory=list)
    numbers_present: bool = False
    quoted_values: List[Dict[str, str]] = Field(default_factory=list)
    gfm_table: Optional[str] = None
    uncertainty_notes: Optional[str] = None
    confidence: float = 0.0
    # cache/meta
    model: str
    prompt_version: str
    image_sha256: str


class Element(BaseModel):
    """
    A single element on a page, such as text, an image, or a figure.
    """

    id: str
    page: int
    path: str  # absolute path to the PNG
    sha256: str
    width: int
    height: int

# src/pdfscribe/models/schema.py
from __future__ import annotations
from typing import Literal, Tuple
from pydantic import BaseModel

BBox = Tuple[int, int, int, int]


class VlmResult(BaseModel):
    """
    Result of vision-language model (VLM) analysis on a single image.
    """

    page: int
    image_id: str
    classification: Literal["INFORMATIVE", "MEANINGLESS"]
    text: str = ""  # Either "MEANINGLESS" or the free description
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

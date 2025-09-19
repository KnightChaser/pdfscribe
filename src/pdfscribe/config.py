# src/pdfscribe/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class DoclingConfig:
    """
    A set of configuration options for document processing.
    """

    images_scale: float = 2.0
    generate_picture_images: bool = True
    generate_page_images: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        """
        return asdict(self)
